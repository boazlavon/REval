import ast
import inspect
import itertools
import sys
import tempfile

import execution

from copy import deepcopy
from types import FrameType
from typing import Callable, Iterator, Generator, Type

__all__ = ['FunctionFactory', 'ClassFactory', 'Sandbox', 'Nil']

class Factory:
    @staticmethod
    def filename():
        '''
        Used in runtime code compilation.
        The special filename marks which code to trace and collect states from.
        '''
        return '<internals>'
    
class FunctionFactory(Factory):
    @staticmethod
    def create(fn_name: str, code: str) -> Callable:
        pyobj = compile(code, Factory.filename(), 'exec')
        exec(pyobj, globals())
        func = globals()[fn_name]
        assert inspect.isroutine(func), f"{func} is not a function"
        setattr(func, '__doc__', code)
        return func
    
    @staticmethod
    def create_from_answer(generated: str, test_class: Type) -> Callable:
        test_fn_code = 'def dreval_output_pred(self):'
        in_string_block = False
        for line in generated.split('\n'):
            if in_string_block:
                test_fn_code += f'\n{line}'
            else:
                test_fn_code += f'\n\t{line}'
            if "'''" in line or '"""' in line:
                in_string_block = not in_string_block
        test_fn_code = test_fn_code.replace('assert', 'self.assert')
        fn = FunctionFactory.create('dreval_output_pred', test_fn_code)
        fn.__doc__ = test_class.__doc__
        setattr(test_class, 'dreval_output_pred', fn)
        return fn
    
class ClassFactory(Factory):
    @staticmethod
    def create(cls_name: str, code: str) -> Type:
        '''
        This method compiles the code and creates a class 
        in the global scope. The class is then returned.
        Note: Only the class is created, no instances are created.
        '''
        pyobj = compile(code, Factory.filename(), 'exec')
        exec(pyobj, globals())
        cls = globals()[cls_name]
        assert isinstance(cls, type), f"{cls} is not a class"
        setattr(cls, '__doc__', code)
        return cls
    
    @staticmethod
    def create_test_classes(cls_name: str, code: str, test_code: str,
                            test_cls_name_pattern: Callable[[str, str], bool],
                            test_cls_validation: Callable[[Type], bool],
                            test_cls_postprocess: Callable[[Type], Type]=None) -> list[Type]:
        '''
        The method compiles the test code and creates all test classes
        in the global scope. The classes are then returned after postprocessing.

        This method support subclasses of `unittest.TestCase` only.
        For standalone test methods, use `FunctionFactory.create` instead.
        '''
        test_classes = []
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py') as f:
            f.write(test_code)
            f.flush()

            pyobj = compile(test_code, f.name, 'exec')
            exec(pyobj, globals())
            for k, v in globals().items():
                if isinstance(v, type) and test_cls_name_pattern(k, cls_name) \
                    and test_cls_validation(v):
                    v.__doc__ = code
                    if test_cls_postprocess is not None:
                        test_cls_postprocess(v)
                    test_classes.append(v)
        return test_classes

_SANDBOX = None

def add_state(line, event, value):
    global _SANDBOX
    if _SANDBOX is None:
        return
    # make lineno 0-indexed
    _SANDBOX.add_state(line - 1, event, value)

def local_trace(frame: FrameType, event: str, arg):
    if event == 'line':
        # arg = None
        _locals = {}
        for k, v in frame.f_locals.items():
            # skip non-serializable local variables
            if v.__class__.__name__ == 'module':
                continue
            if v.__class__.__name__ == 'function':
                continue
            if v.__class__.__name__ == 'builtin_function_or_method':
                continue
            if isinstance(v, Iterator):
                continue
            if isinstance(v, Generator):
                continue
            _locals[k] = deepcopy(v)
        add_state(frame.f_lineno, 'locals', _locals)
    elif event == 'return':
        # arg = return value
        add_state(frame.f_lineno, 'return', arg)
    elif event == 'exception':
        # arg = (exception_type, exception_value, traceback)
        add_state(frame.f_lineno, 'exception', arg[0])
    else:
        return

def global_trace(frame: FrameType, event: str, _):
    if event != 'call':
        return
    if frame.f_code.co_filename == Factory.filename():
        return local_trace
    return

class _NilType:
    '''
    Class used to create a singleton `Nil`, to avoid conflicts with `None`.
    `Nil` acts like None when None might be a valid value.
    Note: Using `object.__new__(_NilType)` is prohibited in any other 
    part of the code.
    '''
    def __new__(cls):
        return Nil
    def __reduce__(self):
        return (_NilType, ())
    def __copy__(self):
        return Nil
    def __deepcopy__(self, _):
        return Nil
    def __call__(self, _):
        pass
    def __repr__(self):
        return 'Nil'
    def __str__(self):
        return 'Nil'

try:
    Nil # type: ignore
except NameError:
    Nil = object.__new__(_NilType)

class VarInterpreter:
    def __init__(self, lineno, name, states: 'States'):
        self.lineno = lineno
        self.name = name
        self.states = states

    def _analyze_node(self, node: ast.Expr) -> list | _NilType:
        if isinstance(node, ast.Constant):
            return [node.value]
        elif isinstance(node, ast.Name):
            return self.states.get_local(self.lineno, node.id)
        elif isinstance(node, ast.Attribute):
            obj = self._analyze_node(node.value)
            if obj is Nil:
                return Nil
            res = []
            for o in obj:
                if hasattr(o, node.attr):
                    res.append(getattr(o, node.attr))
            if len(res) == 0:
                return Nil
            return res
        elif isinstance(node, ast.Subscript):
            arr = self._analyze_node(node.value)
            idx = self._analyze_node(node.slice)
            if arr is Nil or idx is Nil:
                return Nil
            combinations = list(itertools.product(arr, idx))
            res = []
            for a, i in combinations:
                try:
                    res.append(a[i])
                except (TypeError, KeyError, IndexError):
                    pass
            if len(res) == 0:
                return Nil
            return res
        elif isinstance(node, ast.Tuple):
            elts = [self._analyze_node(n) for n in node.elts]
            if any(e is Nil for e in elts):
                return Nil
            return list(itertools.product(*elts))
        else:
            return ValueError(f"Unsupported node type: {node}")
    
    def _analyze(self):
        if not self.states.get_coverage(self.lineno):
            return Nil
        tree = ast.parse(self.name)
        assert len(tree.body) == 1
        expr = tree.body[0]
        assert isinstance(expr, ast.Expr)
        node = expr.value
        return self._analyze_node(node)

    def get(self):
        try:
            return self._analyze()
        except Exception:
            return Nil

class State:
    def __init__(self, lineno: int, code: str):
        self.lineno = lineno
        self.code = code
        self.locals = {}
        self.return_ = Nil
        self.exception = Nil
    
    def __getitem__(self, key: str):
        if key == 'return':
            return self.return_
        return getattr(self, key)
    
    def __setitem__(self, key: str, value):
        if key == 'return':
            self.return_ = value
        else:
            setattr(self, key, value)
    
    def __str__(self) -> str:
        _state = {
            'lineno': self.lineno,
            'code': self.code,
            'locals': self.locals,
            'return': self.return_,
            'exception': self.exception
        }
        return f"{_state}"
    
    def __repr__(self) -> str:
        return str(self)
    
    def get_local(self, var: str):
        if var not in self.locals:
            return Nil
        return self.locals[var]
    
    def get_attr(self, var: str, attr: str):
        if var not in self.locals:
            return Nil
        obj = self.locals[var]
        if not hasattr(obj, attr):
            return Nil
        return getattr(obj, attr)

    def get_subscript(self, var: str, key: str):
        if var not in self.locals:
            return Nil
        obj = self.locals[var]
        try:
            return obj[eval(key)]
        except (TypeError, KeyError, IndexError):
            return Nil

    def to_json(self):
        keys = ['lineno', 'locals']
        if self.return_ is not Nil:
            keys.append('return')
        if self.exception is not Nil:
            keys.append('exception')
        d = {k: self[k] for k in keys}
        # convert special types to serializable types
        for k, v in d['locals'].items():
            if isinstance(v, set):
                d['locals'][k] = list(v)
        # convert exception to its class name
        if 'exception' in d:
            d['exception'] = d['exception'].__class__.__name__
        return d

class States:
    def __init__(self):
        self._states: list[State] = []

    def __getitem__(self, key: int):
        return self._states[key]
    
    def __len__(self):
        return len(self._states)
    
    def __str__(self) -> str:
        return str(self._states)
    
    def __repr__(self) -> str:
        return str(self)
    
    def append(self, state: State):
        self._states.append(state)

    @property
    def trace(self):
        return [state.lineno for state in self._states]

    def get_coverage(self, lineno: int):
        return lineno in self.trace
    
    def get_next_line(self, lineno: int) -> set[int]:
        if not self.get_coverage(lineno):
            return {-1}
        lines = []
        state_idxs = [i for i, state in enumerate(self._states) if state.lineno == lineno]
        for idx in state_idxs:
            if idx + 1 < len(self._states):
                lines.append(self._states[idx + 1].lineno)
            else:
                lines.append(-1)
        return set(lines)
    
    def get_states_before(self, lineno: int) -> list[State]:
        return [state for state in self._states if state.lineno == lineno]
    
    def get_states_after(self, lineno: int) -> list[State]:
        '''
        As `sys.settrace` calls the tracer before the line is executed,
        we need to look at the next executed line.
        This does not affect the lookup of return values and exceptions.
        Note: the next executed line is not necessarily the next line in the code.
        '''
        state_idxs = [i for i, state in enumerate(self._states) if state.lineno == lineno]
        states = []
        for idx in state_idxs:
            # if idx + 1 = len, then idx is the last state
            # its expected to be a return or exception
            # in these cases we don't need to look at the next state
            # to get the locals
            if idx + 1 < len(self._states):
                idx += 1
            states.append(self._states[idx])
        return states
    
    def get_local(self, lineno: int, var: str):
        '''
        For lines that are not executed, this function returns `Nil`,
        although `var` may be valid. This rule also applies to
        `get_attr` and `get_subscript`.
        '''
        vars = []
        for state in self.get_states_after(lineno):
            v = state.get_local(var)
            if v is not Nil:
                vars.append(v)
        if len(vars) == 0:
            return Nil
        return vars
    
    def get_attr(self, lineno: int, var: str, attr: str):
        attrs = []
        for state in self.get_states_after(lineno):
            v = state.get_attr(var, attr)
            if v is not Nil:
                attrs.append(v)
        if len(attrs) == 0:
            return Nil
        return attrs
    
    def get_subscript(self, lineno: int, var: str, key: str):
        keys = []
        for state in self.get_states_after(lineno):
            v = state.get_subscript(var, key)
            if v is not Nil:
                keys.append(v)
        if len(keys) == 0:
            return Nil
        return keys
    
    def interpret_var(self, lineno: int, name: str):
        return VarInterpreter(lineno, name, self).get()
    
    def get_return(self, lineno: int):
        l = [state.return_ for state in self._states if state.lineno == lineno and state.return_ != Nil]
        assert len(l) <= 1, f"Multiple return values found for line {lineno}: {l}"
        return l[0] if len(l) == 1 else Nil
    
    def get_exception(self, lineno: int):
        l = [state.exception for state in self._states if state.lineno == lineno and state.exception != Nil]
        assert len(l) <= 1, f"Multiple exceptions found for line {lineno}: {l}"
        return l[0] if len(l) == 1 else Nil

    def to_json(self):
        return [state.to_json() for state in self._states]

class Sandbox:
    def __init__(self, fn: Callable, timeout: float=120):
        self.fn = fn
        self.timeout = timeout
        self.result = Nil
        self.status = ''
        self.states = States()
        self._codelines = self.fn.__doc__.split('\n')

    def run(self, *args, **kwargs):
        global _SANDBOX
        _SANDBOX = self
        self.result = Nil
        self.status = ''
        self.states = States()

        try:
            with execution.swallow_io():
                with execution.time_limit(self.timeout):
                    sys.settrace(global_trace)
                    self.result = self.fn(*args, **kwargs)
                    sys.settrace(None)
            self.status = "ok"
        except execution.TimeoutException:
            self.status = "timed out"
        except BaseException as e:
            self.status = f"exception: {e}"
            import traceback
            traceback.print_exc()

        return self.result, self.states

    def add_state(self, line, event, value):
        # if new line, create new state
        if len(self.states) == 0 or self.states[-1].lineno != line:
            new_state = State(line, self._codelines[line])
            new_state[event] = value    
            self.states.append(new_state)
        # if same line, update existing state
        else:
            self.states[-1][event] = value
