'''
Module for creating dataset task inputs automatically,
by inspecting the code with specific rules.
'''
import ast
import json
import re
import black
from datasets import load_dataset

import pandas as pd

from staticfg import Block, CFG, CFGBuilder
from tqdm import tqdm

from dataset import DREval
from dynamics import FunctionFactory, ClassFactory, States, Sandbox

ASSERT_PATTERN = r"^assert\s*\(*\s*(\w+)\s*\((.*)\)\s*\)*\s*==\s*(.+)$"
INVOCATION_TEMPLATE = "{function_name}{args}"

# In basic blocks, we are only interested in the following statements
WANTED_STMTS = (ast.Assign, ast.AugAssign, ast.AnnAssign, 
                ast.Return, ast.Expr)
# In standalone expressions (not in statements), 
# we don't want to inspect the following types
EXCLUDE_EXPRS = (ast.Constant,)

ASSERTS = ['assertEqual', 'assertNotEqual', 'assertAlmostEqual', 
           'assertTrue', 'assertFalse', 'assertIsNone',  'assertIsNotNone',
           'assertIn', 'assertNotIn', ]

# Manual patch for `staticfg`
def visit_Call_patched(self, node):
    def visit_func(node):
        if type(node) == ast.Name:
            return node.id
        elif type(node) == ast.Attribute:
            # Recursion on series of calls to attributes.
            func_name = visit_func(node.value)
            if func_name is not None:
                func_name += "." + node.attr
            else:
                func_name = "unknown_attribute"
            return func_name
        elif type(node) == ast.Constant:
            return node.s
        elif type(node) == ast.Subscript:
            if type(node.value) == ast.Attribute:
                return node.value.attr
            else:
                return node.value.id
        else:
            return type(node).__name__
    
    func = node.func
    func_name = visit_func(func)
    self.current_block.func_calls.append(func_name)
    
CFGBuilder.visit_Call = visit_Call_patched

def build_cfg(code) -> CFG:
    '''
    Build the control flow graph from the given code.
    '''
    try:
        return CFGBuilder().build_from_src('<string>', code)
    except Exception as e:
        import traceback

        print(f'Internal library error: {e}')
        traceback.print_exc()
        print('---------')
        print(code)
        exit(1)

def check_skip_values(value: ast.expr) -> bool:
    '''
    We skip the assignment, if RHS is:
    - a constant
    - a collection of constants
    - an empty collection
    '''
    if isinstance(value, ast.Constant):
        return True
    if isinstance(value, ast.List) and len(value.elts) == 0:
        return True
    if isinstance(value, ast.Tuple) and len(value.elts) == 0:
        return True
    if isinstance(value, ast.Dict) and len(value.keys) == 0:
        return True
    if isinstance(value, ast.Set) and len(value.elts) == 0:
        return True
    if isinstance(value, ast.List) or isinstance(value, ast.Tuple) or isinstance(value, ast.Set):
        if all(isinstance(elt, ast.Constant) for elt in value.elts):
            return True
    return False

def check_general(stmt: ast.stmt) -> bool:
    '''
    We skip the statement, if it is:
    - not in the wanted types
    - an expression but in the exclude types
    '''
    if not isinstance(stmt, WANTED_STMTS):
        return True
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, EXCLUDE_EXPRS):
        return True
    return False

def inspect_execution(code) -> set[int]:
    '''
    Inspect the code, analyze the control flow graph,
    and then recommend the line numbers to be included
    in the `coverage` and `next execution line` tasks.

    The returned line numbers are 1-indexed.
    '''
    stmts: list[ast.stmt] = []
    cfg = build_cfg(code)
    for block in cfg:
        block: Block
        # We prioritize the last statement in the block
        # as it makes the `next execution line` task more
        # challenging.
        for stmt in reversed(block.statements):
            stmt: ast.stmt
            if check_general(stmt):
                continue
            stmts.append(stmt)
            break
    return set(map(lambda s: s.lineno, stmts))

def classeval_var_adhoc(vars: list[tuple[int, str]]) -> list[tuple[int, str]]:
    '''
    Ad-hoc rule for ClassEval,
    array subscripts should not be function calls.
    '''
    for i in range(len(vars)):
        lineno, var = vars[i]
        if var == 'self._data[self._convert_key(key)]':
            vars[i] = (lineno, 'self._data')
    return vars

def inspect_variable(code, states: States) -> set[tuple[int, str]]:
    '''
    Inspect the code, analyze the statements, and then
    recommend the line numbers and the variable names to
    be included in the `program state` task.
    
    The returned line numbers are 1-indexed.
    '''
    variables: list[tuple[int, str]] = []
    cfg = build_cfg(code)
    for block in cfg:
        block: Block
        for stmt in block.statements:
            stmt: ast.stmt
            if check_general(stmt):
                continue
            # For assignment, we extract the variable(s) in LHS.
            # Possible types are: identifier, subscript, attribute, etc.
            # Here we just stringify the LHS, and the interpretation
            # of complex names (e.g., self.xxx, arr[...]) will be 
            # handled by the evaluation module. In most cases, the 
            # names are just identifiers.
            if isinstance(stmt, ast.Assign):
                # Skip naive cases like `a = 0, b = [], etc.`
                if check_skip_values(stmt.value):
                    continue
                for target in stmt.targets:
                    name = ast.unparse(target).strip()
                    if name != '_':
                        variables.append((stmt.lineno, name))
            elif isinstance(stmt, ast.AugAssign) or isinstance(stmt, ast.AnnAssign):
                # Treat naive ann_assign, e.g., `a: int = 0`, similarly
                # But don't skip aug_assign, e.g., `a += 1`, which is useful
                if isinstance(stmt, ast.AnnAssign) and check_skip_values(stmt.value):
                    continue
                name = ast.unparse(stmt.target).strip()
                if name != '_':
                    variables.append((stmt.lineno, name))
            # For return statements, if local variables are returned, 
            # we extract the name of the variables.
            elif isinstance(stmt, ast.Return):
                if isinstance(stmt.value, ast.Name):
                    variables.append((stmt.lineno, stmt.value.id))
                elif isinstance(stmt.value, ast.Tuple):
                    for name in stmt.value.elts:
                        if isinstance(name, ast.Name):
                            variables.append((stmt.lineno, name.id))
                # If the return value is a constant, we select the
                # nearest name in the `variables` list
                elif isinstance(stmt.value, ast.Constant):
                    for (lineno, name) in reversed(variables):
                        if lineno < stmt.lineno:
                            variables.append((stmt.lineno, name))
                            break
            # For other situations, if some variables after the current line
            # are changed, we extract the names of the changed variables.
            elif isinstance(stmt, ast.Expr):
                lineno = stmt.lineno - 1 # to 0-indexed
                before = states.get_states_before(lineno)
                after = states.get_states_after(lineno)
                names = set()
                for (s1, s2) in zip(before, after):
                    s1_locals = s1.locals
                    s2_locals = s2.locals
                    # 1. new variables ?
                    for name in s2_locals.keys() - s1_locals.keys():
                        names.add(name)
                    # 2. changed variables ?
                    for name in s1_locals.keys() & s2_locals.keys():
                        try:
                            if s1_locals[name] != s2_locals[name]:
                                names.add(name)
                        except ValueError:
                            # just ignore the numpy arrays
                            pass
                    # 3. changed attributes (self.xxx) ?
                    if 'self' in s1_locals.keys() & s2_locals.keys():
                        s1_self = s1_locals['self'].__dict__
                        s2_self = s2_locals['self'].__dict__
                        for name in s1_self.keys() & s2_self.keys():
                            try:
                                if s1_self[name] != s2_self[name]:
                                    names.add(f'self.{name}')
                            except ValueError:
                                # just ignore the numpy arrays
                                pass
                for name in names:
                    # the self object is hard to express and compare
                    if name == 'self':
                        continue
                    # use the 1-indexed lineno
                    variables.append((stmt.lineno, name))
            else:
                raise RuntimeError('unreachable')
    variables = classeval_var_adhoc(variables)
    return set(variables)

def inspect_test(test_code):
    '''
    For ClassEval only, inspect the test code,
    select an assert statement, and replace the
    expected value with `??`. 
    '''
    tree = ast.parse(test_code)
    assert_exprs: list[ast.Call] = []
    for stmt in tree.body:
        if isinstance(stmt, ast.Expr):
            expr = stmt.value
            if isinstance(expr, ast.Call):
                if isinstance(expr.func, ast.Name) and expr.func.id in ASSERTS:
                    assert_exprs.append(expr)
    assert_exprs = sorted(assert_exprs, key=lambda x: ASSERTS.index(x.func.id))
    if len(assert_exprs) == 0:
        return None
    for expr in assert_exprs:
        idx = 1 if len(expr.args) >= 2 else 0
        expr.args[idx] = ast.Name(id='??')
    return ast.unparse(tree)


def parse_assert_statement(assert_statement):
    match = re.match(ASSERT_PATTERN, assert_statement.strip())
    if not match:
        print(assert_statement)
        raise ValueError(f"Invalid assert statement format. {assert_statement}")

    function_name = match.group(1)
    args_str = match.group(2)
    args_str = f"({args_str})"
    expected_result_str = match.group(3)
    # args = ast.literal_eval(f"({args_str})") if args_str else ()
    # expected_result = ast.literal_eval(expected_result_str)

    return (function_name, args_str, expected_result_str)


def process_mbpp_dataset():
    mbpp_res = []
    mbpp_data = []
    empty_tasks = []
    invalid_tasks = []
    valid_tasks = []
    ds_full = load_dataset("google-research-datasets/mbpp", "full")
    for idx, d in enumerate(tqdm(ds_full['test'])):
        idx += DREval.MBPP_START
        if idx in (266, 265, 210):
            continue
        if idx in (272,276, 285, 438, 475, 483, 541, 562):
            continue
        item = {'task_id': f'DREval/{idx}', 'idx': idx, 'tasks': []}
        code = d["code"].replace("\r\n", "\n")
        code = black.format_str(code, mode=black.Mode(line_length=120))
        #test_list = d["test_list"] + d["challenge_test_list"]
        test_list = d["test_list"]
        inputs = []
        innvocations = []
        outputs = []
        fn_names = []

        test_setup = d["test_setup_code"].replace("\r\n", "\n")
        if test_setup:
            print(f"Skipping test setups for now")
            continue

        for input_idx, test_assert_statment in enumerate(test_list):
            try:
                if len(item['tasks']) >= DREval.MAX_INPUTS:
                    break
                print(f"Start Processing Task {idx}, Test {input_idx}")
                try:
                    fn_name, _input, _ = parse_assert_statement(test_assert_statment)
                except Exception as e:
                    print(f"Skipping test due to error: {e}")
                
                invocation = INVOCATION_TEMPLATE.format(function_name=fn_name, args=_input)
                invocation = black.format_str(invocation, mode=black.Mode(line_length=120))
                fn = FunctionFactory.create(fn_name, code)
                sandbox = Sandbox(fn)
                s1 = inspect_execution(code)
                for i in range(2):
                    try:
                        _output, states = sandbox.run(*eval(_input))
                    except TypeError:
                        old_inputs = _input
                        _input = _input.replace(')', ',)')
                        print(f'#2 Fix inputs: {old_inputs} -> {_input}')
                        continue

                    if 'exception' in sandbox.status:
                        if i == 0:
                            old_inputs = _input
                            _input = f"[{_input},]"
                            print()
                            print(f'#1 Fix inputs: {old_inputs} -> {_input}')
                
                assert sandbox.status == 'ok', f'Error: {sandbox.status} caused by DREval/{idx}: {fn_name}{_input}'
                inputs.append(_input)
                fn_names.append(fn_name)
                outputs.append(_output)
                innvocations.append(invocation)

                s2 = inspect_variable(code, states)
                s = s1 & set(map(lambda x: x[0], s2))
                s = list(map(lambda x: (x, list(filter(lambda y: y[0] == x, s2))[0][1]), s))
                task = [{'lineno': lineno, 'var': var} for lineno, var in s]
                if len(task) > 0:
                    item['tasks'].append({'input_idx': input_idx, 'task': task, 'output_pred': f'assert {invocation}) == ??'})
                    print(f"Finished Processing Task {idx}, Test {input_idx}")
                    valid_tasks.append((idx, input_idx))
                else:
                    print(f"Empty Processing Task {idx}, Test {input_idx}")
                    empty_tasks.append((idx, input_idx))

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Skipping Task {idx}, Test {input_idx}")
                invalid_tasks.append((idx, input_idx))
            print()
        
        # make sure fn_name is consistent
        # 
        fn_names = set(fn_names)
        assert len(fn_names) == 1, f"Different function names: {fn_names}"
        fn_name = list(fn_names)[0]
        data_entry = {
            'task_id': item['task_id'],
            'code': code,
            'entry_point': fn_name,
            'inputs': inputs,
            'outputs': outputs,
            'innvocations': innvocations,
        }
        import pprint
        pprint.pprint(data_entry)
        mbpp_data.append(data_entry)
        mbpp_res.append(item)
    
    print(f"Valid tasks: {len(valid_tasks)}")
    print(len(set([idx for idx, _ in valid_tasks])))

    print(f"Invalid tasks: {len(invalid_tasks)}")
    print(len(set([idx for idx, _ in invalid_tasks])))

    print(f"Empty tasks: {len(empty_tasks)}")
    print(len(set([idx for idx, _ in empty_tasks])))

    print(f"Total tasks: {len(ds_full['test'])}")
    print(len(set([idx for idx, _ in enumerate(ds_full['test'])])))

    final_mbpp_data = []
    final_mbpp_res  = []
    for data, item in zip(mbpp_data, mbpp_res):
        try:
            json.dumps(data)
        except:
            continue
        if not len(item['tasks']):
            continue

        final_mbpp_data.append(data)
        final_mbpp_res.append(item)

    with open('data/DREval_tasks_mbpp.black.jsonl', 'w') as f:
        f.writelines([json.dumps(r) + '\n' for r in final_mbpp_res])

    with open('data/DREval_data_mbpp.black.jsonl', 'w') as f:
        f.writelines([json.dumps(r) + '\n' for r in final_mbpp_data])

    
# Note: The generated task data might change very slightly (e.g., lineno order) 
# after re-run due to the `set`s in the implementation.
def process_dataset():
    with open('data/DREval_data.jsonl', 'r') as f:
        df = pd.read_json(f, lines=True).to_dict(orient='records')
    
    res = []
    for idx, d in enumerate(tqdm(df)):
        item = {'task_id': f'DREval/{idx}', 'idx': idx, 'tasks': []}
        if DREval.HUMANEVAL_START <= idx <= DREval.HUMANEVAL_END:
            code = d['code']
            fn_name = d['entry_point']
            inputs = d['inputs']
            fn = FunctionFactory.create(fn_name, code)
            sandbox = Sandbox(fn)
            s1 = inspect_execution(code)
            for input_idx, _input in enumerate(inputs):
                if len(item['tasks']) >= DREval.MAX_INPUTS:
                    break
                _, states = sandbox.run(*eval(_input))
                assert sandbox.status == 'ok', f'Error: {sandbox.status} caused by DREval/{idx}: {fn_name}{_input}'
                s2 = inspect_variable(code, states)
                s = s1 & set(map(lambda x: x[0], s2))
                s = list(map(lambda x: (x, list(filter(lambda y: y[0] == x, s2))[0][1]), s))
                task = [{'lineno': lineno, 'var': var} for lineno, var in s]
                if len(task) > 0:
                    item['tasks'].append({'input_idx': input_idx, 'task': task, 'output_pred': f'assert {fn_name}{_input[:-2]}) == ??'})
        elif DREval.CLASSEVAL_START <= idx <= DREval.CLASSEVAL_END:
            cls_code = d['code']
            cls_name = d['entry_point']
            test_code = d['test']
            ClassFactory.create(cls_name, cls_code)
            test_classes = ClassFactory.create_test_classes(cls_name, cls_code, test_code, 
                                                            DREval.tcls_pattern, 
                                                            DREval.tcls_validation, 
                                                            DREval.tcls_postprocess)
            assert len(test_classes) == len(d['inputs'])
            s1 = inspect_execution(cls_code)
            for input_idx, tcls in enumerate(test_classes):
                if len(item['tasks']) >= DREval.MAX_INPUTS:
                    break
                output_pred = inspect_test(d['inputs'][input_idx])
                if output_pred is None:
                    continue
                obj = tcls()
                if hasattr(obj, 'setUp'):
                    obj.setUp()
                sandbox = Sandbox(obj.dreval_test)
                _, states = sandbox.run()
                assert sandbox.status == 'ok', f'{sandbox.status} caused by DREval/{idx}, code:\n{d["inputs"][input_idx]}'
                s2 = inspect_variable(cls_code, states)
                s = s1 & set(map(lambda x: x[0], s2))
                s = list(map(lambda x: (x, list(filter(lambda y: y[0] == x, s2))[0][1]), s))
                task = [{'lineno': lineno, 'var': var} for lineno, var in s]
                if len(task) > 0:
                    item['tasks'].append({'input_idx': input_idx, 'task': task, 'output_pred': output_pred})
        else:
            raise RuntimeError('unreachable')
        res.append(item)
    
    with open('data/DREval_tasks.jsonl', 'w') as f:
        f.writelines([json.dumps(r) + '\n' for r in res])

if __name__ == '__main__':
    #process_dataset()
    process_mbpp_dataset()
