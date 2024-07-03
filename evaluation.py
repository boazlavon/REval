import argparse
import glob
import inspect
import os
import re
import json
import sys

import numpy as np
import pandas as pd
import pytz

from datetime import datetime
from pydoc import locate

from bullet import Bullet, Input
from tqdm import tqdm

from inference import Model
from dataset import DREval
from dynamics import Nil, _NilType, FunctionFactory, ClassFactory, Sandbox
from prompt import build_direct_prompt, build_cot_prompt

def get_time():
    return datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%y-%m-%d-%H-%M")

def is_builtin_type(cls) -> bool:
    if cls is None:
        raise ValueError(f'invalid type {cls}')
    assert inspect.isclass(cls), f'Use a class instead of instance: {cls}'
    return cls.__module__ == 'builtins'

class Task:
    def __init__(self, name: str, model: Model, prompt_type: str, **kwargs):
        self.name = name
        self.model = model
        self.prompt_type = prompt_type
        assert prompt_type in ['direct', 'cot'], 'Use a valid prompt type: direct, cot'
        assert self.model.prompt_type == prompt_type, 'Model prompt type must match task prompt type'
        self.data = pd.read_json(f'data/DREval_data.jsonl', lines=True).to_dict('records')
        self.task_data = pd.read_json(f'data/DREval_tasks.jsonl', lines=True).to_dict('records')
        self.records = []
    
    def _get_code(self, idx) -> str:
        return self.data[idx]['code']
    
    def _get_entry_point(self, idx) -> str:
        return self.data[idx]['entry_point']

    def _get_inputs(self, idx) -> str:
        return self.data[idx]['inputs']
    
    def _get(self, idx, key) -> str:
        return self.data[idx][key]
    
    def _build_prompt(self, **kwargs):
        if self.prompt_type == 'direct':
            return build_direct_prompt(self.name, **kwargs)
        else:
            assert self.prompt_type == 'cot'
            return build_cot_prompt(self.name, **kwargs)
    
    def _prompt_model(self, prompt: str):
        resp = self.model.infer(prompt)
        return self._postprocess(resp), resp

    def _postprocess(self, resp: str):
        raise NotImplementedError()
    
    def _humaneval_task_impl(self, fn_name, code, task, sandbox: Sandbox, _input):
        raise NotImplementedError()

    def _classeval_task_impl(self, test_cls, task, _input):
        raise NotImplementedError()

    @property
    def _metrics(self):
        raise NotImplementedError()
    
    @property
    def _save_path(self):
        return f'model_generations/{self.name}@{self.model.info}'
    
    def run(self):
        os.makedirs(self._save_path, exist_ok=True)
        for task in tqdm(self.task_data):
            idx = task['idx']
            pairs = task['tasks']
            self.records.append({'task_id': f'DREval/{idx}', 'generation': []})
            if DREval.HUMANEVAL_START <= idx <= DREval.HUMANEVAL_END:
                code = self._get_code(idx)
                fn_name = self._get_entry_point(idx)
                fn = FunctionFactory.create(fn_name, code)
                sandbox = Sandbox(fn)
                inputs = self._get_inputs(idx)
                for pair in pairs:
                    if self.__class__.__name__ == 'Output':
                        _input = pair['output_pred']
                    else:
                        _input = inputs[pair['input_idx']]
                    res = self._humaneval_task_impl(fn_name, code, pair['task'], sandbox, _input)
                    self.records[-1]['generation'].append({'input_idx': pair['input_idx'], 'results': res})
            elif DREval.CLASSEVAL_START <= idx <= DREval.CLASSEVAL_END:
                cls_code = self._get_code(idx)
                cls_name = self._get_entry_point(idx)
                test_code = self._get(idx, 'test')
                ClassFactory.create(cls_name, cls_code)
                test_classes = ClassFactory.create_test_classes(cls_name, cls_code, test_code,
                                                                DREval.tcls_pattern,
                                                                DREval.tcls_validation,
                                                                DREval.tcls_postprocess)
                inputs = self._get_inputs(idx)
                for pair in pairs:
                    test_cls = test_classes[pair['input_idx']]
                    if self.__class__.__name__ == 'Output':
                        _input = pair['output_pred']
                    else:
                        _input = inputs[pair['input_idx']]
                    res = self._classeval_task_impl(test_cls, pair['task'], _input)
                    self.records[-1]['generation'].append({'input_idx': pair['input_idx'], 'results': res})
            else:
                raise ValueError(f'Invalid data index: {idx}')
        self.records.append(self._metrics)
        print(self._metrics)
        with open(f'{self._save_path}/{get_time()}.jsonl', 'w+') as f:
            f.writelines([json.dumps(item) + '\n' for item in self.records])

class Coverage(Task):
    def __init__(self, model: Model, prompt_type='direct', **kwargs):
        super().__init__('coverage', model, prompt_type)
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def _acc(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
    
    def _prec(self):
        return self.tp / (self.tp + self.fp)
    
    def _rec(self):
        return self.tp / (self.tp + self.fn)
    
    def _f1(self):
        if self._prec() + self._rec() == 0:
            return 0
        return 2 * (self._prec() * self._rec()) / (self._prec() + self._rec())

    @property
    def _metrics(self):
        return {
            'acc': self._acc(),
            'prec': self._prec(),
            'rec': self._rec(),
            'f1': self._f1(),
        }

    def _postprocess(self, resp: str) -> bool:
        ans = resp.upper().strip()
        if self.prompt_type == 'cot' and '[/THOUGHT]' not in ans:
            print(f'Warning: CoT not completed or exceeded max length, assuming `NO`')
            return False
        idx = ans.find('[ANSWER]')
        if idx != -1:
            ans = ans[idx+8:].strip()
        idx = ans.find('[/ANSWER]')
        if idx != -1:
            ans = ans[:idx].strip()
        idx = ans.find('[/ANSWER')
        if idx != -1:
            ans = ans[:idx].strip()
        if ans == '':
            print('Warning: Empty response, assuming `NO`')
            return False
        ans = ans[:3]
        b1 = 'YES' in ans
        b2 = 'NO' in ans
        if b1 == b2:
            print(f'Warning: Ambiguous response `{ans}`, assuming `NO`')
            return False
        if b1:
            return True
        if b2:
            return False
        raise RuntimeError('unreachable')
    
    def _update_metrics(self, ans, actual):
        if ans and actual:       # (True, True)
            self.tp += 1
        elif ans and not actual: # (True, False)
            self.fp += 1
        elif not ans and actual: # (False, True)
            self.fn += 1
        else:                    # (False, False)
            self.tn += 1
    
    def _humaneval_task_impl(self, fn_name, code, task, sandbox, _input):
        _, states = sandbox.run(*eval(_input))
        assert sandbox.status == 'ok', f'Error: {sandbox.status} caused by {fn_name}{_input}'
        invocation = f'{fn_name}{_input[:-2]})'
        codelines = code.split('\n')
        gens = []
        for t in task:
            line = t['lineno']
            codeline = codelines[line-1] # to 0-indexed
            p = self._build_prompt(code=code, invocation=invocation, 
                                    invocation_abbr=invocation, 
                                    line=line, codeline=codeline)
            ans, model_gen = self._prompt_model(p)
            actual = states.get_coverage(line-1) # to 0-indexed
            gens.append({'generated': model_gen, 'response': ans, 'expected': actual})
            self._update_metrics(ans, actual)
        return gens

    def _classeval_task_impl(self, test_cls, task, _input):
        obj = test_cls()
        if hasattr(obj, 'setUp'):
            obj.setUp()
            setup = '\n# setup code executed previously\n' + '\n# '.join(test_cls.__setup__.split('\n')[1:])
            if 'Hook method for setting up the test fixture' in setup:
                setup = ''
        else:
            setup = ''
        sandbox = Sandbox(obj.dreval_test)
        _, states = sandbox.run()
        assert sandbox.status == 'ok', f'{sandbox.status} caused by code:\n{_input}'
        code = obj.dreval_test.__doc__
        codelines = code.split('\n')
        gens = []
        for t in task:
            line = t['lineno']
            codeline = codelines[line-1] # to 0-indexed
            p = self._build_prompt(code=code, invocation=setup + '\n' + _input.rstrip(), 
                                    invocation_abbr='the above test code', 
                                    line=line, codeline=codeline)
            ans, model_gen = self._prompt_model(p)
            actual = states.get_coverage(line-1) # to 0-indexed
            gens.append({'generated': model_gen, 'response': ans, 'expected': actual})
            self._update_metrics(ans, actual)
        return gens

class Path(Task):
    def __init__(self, model: Model, prompt_type='direct', **kwargs):
        super().__init__('path', model, prompt_type)
        self._correct = 0
        self._total = 0

    @property
    def _metrics(self):
        return {'acc': self._correct / self._total}
    
    def _update_metrics(self, ans, actual):
        self._total += 1
        if any(a in actual for a in ans):
            self._correct += 1

    def _postprocess(self, resp: str):
        if self.prompt_type == 'cot' and '[/THOUGHT]' not in resp:
            print(f'Warning: CoT not completed or exceeded max length, assuming `-2`')
            return -2
        idx = resp.find('[ANSWER]')
        if idx != -1:
            resp = resp[idx+8:].strip()
        idx = resp.find('[/ANSWER]')
        if idx != -1:
            resp = resp[:idx].strip()
        idx = resp.find('[/ANSWER')
        if idx != -1:
            resp = resp[:idx].strip()
        resp = resp.split('\n')[0].strip()
        if resp == '':
            return -2
        elif resp == '-1':
            return -1
        else:
            return resp

    def _humaneval_task_impl(self, fn_name, code, task, sandbox: Sandbox, _input):
        _, states = sandbox.run(*eval(_input))
        assert sandbox.status == 'ok', f'Error: {sandbox.status} caused by {fn_name}{_input}'
        invocation = f'{fn_name}{_input[:-2]})'
        codelines = code.split('\n')
        code = ''.join([str(i+1) + '\t' + codelines[i] + '\n' for i in range(len(codelines))])
        gens = []
        for t in task:
            line = t['lineno']
            codeline = codelines[line-1] # to 0-indexed
            p = self._build_prompt(code=code, invocation=invocation, 
                                    invocation_abbr=invocation, 
                                    line=line, codeline=codeline)
            ans, model_gen = self._prompt_model(p)
            ans_to_lines = []
            if type(ans) == int:
                ans_to_lines.append(ans)
            else:
                for i, _line in enumerate(codelines):
                    if ans == _line.strip():
                        ans_to_lines.append(i+1)
                if len(ans_to_lines) == 0:
                    ans_to_lines.append(-2)
            _actual = states.get_next_line(line-1) # to 0-indexed
            actual = []
            for a in _actual:
                if a == -1:
                    actual.append(-1)
                else:
                    actual.append(a+1)
            gens.append({'generated': model_gen, 'response': ans_to_lines, 'expected': actual})
            self._update_metrics(ans_to_lines, actual)
        return gens
    
    def _classeval_task_impl(self, test_cls, task, _input):
        obj = test_cls()
        if hasattr(obj, 'setUp'):
            obj.setUp()
            setup = '\n# setup code executed previously\n# ' + '\n# '.join(test_cls.__setup__.split('\n')[1:])
            if 'Hook method for setting up the test fixture' in setup:
                setup = ''
        else:
            setup = ''
        sandbox = Sandbox(obj.dreval_test)
        _, states = sandbox.run()
        assert sandbox.status == 'ok', f'{sandbox.status} caused by code:\n{_input}'
        code = obj.dreval_test.__doc__
        codelines = code.split('\n')
        gens = []
        for t in task:
            line = t['lineno']
            codeline = codelines[line-1] # to 0-indexed
            p = self._build_prompt(code=code, invocation=setup + '\n' + _input.rstrip(), 
                                    invocation_abbr='the above test code', 
                                    line=line, codeline=codeline)
            ans, model_gen = self._prompt_model(p)
            ans_to_lines = []
            if type(ans) == int:
                ans_to_lines.append(ans)
            else:
                for i, _line in enumerate(codelines):
                    if ans == _line.strip():
                        ans_to_lines.append(i+1)
                if len(ans_to_lines) == 0:
                    ans_to_lines.append(-2)
            _actual = states.get_next_line(line-1) # to 0-indexed
            actual = []
            for a in _actual:
                if a == -1:
                    actual.append(-1)
                else:
                    actual.append(a+1)
            gens.append({'generated': model_gen, 'response': ans_to_lines, 'expected': actual})
            self._update_metrics(ans_to_lines, actual)
        return gens

class State(Task):
    def __init__(self, model: Model, prompt_type='direct', **kwargs):
        super().__init__('state', model, prompt_type)
        self._correct = 0
        self._total = 0

    @property
    def _metrics(self):
        return {'acc': self._correct / self._total}

    def _update_metrics(self, ans, actual):
        self._total += 1
        if ans == 'ERROR':
            return False
        if self._eq(ans, actual):
            self._correct += 1
            return True
        else:
            return False

    def _eq(self, ans: _NilType | tuple[any, type], actual: list):
        if ans is Nil and actual is Nil:
            return True
        if ans is Nil or actual is Nil:
            return False
        ans_val, ans_type = ans
        actual_types = [type(a) for a in actual]
        # None of the actual types match the generated type
        if all(ans_type != t for t in actual_types):
            return False
        # generated value conflicts with generated type
        if type(ans_val) != ans_type:
            return False
        # start value comparison
        if ans_type == float:
            for a in actual:
                try:
                    if abs(ans_val - a) < 1e-6:
                        return True
                except Exception:
                    continue
            return False
        else:
            try:
                return ans_val in actual
            except ValueError:
                # dirty hack for numpy arrays
                for a in actual:
                    try:
                        if isinstance(ans_val, np.ndarray) and isinstance(a, np.ndarray):
                            if np.allclose(ans_val, a):
                                return True
                        else:
                            if ans_val == a:
                                return True
                    except Exception:
                        continue
                return False
    
    def _postprocess(self, resp: str):
        '''
        Postprocess model generation and extract (value, type) pair
        Also apply reasonable fixes for common types.
        '''
        if self.prompt_type == 'cot' and '[/THOUGHT]' not in resp:
            print(f'Warning: CoT not completed or exceeded max length, assuming `ERROR`')
            return 'ERROR'
        # replace unicode quotes with ascii quotes
        resp = resp.replace('\u2018', "'").replace('\u2019', "'")
        resp = resp.replace('\u201c', '"').replace('\u201d', '"')
        resp = resp.strip()
        idx = resp.find('[ANSWER]')
        if idx != -1:
            resp = resp[idx+8:].strip()
        idx = resp.find('[/ANSWER]')
        if idx != -1:
            resp = resp[:idx].strip()
        idx = resp.find('[/ANSWER')
        if idx != -1:
            resp = resp[:idx].strip()
        # Case: model generates `Nil` directly
        if resp.capitalize() == 'Nil' or resp == '[Nil]':
            return Nil
        semicolon = resp.rfind(';')
        # Case: no semicolon found, return `ERROR`
        if semicolon == -1:
            return 'ERROR'
        parts = [resp[:semicolon].strip(), resp[semicolon+1:].strip().lower()]
        # Case: model generates `Nil` as value
        if parts[0].capitalize() == 'Nil':
            return Nil
        # Case: generated type format is <class '...'>, unwrap class name
        _match = re.match(r"<class '(.*)'>", parts[1])
        if _match:
            parts[1] = _match.group(1)
        # Case: remove generics
        _match = re.match(r"(.*)\[.*\]", parts[1])
        if _match:
            parts[1] = _match.group(1)
        # Ad-hoc fixes
        if parts[1] == 'string':
            parts[1] = 'str'
        if parts[1] == 'integer':
            parts[1] = 'int'
        if ',' in parts[1] or 'tuple' in parts[1]:
            parts[1] = 'tuple'
        # Ad-hoc Case #1: type is `str`, first try to unquote the string with `eval`,
        # if it fails, return the string as is
        if parts[1] == 'str':
            try:
                return eval(parts[0]), type('')
            except Exception as e:
                return parts[0], type('')
        # Ad-hoc Case #2: type is `datetime.datetime`, 
        # use `dateutil.parser.parse` to parse the string
        if parts[1] == 'datetime.datetime' or parts[1] == 'datetime':
            from dateutil.parser import parse
            try:
                return parse(parts[0]), locate(parts[1])
            except Exception:
                return 'ERROR'
        # Ad-hoc Case #3: type is `numpy.ndarray`
        if parts[1] == 'numpy.ndarray' or parts[1] == 'np.ndarray':
            import numpy as np
            try:
                return np.array(eval(parts[0])), locate(parts[1])
            except Exception:
                return 'ERROR'
        # Ad-hoc Case #4: val is `None` or type is `NoneType`, return `None`
        if parts[0] == 'None' or parts[1] == 'NoneType':
            return None, type(None)
        # General situation, try to `locate` the type and parse the value.
        # For builtin types, use `eval` directly. 
        # For other types, use the corresponding type constructor.
        try:
            _type = locate(parts[1])
            if is_builtin_type(_type):
                _val = eval(parts[0])
            else:
                try:
                    _val = _type(eval(parts[0]))
                except Exception:
                    _val = _type(parts[0])
            return _val, _type
        except Exception:
            return 'ERROR'
    
    def _humaneval_task_impl(self, fn_name, code, task, sandbox: Sandbox, _input):
        _, states = sandbox.run(*eval(_input))
        assert sandbox.status == 'ok', f'Error: {sandbox.status} caused by {fn_name}{_input}'
        invocation = f'{fn_name}{_input[:-2]})'
        codelines = code.split('\n')
        gens = []
        for t in task:
            line = t['lineno']
            var = t['var']
            codeline = codelines[line-1] # to 0-indexed
            p = self._build_prompt(code=code, invocation=invocation, 
                                    invocation_abbr=invocation, 
                                    line=line, codeline=codeline, var=var)
            ans, model_gen = self._prompt_model(p)
            actual = states.interpret_var(line-1, var) # to 0-indexed
            res = self._update_metrics(ans, actual)
            gens.append({'generated': model_gen, 'eq': res})
        return gens

    def _classeval_task_impl(self, test_cls, task, _input):
        obj = test_cls()
        if hasattr(obj, 'setUp'):
            obj.setUp()
            setup = '\n# setup code executed previously\n' + '\n# '.join(test_cls.__setup__.split('\n')[1:])
            if 'Hook method for setting up the test fixture' in setup:
                setup = ''
        else:
            setup = ''
        sandbox = Sandbox(obj.dreval_test)
        _, states = sandbox.run()
        assert sandbox.status == 'ok', f'{sandbox.status} caused by code:\n{_input}'
        code = obj.dreval_test.__doc__
        codelines = code.split('\n')
        gens = []
        for t in task:
            line = t['lineno']
            var = t['var']
            codeline = codelines[line-1] # to 0-indexed
            p = self._build_prompt(code=code, invocation=setup + '\n' + _input.rstrip(), 
                                    invocation_abbr='the above test code', 
                                    line=line, codeline=codeline, var=var)
            ans, model_gen = self._prompt_model(p)
            actual = states.interpret_var(line-1, var) # to 0-indexed
            res = self._update_metrics(ans, actual)
            gens.append({'generated': model_gen, 'eq': res})
        return gens

def penalty_pattern(code: str, _input: str) -> bool:
    '''
    Mark generated test code as failed if it contains any of the following patterns:
    - Assert on `True` or `False`
    - Less number of asserts than given question
    '''
    if 'assertTrue(True)' in code or 'assertFalse(False)' in code \
        or 'assert True' in code or 'assert False' in code \
        or 'assert True == True' in code or 'assert False == False' in code:
        return True
    given_asserts_num = _input.count('assert')
    assert given_asserts_num > 0
    gen_asserts_num = code.count('assert')
    if gen_asserts_num < given_asserts_num:
        return True
    return False

class Output(Task):
    def __init__(self, model: Model, prompt_type='direct', **kwargs):
        super().__init__('output', model, prompt_type)
        self._total = 0
        self._pass = 0
    
    @property
    def _metrics(self):
        return {'acc': self._pass / self._total}
    
    def _update_metrics(self, status):
        self._total += 1
        if status:
            self._pass += 1
    
    def _postprocess(self, resp: str):
        if self.prompt_type == 'cot' and '[/THOUGHT]' not in resp:
            print(f'Warning: CoT not completed or exceeded max length, assuming `ERROR`')
            return 'ERROR'
        idx = resp.find('[ANSWER]')
        if idx != -1:
            resp = resp[idx+8:].strip()
        idx = resp.find('[/ANSWER]')
        if idx != -1:
            resp = resp[:idx].strip()
        # weird case with local codellama,
        # the stop words do not function properly
        idx = resp.find('[/ANSWER')
        if idx != -1:
            resp = resp[:idx].strip()
        return resp
    
    def _postprocess_phase2(self, resp: str, _input: str):
        if resp == 'ERROR':
            return 'assert False'
        in_lines = _input.strip().split('\n')
        res_lines = resp.strip().split('\n')
        if len(res_lines) >= len(in_lines):
            return resp
        else:
            diff = len(in_lines) - len(res_lines)
            # pad resp with the first `diff` lines of _input
            # pad before the first line of resp
            return '\n'.join(in_lines[:diff] + res_lines)

    def _humaneval_task_impl(self, fn_name, code, task, sandbox: Sandbox, _input):
        # make `fn_name` callable in current scope
        locals()[fn_name] = FunctionFactory.create(fn_name, code)
        p = self._build_prompt(code=code, invocation='\n' + _input)
        ans, model_gen = self._prompt_model(p)
        ans = self._postprocess_phase2(ans, _input)
        status = False
        if not penalty_pattern(ans, _input):
            try:
                exec(ans)
                status = True
            except Exception:
                pass
        self._update_metrics(status)
        return [{'generated': model_gen, 'pass': status}]
    
    def _classeval_task_impl(self, test_cls, task, _input):
        if hasattr(test_cls, 'setUp'):
            setup = '\n# setup code executed previously\n# ' + '\n# '.join(test_cls.__setup__.split('\n')[1:])
            if 'Hook method for setting up the test fixture' in setup:
                setup = ''
        else:
            setup = ''
        prelude = '\n# Test code starts here. Only write the completed test code in your answer.\n'
        p = self._build_prompt(code=test_cls.__doc__, invocation=setup + prelude + _input)
        ans, model_gen = self._prompt_model(p)
        ans = self._postprocess_phase2(ans, _input)
        status = False
        if not penalty_pattern(ans, _input):
            try:
                fn = FunctionFactory.create_from_answer(ans, test_cls)
                setattr(test_cls, 'dreval_output_pred', fn)
                obj = test_cls()
                if hasattr(obj, 'setUp'):
                    obj.setUp()
                obj.dreval_output_pred()
                status = True
            except Exception:
                pass
        self._update_metrics(status)
        return [{'generated': model_gen, 'pass': status}]

class Consistency(Task):
    def __init__(self, model: Model, prompt_type='direct', **kwargs):
        super().__init__('consistency', model, prompt_type)
        self.task_paths = [f'model_generations/{task}@{self.model.info}' for task in ['coverage', 'state', 'path', 'output']]
        self.generation_logs = []
        for task_path in self.task_paths:
            # find the latest jsonl file in this path
            file_path = max(glob.glob(f'{task_path}/*.jsonl'), key=os.path.getctime)
            print(f'Load {file_path}')
            df = pd.read_json(file_path, lines=True).to_dict('records')
            self.generation_logs.append(df)
        assert len(self.generation_logs) == 4

    def _count_statistics(self, task_idx, rule):
        l = []
        for i, task_log in enumerate(self.generation_logs[task_idx]):
            if i == len(self.generation_logs[task_idx]) - 1:
                break
            for input_log in task_log['generation']:
                for atomic_log in input_log['results']:
                    test = rule(atomic_log)
                    assert type(test) == bool
                    l.append(test)
        return l

    def run(self):
        coverage = self._count_statistics(0, lambda x: x['response'] == x['expected'])
        state = self._count_statistics(1, lambda x: x['eq'])
        path = self._count_statistics(2, lambda x: any(y in x['expected'] for y in x['response']))
        output = []
        for i,task_log in enumerate(self.generation_logs[3]):
            if i == len(self.generation_logs[3]) - 1:
                break
            for j,input_log in enumerate(task_log['generation']):
                atomic_log = input_log['results'][0]
                repeats = len(self.generation_logs[0][i]['generation'][j]['results'])
                output.extend([atomic_log['pass']] * repeats)
        assert len(coverage) == len(path) == len(state) == len(output)
        total = len(coverage)
        score = 0
        for i in range(total):
            if coverage[i] and state[i] and path[i] and output[i]:
                score += 1
            elif coverage[i] and state[i] and path[i] and not output[i]:
                score += 0.5
            elif coverage[i] and state[i] and not path[i] and not output[i]:
                score += 0.25
            elif coverage[i] and not state[i] and not path[i] and not output[i]:
                score += 0.125
        print(f'Consistency score: {100 * score/total}')

class Cli:
    def __init__(self):
        self.kwargs = {}
        self._bullet_kwargs = {'indent': 0, 'align': 5, 'margin': 2, 'shift': 0, 'bullet': '\u27f6'}

    def get_input(self):
        import readline, glob
        def complete(text, state):
            return (glob.glob(os.path.expanduser(text)+'*')+[None])[state]

        readline.set_completer_delims('\t\n;')
        readline.parse_and_bind("tab: complete")
        readline.set_completer(complete)
        cli = Bullet(
            prompt='Select a task:',
            choices=['coverage', 'path', 'state', 'output', 'consistency'],
            **self._bullet_kwargs
        )
        self.kwargs['task'] = cli.launch()
        cli = Bullet(
            prompt='Select prompt type:',
            choices=['direct', 'cot'],
            **self._bullet_kwargs
        )
        self.kwargs['prompt_type'] = cli.launch()
        cli = Bullet(
            prompt='Select model type:',
            choices=['OpenAI', 'HuggingFace'],
            **self._bullet_kwargs
        )
        model_type = cli.launch()
        if model_type == 'OpenAI':
            cli = Bullet(
                prompt='Select a model:',
                choices=['gpt-3.5', 'gpt-4'],
                **self._bullet_kwargs
            )
            self.kwargs['model_id'] = cli.launch()
        else:
            cli = Bullet(
                prompt='Select deployment type:',
                choices=['Python Instance', 'Local API Server'],
                **self._bullet_kwargs
            )
            deploy_type = cli.launch()
            if deploy_type == 'Local API Server':
                cli = Input(prompt='Enter port number: ', default=3000, strip=True)
                self.kwargs['port'] = int(cli.launch())
            cli = Input(prompt='Enter model name: ', strip=True)
            self.kwargs['model_id'] = cli.launch()
            path = input('Enter model path: ')
            self.kwargs['model_path'] = path
            cli = Input(prompt='Enter number of GPUs to use: ', default=1, strip=True)
            self.kwargs['num_gpus'] = int(cli.launch())
            assert self.kwargs['num_gpus'] > 0, 'At least one GPU is required'
            default_devices = ','.join([str(i) for i in range(self.kwargs['num_gpus'])])
            cli = Input(prompt='Set `CUDA_VISIBLE_DEVICES`: ', default=default_devices, strip=True)
            ordinals = cli.launch().split(',')
            self.kwargs['gpu_ordinals'] = [int(_ord) for _ord in ordinals]
        cli = Input(prompt='Set temperature: ', default=0.8, strip=True)
        self.kwargs['temp'] = float(cli.launch())
    
    @staticmethod
    def config(save_path='.eval_config'):
        cli = Cli()
        cli.get_input()
        with open(save_path, 'w+') as f:
            f.write(json.dumps(cli.kwargs))
            print(f'Configuration saved to {save_path}')

    def _run(self):
        print(f'The arguments for this run: {self.kwargs}')
        if self.kwargs['task'] == 'consistency':
            self.kwargs['mock'] = True
        model = Model.new(**self.kwargs)
        self.kwargs['model'] = model
        task = getattr(sys.modules[__name__], self.kwargs['task'].capitalize())(**self.kwargs)
        task.run()

    @staticmethod
    def run_with_config(load_path='.eval_config'):
        cli = Cli()
        if not os.path.exists(load_path):
            print(f'Error: {load_path} file not found')
            sys.exit(1)
        with open(load_path, 'r') as f:
            cli.kwargs = json.load(f)
        cli._run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run evaluation for DREval tasks')
    parser.add_argument('command', nargs='?', type=str, default='run', choices=['config', 'run'], help='Command to run')
    parser.add_argument('-i', '--input', type=str, default='.eval_config', help='specify configuration file to load')
    parser.add_argument('-o', '--output', type=str, default='.eval_config', help='specify configuration file to save')
    args = parser.parse_args()
    if args.command == 'config':
        Cli.config(args.output)
        sys.exit(0)
    elif args.command == 'run':
        Cli.run_with_config(args.input)
        sys.exit(0)
    else:
        raise RuntimeError('unreachable')
