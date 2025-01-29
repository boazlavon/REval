import argparse
import traceback
import black
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

from trace_of_thoughts_parser import TraceOfThoughtsParser, VALID_TASKS, RE_TASK__STATE, RE_TASK__PATH, RE_TASK__COVERAGE
def get_time():
    return datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%y-%m-%d-%H-%M")

def is_builtin_type(cls) -> bool:
    if cls is None:
        raise ValueError(f'invalid type {cls}')
    assert inspect.isclass(cls), f'Use a class instead of instance: {cls}'
    return cls.__module__ == 'builtins'

TRACE_OF_THOUGHTS_PROMPT_TYPE = 'tot'
VALID_PROMPT_TYPES = ['direct', 'cot', TRACE_OF_THOUGHTS_PROMPT_TYPE]
class Task:
    def __init__(self, name: str, model: Model, prompt_type: str, **kwargs):
        self.name = name
        self.model = model
        self.prompt_type = prompt_type
        self.kwargs = kwargs
        assert prompt_type in VALID_PROMPT_TYPES, f'Use a valid prompt type: {VALID_PROMPT_TYPES}'
        self.mock = False
        if ('custom_mock' in kwargs) and kwargs['custom_mock']:
            self.mock = True
        if not self.mock and prompt_type != TRACE_OF_THOUGHTS_PROMPT_TYPE:
            assert self.model.prompt_type == prompt_type, 'Model prompt type must match task prompt type'
        self.dataset = self.kwargs['dataset']
        assert self.dataset in ['mbpp', 'humaneval', 'classeval']
        if prompt_type == TRACE_OF_THOUGHTS_PROMPT_TYPE:
            self.model_id = kwargs['model_id']
            with open('.tot_config', 'r') as f:
                tot_kwargs = json.load(f)
            basedir = tot_kwargs['base_dir']
            # dataset = tot_kwargs['dataset']
            inference_output_dir = tot_kwargs['inference_output_dir']
            self.tot_parser = TraceOfThoughtsParser(basedir, self.dataset, inference_output_dir)
        # data_path = f'data/DREval_data.jsonl'
        # task_path = f'data/DREval_tasks.jsonl'
        data_path = f'data/DREval_data_mbpp.black.jsonl'
        task_path = f'data/DREval_tasks_mbpp.black.jsonl'
        print(f"Data path: {data_path}")
        print(f"Task path: {task_path}")
        self.data = pd.read_json(data_path, lines=True).to_dict('records')
        self.task_data = pd.read_json(task_path, lines=True).to_dict('records')
        self.records = []
        self._valid_test_cases = []
        valid_test_cases_path =  kwargs.get('valid_test_cases_path')
        self._no_skip_test_cases = []
        if valid_test_cases_path:
            with open(valid_test_cases_path, 'r') as f:
                self._no_skip_test_cases = json.loads(f.read())
    
    def _get_code(self, idx) -> str:
        return self._get(idx, 'code')
    
    def _get_entry_point(self, idx) -> str:
        return self._get(idx, 'entry_point')

    def _get_inputs(self, idx) -> str:
        return self._get(idx, 'inputs')

    def _get_innvocations(self, idx) -> str:
        return self._get(idx, 'innvocations')
    
    def _get(self, idx, key) -> str:
        for entry in self.data:
            if entry['task_id'] == f'DREval/{idx}':
                return entry[key]
    
    def _build_prompt(self, **kwargs):
        if self.prompt_type == 'direct':
            return build_direct_prompt(self.name, **kwargs)
        elif self.prompt_type == 'cot':
            return build_cot_prompt(self.name, **kwargs)
        elif self.prompt_type == TRACE_OF_THOUGHTS_PROMPT_TYPE:
            return None
        else:
            raise ValueError(f"Invalid Prompt Type: {self.prompt_type}")
    
    def _prompt_model(self, prompt: str):
        resp = self.model.infer(prompt)
        return self._postprocess(resp), resp

    def _postprocess(self, resp: str):
        raise NotImplementedError()
    
    def _humaneval_task_impl(self, fn_name, code, task, sandbox: Sandbox, _input, invocation=None, mbpp_task_idx=None, input_idx=None):
        raise NotImplementedError()

    def _classeval_task_impl(self, test_cls, task, _input):
        raise NotImplementedError()

    @property
    def _metrics(self):
        raise NotImplementedError()
    
    @property
    def _save_path(self):
        name = self.name
        if self.mock:
            model_info = f'mock_model_{self.prompt_type}'
        elif self.prompt_type == TRACE_OF_THOUGHTS_PROMPT_TYPE:
            model_info = f"{self.model_id}_{TRACE_OF_THOUGHTS_PROMPT_TYPE}"
        else:
            model_info = self.model.info
        # result = os.path.join('model_generations', f'{name}@{model_info}'.replace('/', '_'))
        result = os.path.join('model_generations', f'{name}@{model_info}')
        return result
    
    def run(self):
        os.makedirs(self._save_path, exist_ok=True)
        for task in tqdm(self.task_data):
            idx = task['idx']
            pairs = task['tasks']
            self.records.append({'task_id': f'DREval/{idx}', 'generation': []})
            if DREval.HUMANEVAL_START <= idx <= DREval.HUMANEVAL_END:
                if not (self.dataset == 'humaneval'):
                    continue
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
                if not (self.dataset == 'classeval'):
                    continue
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
            elif DREval.MBPP_START <= idx <= DREval.MBPP_END:
                if not (self.dataset == 'mbpp'):
                    continue
                mbpp_task_idx = (idx - DREval.MBPP_START) + 11
                code = self._get_code(idx)
                fn_name = self._get_entry_point(idx)
                fn = FunctionFactory.create(fn_name, code)
                sandbox = Sandbox(fn)
                inputs = self._get_inputs(idx)
                innvocations = self._get_innvocations(idx)

                for pair in pairs:
                    if self.__class__.__name__ == 'Output':
                        _input = pair['output_pred']
                    else:
                        _input = inputs[pair['input_idx']]

                    _invocation = innvocations[pair['input_idx']]
                    res = self._humaneval_task_impl(fn_name, code, pair['task'], sandbox, _input, invocation=_invocation, mbpp_task_idx=mbpp_task_idx, input_idx=pair['input_idx'])
                    self.records[-1]['generation'].append({'input_idx': pair['input_idx'], 'results': res})
            else:
                raise ValueError(f'Invalid data index: {idx}')
        self.records.append(self._metrics)
        with open(f'{self._save_path}/{get_time()}.{self.dataset}.jsonl', 'w+') as f:
            f.writelines([json.dumps(item) + '\n' for item in self.records])
        if self.prompt_type == TRACE_OF_THOUGHTS_PROMPT_TYPE:
            print(f"Valid test cases: {len(self._valid_test_cases)}")
            with open(f'{self._save_path}/{get_time()}.valid_test_cases.{self.dataset}.json', 'w+') as f:
                f.write(json.dumps(self._valid_test_cases))
        print(self._metrics)
        print(self.prompt_type)
        print(self.kwargs)

class Coverage(Task):
    def __init__(self, model: Model, prompt_type='direct', **kwargs):
        super().__init__('coverage', model, prompt_type, **kwargs)
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self._total = 0

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
            'total': self._total,
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
        self._total += 1
        if ans and actual:       # (True, True)
            self.tp += 1
        elif ans and not actual: # (True, False)
            self.fp += 1
        elif not ans and actual: # (False, True)
            self.fn += 1
        else:                    # (False, False)
            self.tn += 1
    
    def _process_tot_task_impl(self, mbpp_task_idx, input_idx, line, code, invocation, use_labels):
        re_task_kwargs = {'line': line}
        invocation = black.format_str(invocation, mode=black.Mode(line_length=120))
        validation_kwargs = {'invocation': invocation, 'code': code}
        try:
            self.tot_parser.validate_task(mbpp_task_idx, input_idx, self.__class__.__name__, re_task_kwargs, validation_kwargs)
        except Exception as e:
            traceback_err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            raise ValidationError(traceback_err)
        
        ans, model_gen = self.tot_parser.process_task(mbpp_task_idx, input_idx, self.__class__.__name__, re_task_kwargs, validation_kwargs, use_labels)
        return ans, model_gen
    
    def _process_tot_task_impl_with_validation(self, actual, mbpp_task_idx, input_idx, line, code, invocation):
        is_valid_test_case = False
        try:
            ans, _ = self._process_tot_task_impl(mbpp_task_idx, input_idx, line, code, invocation, True)
            is_valid_test_case = (ans == actual)
        except Exception as e:
            pass
        
        if not is_valid_test_case:
            # print(f'Invalid test case {mbpp_task_idx}:{input_idx}, continue')
            return None
        
        self._valid_test_cases.append((mbpp_task_idx, input_idx, line))
        print(f'Valid test case {mbpp_task_idx}:{input_idx}')
        try:
            error = None
            ans, model_gen = self._process_tot_task_impl(mbpp_task_idx, input_idx, line, code, invocation, False)
        except ValidationError as e:
            # should not happen
            error = f'Validate task {mbpp_task_idx}:{input_idx} Error: {type(e)}'
            ans = 'VALIDATION_ERROR'
            ans = False
            model_gen = error
        except EmptyAnswerError as e:
            error = f'Task {mbpp_task_idx}:{input_idx} Returned an empty answer'
            print(error)
            ans = 'EMPTY_ANSWER_ERROR'
            ans = False
            model_gen = error
        except Exception as e:
            error = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            print(f'Process task Error: {type(e)} : {error}')
            ans = 'GENERAL_ERROR'
            ans = False
            model_gen = error
        return ans, model_gen

    def _humaneval_task_impl(self, fn_name, code, task, sandbox, _input, invocation=None, mbpp_task_idx=None, input_idx=None):
        _, states = sandbox.run(*eval(_input))
        assert sandbox.status == 'ok', f'Error: {sandbox.status} caused by {fn_name}{_input}'
        if invocation is None:
            invocation = f'{fn_name}{_input[:-2]})'
        codelines = code.split('\n')
        gens = []
        for t in task:
            line = t['lineno']
            if self._no_skip_test_cases and list((mbpp_task_idx, input_idx, line)) not in self._no_skip_test_cases:
                print(f'Skipping {(mbpp_task_idx, input_idx, line)}')
                continue
            actual = states.get_coverage(line-1) # to 0-indexed
            if self.prompt_type != TRACE_OF_THOUGHTS_PROMPT_TYPE:
                codeline = codelines[line-1] # to 0-indexed
                p = self._build_prompt(code=code, invocation=invocation, 
                                        invocation_abbr=invocation, 
                                        line=line, codeline=codeline)
                ans, model_gen = self._prompt_model(p)
            else:
                p = ''
                code = '\n'.join(codelines)
                tot_result = self._process_tot_task_impl_with_validation(actual, mbpp_task_idx, input_idx, line, code, invocation)
                if tot_result is None:
                    continue
                ans, model_gen = tot_result
            gens.append({'generated': model_gen, 'response': ans, 'expected': actual})
            self._update_metrics(ans, actual)
            print(ans, actual)
            print(self._acc())
            print()
            print(f"Acc: {self._acc()}, {self._total}")
            print()
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
        super().__init__('path', model, prompt_type, **kwargs)
        self._correct = 0
        self._total = 0

    @property
    def _metrics(self):
        return {
                'acc': self._correct / self._total,
                'correct': self._correct,
                'total': self._total
                }
    
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

    def _process_tot_task_impl(self, mbpp_task_idx, input_idx, line, code, invocation, use_labels):
        re_task_kwargs = {'line': line}
        invocation = black.format_str(invocation, mode=black.Mode(line_length=120))
        validation_kwargs = {'invocation': invocation, 'code': code}
        try:
            self.tot_parser.validate_task(mbpp_task_idx, input_idx, self.__class__.__name__, re_task_kwargs, validation_kwargs)
        except Exception as e:
            traceback_err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            raise ValidationError(traceback_err)
        
        ans_to_lines, model_gen = self.tot_parser.process_task(mbpp_task_idx, input_idx, self.__class__.__name__, re_task_kwargs, validation_kwargs, use_labels)
        if not ans_to_lines:
            raise EmptyAnswerError(f'task {mbpp_task_idx}:{input_idx} Returned an empty answer, Skip')
        return ans_to_lines, model_gen
    
    def _process_tot_task_impl_with_validation(self, actual, mbpp_task_idx, input_idx, line, code, invocation):
        is_valid_test_case = False
        try:
            ans_to_lines, _ = self._process_tot_task_impl(mbpp_task_idx, input_idx, line, code, invocation, True)
            is_valid_test_case = any(a in actual for a in ans_to_lines)
        except Exception as e:
            pass
        
        if not is_valid_test_case:
            # print(f'Invalid test case {mbpp_task_idx}:{input_idx}, continue')
            return None
        
        self._valid_test_cases.append((mbpp_task_idx, input_idx, line))
        print(f'Valid test case {mbpp_task_idx}:{input_idx}')
        try:
            error = None
            ans_to_lines, model_gen = self._process_tot_task_impl(mbpp_task_idx, input_idx, line, code, invocation, False)
        except ValidationError as e:
            # should not happen
            error = f'Validate task {mbpp_task_idx}:{input_idx} Error: {type(e)}'
            ans_to_lines = 'VALIDATION_ERROR'
            model_gen = error
        except EmptyAnswerError as e:
            error = f'Task {mbpp_task_idx}:{input_idx} Returned an empty answer'
            print(error)
            ans_to_lines = 'EMPTY_ANSWER_ERROR'
            model_gen = error
        except Exception as e:
            error = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            print(f'Process task Error: {type(e)} : {error}')
            ans_to_lines = 'GENERAL_ERROR'
            model_gen = error
        if error:
            ans_to_lines = [ans_to_lines]
        return ans_to_lines, model_gen

    def _humaneval_task_impl(self, fn_name, code, task, sandbox: Sandbox, _input, invocation=None, mbpp_task_idx=None, input_idx=None):
        _, states = sandbox.run(*eval(_input))
        assert sandbox.status == 'ok', f'Error: {sandbox.status} caused by {fn_name}{_input}'
        if invocation is None:
            invocation = f'{fn_name}{_input[:-2]})'
        codelines = code.split('\n')
        code = ''.join([str(i+1) + '\t' + codelines[i] + '\n' for i in range(len(codelines))])
        gens = []
        for t in task:
            line = t['lineno']
            if self._no_skip_test_cases and list((mbpp_task_idx, input_idx, line)) not in self._no_skip_test_cases:
                print(f'Skipping {(mbpp_task_idx, input_idx, line)}')
                continue
            print(f"Valid {(mbpp_task_idx, input_idx, line)}")
            _actual = states.get_next_line(line-1) # to 0-indexed
            actual = []
            for a in _actual:
                if a == -1:
                    actual.append(-1)
                else:
                    actual.append(a+1)
            if self.prompt_type != TRACE_OF_THOUGHTS_PROMPT_TYPE:
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
            else:
                p = ''
                code = '\n'.join(codelines)
                tot_result = self._process_tot_task_impl_with_validation(actual, mbpp_task_idx, input_idx, line, code, invocation)
                if tot_result is None:
                    continue
                ans_to_lines, model_gen = tot_result
            gens.append({'generated': model_gen, 'response': ans_to_lines, 'expected': actual})
            try:
                result = any(a in actual for a in ans_to_lines)
                gens.append({'generated': model_gen, 'response': ans_to_lines, 'expected': actual, 'line': line, 'prompt': p, 'result': result})
            except:
                gens.append({'generated': model_gen, 'response': ans_to_lines, 'expected': actual})
            self._update_metrics(ans_to_lines, actual)
            print(ans_to_lines, actual)
            acc = (1.0 *self._correct) / self._total
            print(f"Acc: {acc}, {self._correct}, {self._total}")
            print()
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

class ValidationError(Exception):
    pass

class EmptyAnswerError(Exception):
    pass

class State(Task):
    def __init__(self, model: Model, prompt_type='direct', **kwargs):
        super().__init__('state', model, prompt_type, **kwargs)
        self._correct = 0
        self._total = 0

    @property
    def _metrics(self):
        acc = self._correct / self._total
        if self._total:
            acc = self._correct / self._total
        else:
            acc = 0
        return {'acc': acc,
                'correct': self._correct,
                'total': self._total}

    def _update_metrics(self, ans, actual):
        self._total += 1
        if ans == 'ERROR':
            return False
        if self.prompt_type == TRACE_OF_THOUGHTS_PROMPT_TYPE:
            for _ans in ans:
                if _ans is not None:
                    if self._eq(_ans, actual):
                        self._correct += 1
                        return True
            return False
        else:
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
    
    def _process_tot_task_impl(self, mbpp_task_idx, input_idx, var, line, code, invocation, use_labels):
        re_task_kwargs = {'var': var, 'line': line}
        code = black.format_str(code, mode=black.Mode(line_length=120))
        invocation = black.format_str(invocation, mode=black.Mode(line_length=120))
        validation_kwargs = {'invocation': invocation, 'code': code}
        try:
            self.tot_parser.validate_task(mbpp_task_idx, input_idx, self.__class__.__name__, re_task_kwargs, validation_kwargs)
        except Exception as e:
            traceback_err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            raise ValidationError(traceback_err)

        ans, model_gen = self.tot_parser.process_task(mbpp_task_idx, input_idx, self.__class__.__name__, re_task_kwargs, validation_kwargs, use_labels)
        processed_ans = []
        for _ans in ans:
            var_val, var_type = _ans
            if 'Nil' == var_val and 'Nil' == var_type:
                _ans = Nil
            processed_ans.append(_ans)
        ans = processed_ans
        if not ans:
            raise EmptyAnswerError(f'task {mbpp_task_idx}:{input_idx} Returned an empty answer, Skip')
        return ans, model_gen
    
    def _process_tot_task_impl_with_validation(self, actual, mbpp_task_idx, input_idx, var, line, code, invocation):
        is_valid_test_case = False
        try:
            ans, _ = self._process_tot_task_impl(mbpp_task_idx, input_idx, var, line, code, invocation, True)
            for _ans in ans:
                if self._eq(_ans, actual):
                    is_valid_test_case = True
        except Exception as e:
            pass
        
        if not is_valid_test_case:
            # print(f'Invalid test case {mbpp_task_idx}:{input_idx}, continue')
            return None
        
        self._valid_test_cases.append((mbpp_task_idx, input_idx, var, line))
        # print(f'Valid test case {mbpp_task_idx}:{input_idx}')
        try:
            ans, model_gen = self._process_tot_task_impl(mbpp_task_idx, input_idx, var, line, code, invocation, False)
        except ValidationError as e:
            # should not happen
            error = f'Validate task {mbpp_task_idx}:{input_idx} Error: {type(e)}'
            ans = 'ERROR'
            model_gen = error
        except EmptyAnswerError as e:
            error = f'Task {mbpp_task_idx}:{input_idx} Returned an empty answer'
            print(error)
            ans = 'ERROR'
            model_gen = error
        except Exception as e:
            error = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            print(f'Process task Error: {type(e)} : {error}')
            ans = 'ERROR'
            model_gen = error
        return ans, model_gen

        
    def _humaneval_task_impl(self, fn_name, code, task, sandbox: Sandbox, _input, invocation=None, mbpp_task_idx=None, input_idx=None):
        _, states = sandbox.run(*eval(_input))
        assert sandbox.status == 'ok', f'Error: {sandbox.status} caused by {fn_name}{_input}'
        if invocation is None:
            invocation = f'{fn_name}{_input[:-2]})'
        codelines = code.split('\n')
        gens = []
        for t in task:
            print()
            line = t['lineno']
            var = t['var']
            if self._no_skip_test_cases and list((mbpp_task_idx, input_idx, var, line)) not in self._no_skip_test_cases:
                print(f'Skipping {(mbpp_task_idx, input_idx, var, line)}')
                continue
            print(f"Valid {(mbpp_task_idx, input_idx, var, line)}")
            actual = states.interpret_var(line-1, var) # to 0-indexed
            if self.prompt_type != TRACE_OF_THOUGHTS_PROMPT_TYPE:
                codeline = codelines[line-1] # to 0-indexed
                p = self._build_prompt(code=code, invocation=invocation, 
                                        invocation_abbr=invocation, 
                                        line=line, codeline=codeline, var=var)
                if self.mock:
                    ans, model_gen = (('mock_value','mock_type'), 'mock_model_gen')
                else:
                    ans, model_gen = self._prompt_model(p)
                self._valid_test_cases.append((mbpp_task_idx, input_idx, var, line))
            else:
                p = ''
                tot_result = self._process_tot_task_impl_with_validation(actual, mbpp_task_idx, input_idx, var, line, code, invocation)
                if tot_result is None:
                    continue
                ans, model_gen = tot_result
            res = self._update_metrics(ans, actual)
            acc = (1.0 *self._correct) / self._total
            print(f"Acc: {acc}, {self._correct}, {self._total}")
            print()
            print(ans, actual, res)
            try:
                generated_sample = {'generated': model_gen, 'eq': res, 'line': line, 'var': var, 'prompt': p, 'ans': ans, 'actual': actual}
                for key in generated_sample.keys():
                    try:
                        json.dumps(generated_sample[key])
                    except:
                        generated_sample[key] = f"STRINGIFIED, {str(generated_sample[key])}"
                gens.append(generated_sample)
            except:
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

    def _humaneval_task_impl(self, fn_name, code, task, sandbox: Sandbox, _input, invocation=None, mbpp_task_idx=None, input_idx=None):
        # make `fn_name` callable in current scope
        locals()[fn_name] = FunctionFactory.create(fn_name, code)
        if invocation is None:
            invocation='\n' + _input
        p = self._build_prompt(code=code, invocation=invocation)
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

    def _run(self, mock=False):
        print(f'The arguments for this run: {self.kwargs}')
        if self.kwargs['task'] == 'consistency':
            self.kwargs['mock'] = True
        if mock:
            model = None
            self.kwargs['custom_mock'] = True
        elif self.kwargs['prompt_type'] == TRACE_OF_THOUGHTS_PROMPT_TYPE:
            model = None
        else:
            model = Model.new(**self.kwargs)
        self.kwargs['model'] = model
        TASKS = {
            'coverage': Coverage,
            'path': Path,
            'state': State,
            'output': Output
        }
        VALID_TEST_CASES_PATH = {
            'coverage': "/a/home/cc/students/cs/boazlavon/code/REval/model_generations/coverage@google/gemma-1-2b-it_tot/25-01-29-22-59.valid_test_cases.mbpp.json",
            'path'   : "/a/home/cc/students/cs/boazlavon/code/REval/model_generations/path@google/gemma-1-2b-it_tot/25-01-29-23-09.valid_test_cases.mbpp.json",
            'state'   : "/a/home/cc/students/cs/boazlavon/code/REval/model_generations/state@google/gemma-1-2b-it_tot/25-01-29-23-01.valid_test_cases.mbpp.json"
        }
        if self.kwargs['prompt_type'] != TRACE_OF_THOUGHTS_PROMPT_TYPE:
            self.kwargs['valid_test_cases_path'] = VALID_TEST_CASES_PATH[self.kwargs['task']]
            print(f"Valid testcases: {self.kwargs['valid_test_cases_path']}")
        #task = getattr(sys.modules[__name__], self.kwargs['task'].capitalize())(**self.kwargs)
        task = TASKS[self.kwargs['task']](**self.kwargs)
        task.run()

    @staticmethod
    def run_with_config(load_path='.eval_config', mock=False):
        cli = Cli()
        if not os.path.exists(load_path):
            print(f'Error: {load_path} file not found')
            sys.exit(1)
        with open(load_path, 'r') as f:
            cli.kwargs = json.load(f)
        cli._run(mock)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run evaluation for DREval tasks')
    parser.add_argument('command', nargs='?', type=str, default='run', choices=['config', 'run'], help='Command to run')
    parser.add_argument('-i', '--input', type=str, default='.eval_config', help='specify configuration file to load')
    parser.add_argument('-o', '--output', type=str, default='.eval_config', help='specify configuration file to save')
    parser.add_argument('--mock', type=bool, default=False, required=False, help='specify whether using the model')
    args = parser.parse_args()
    if args.command == 'config':
        Cli.config(args.output)
        sys.exit(0)
    elif args.command == 'run':
        Cli.run_with_config(args.input, args.mock)
        sys.exit(0)
    else:
        raise RuntimeError('unreachable')
