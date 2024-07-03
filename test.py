import inspect
import unittest

import pandas as pd

from dataset import DREval
from dynamics import Nil, ClassFactory, FunctionFactory, Sandbox

class TestSandbox(unittest.TestCase):
    def test_nil(self):
        import pickle

        self.assertNotEqual(Nil, None)
        self.assertNotEqual(Nil, 0)
        self.assertNotEqual(Nil, False)
        a = Nil
        self.assertTrue(a is Nil)
        self.assertTrue(a == Nil)
        self.assertEqual(Nil, pickle.loads(pickle.dumps(Nil)))

    def test_function_factory(self):
        code = '''def f(x):\n\treturn x**2'''
        fn = FunctionFactory.create('f', code)

        self.assertEqual(fn(2), 4)
        self.assertEqual(fn.__doc__, code)

    def test_class_factory(self):
        code = '''class A:\n\tdef __init__(self, x):\n\t\tself.x = x\n\tdef f(self):\n\t\treturn self.x**2'''
        cls = ClassFactory.create('A', code)

        self.assertEqual(cls(2).f(), 4)
        self.assertEqual(cls.__doc__, code)

    def test_sandbox_1(self):
        '''
        Basic test
        '''
        code = '''def f(x):\n\treturn x**2'''
        fn = FunctionFactory.create('f', code)
        sandbox = Sandbox(fn)
        result, states = sandbox.run(2)

        self.assertEqual(result, 4)
        self.assertEqual(states.get_return(1), 4)
        self.assertEqual(states.get_local(1, 'x'), [2])
        self.assertEqual(states.get_exception(1), Nil)
        self.assertFalse(states.get_coverage(0))
        self.assertTrue(states.get_coverage(1))
        self.assertTrue(-1 in states.get_next_line(1))
        self.assertEqual(sandbox.status, 'ok')

    def read_dataset(self, idx):
        df = pd.read_json('data/DREval_data.jsonl', lines=True).to_dict(orient='records')
        item = df[idx]
        if DREval.HUMANEVAL_START <= idx <= DREval.HUMANEVAL_END:
            return item['entry_point'], item['code']
        elif DREval.CLASSEVAL_START <= idx <= DREval.CLASSEVAL_END:
            return item['entry_point'], item['code'], item['test']
        else:
            raise ValueError(f'Invalid index: {idx}')
    
    def test_sandbox_2(self):
        fn_name, code = self.read_dataset(5)
        fn = FunctionFactory.create(fn_name, code)
        sandbox = Sandbox(fn)
        result, states = sandbox.run([1, 2, 3, 4])

        self.assertEqual(result, (10, 24))
        self.assertTrue(0 in states.get_local(14, 'sum_value'))
        self.assertTrue(6 in states.get_local(15, 'sum_value'))
        self.assertTrue(6 in states.get_local(16, 'prod_value'))

    def test_sandbox_3(self):
        '''
        Functions with dependencies
        '''
        code = '''def f(x):\n\treturn x**2\ndef g(x):\n\ta = f(x)\n\treturn a'''
        fn = FunctionFactory.create('g', code)
        sandbox = Sandbox(fn)
        result, states = sandbox.run(2)

        self.assertEqual(result, 4)
        self.assertEqual(states.get_return(1), 4)
        self.assertEqual(states.get_return(4), 4)
        self.assertTrue(states.get_coverage(1))

    def test_sandbox_4(self):
        '''
        Nested functions
        '''
        code = '''def g(x):\n\tdef f(x):\n\t\ty = x**2\n\t\treturn y\n\ta = f(x)\n\treturn a'''
        fn = FunctionFactory.create('g', code)
        sandbox = Sandbox(fn)
        result, states = sandbox.run(2)

        self.assertEqual(result, 4)
        self.assertTrue(4 in states.get_local(2, 'y'))

    def test_sandbox_5(self):
        cls_name, code, test = self.read_dataset(85)
        ClassFactory.create(cls_name, code)
        tcls = ClassFactory.create_test_classes(cls_name, code, test,
                                                DREval.tcls_pattern,
                                                DREval.tcls_validation,
                                                DREval.tcls_postprocess)[0]
        obj = tcls()
        sandbox = Sandbox(obj.dreval_test)
        _, states = sandbox.run()

        self.assertEqual(sandbox.status, 'ok')
        self.assertTrue(13 in states.get_next_line(11))
        self.assertTrue(states.get_coverage(13))
        self.assertTrue('GET' in states.get_local(11, 'method'))
        self.assertTrue('GET' in states.get_subscript(11, 'request', '"method"'))
        self.assertTrue(inspect.isroutine(states.get_attr(11, 'self', 'is_start_with')[0]))
        self.assertTrue('GET' in states.interpret_var(11, 'method'))
        self.assertTrue('GET' in states.interpret_var(11, 'request["method"]'))
        self.assertTrue(inspect.isroutine(states.interpret_var(11, 'self.is_start_with')[0]))

if __name__ == '__main__':
    unittest.main()