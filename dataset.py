import inspect

from typing import Type

class ClassEval:
    @staticmethod
    def tcls_pattern(test_cls_name: str, cls_name: str) -> bool:
        '''
        For ClassEval, the test class name should be "{cls_name}Test.*"
        '''
        import re
        return re.match(f"{cls_name}Test.*", test_cls_name) is not None
    
    @staticmethod
    def tcls_validation(cls: Type) -> bool:
        import unittest
        return issubclass(cls, unittest.TestCase)

    @staticmethod
    def tcls_postprocess(cls: Type) -> Type:
        '''
        For a test class in ClassEval, there may be multiple test methods.
        For efficiency, we only retain the first test method and rename it 
        to 'dreval_test'.
        '''
        test_fns = []
        for k in cls.__dict__.copy():
            if k.startswith('test'):
                test_fns.append(k)
        assert len(test_fns) > 0, f"No test methods found in {cls}"
        fn = getattr(cls, test_fns[0])
        lines, _ = inspect.getsourcelines(fn)
        assert len(lines) > 0
        fn.__doc__ = cls.__doc__
        fn.__source__ = inspect.getsource(fn)
        fn.__input__ = ''.join(list(map(lambda x: x.replace('self.assert', 'assert').lstrip(), lines[1:])))
        if hasattr(cls, 'setUp'):
            cls.__setup__ = inspect.getsource(cls.setUp)
        setattr(cls, 'dreval_test', fn)
        for k in test_fns:
            delattr(cls, k)
        return cls

class DREval(ClassEval):
    HUMANEVAL_START = 0
    HUMANEVAL_END   = 84
    CLASSEVAL_START = 85
    CLASSEVAL_END   = 153
    MBPP_START      = 154
    MBPP_END        = 654

    # Limit max number of inputs
    # due to computational budgets
    MAX_INPUTS = 5