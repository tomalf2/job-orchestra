from typing import Any, List, Iterable

import sklearn.cluster
import sklearn.decomposition
from flu_dev_refact.pipeline import Context, Step
from os import makedirs, getcwd, listdir
from shutil import rmtree
from os.path import exists, sep, abspath
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest
import matplotlib.pyplot as plt
from cleverdict import CleverDict
from flatdict import FlatDict
import sklearn

test_temp_dir = "test_temp"

@pytest.fixture(scope='module')
def environment():
    # setup
    makedirs(test_temp_dir, exist_ok=True)
    assert listdir(test_temp_dir) == []

    yield ""
    
    # cleanup
    rmtree(test_temp_dir)

class TestContext:

    @staticmethod
    def test_flat_context_store_and_load(environment):
        # create a context and store it on disk
        ex = Context({
            # context_meta attributes are automatically added
            # simple types
            'text': "text",
            'num': 0,
            'check': True,
            'dec': 1.1,
            'list': [1,2,3],
            'dictionary': {"key": [1,2,3]},
            # complex types
            'dataframe': pd.DataFrame({'a': [1, 2], 'b': [3.0, 4.0]}),
            'series': pd.Series([3.0, 4.0], name='b'),
            'clevdict': CleverDict({'key': [1,2,3]}),
            'np.ndarray': np.array([1, 2, 3]),
            # pickables
            # 'kmeans': sklearn.cluster._kmeans.KMeans(),   # no simple way to check equality of sklearn objects
            # 'pca': sklearn.decomposition.PCA()
            # ignored keys
            '.ignore_this': "this should be not stored",
            'dictionary_with_ignore': {'key': [1,2,3], '.ignore_this_key': [1,2,3]},
        })
        test_location = test_temp_dir + sep + "test_flat_context_store_and_load"
        ex.store(test_location)
        
        # load the context from disk
        ex1 = Context.load(test_location)

        # compare the original context matches with the one loaded from memory (except ignored)
        ## context meta
        assert ex['context_meta'] == ex1['context_meta']
        ## simple types
        for item in ["text", "num", "check", "dec", "list", "dictionary"]:
            assert type(ex[item] == ex1[item]) and ex[item] == ex1[item]
        ## check that ignored keys are really ignored
        assert '.ignore_this' not in ex1.keys()
        assert ex['dictionary_with_ignore']['key'] == ex1['dictionary_with_ignore']['key'] and len(ex1['dictionary_with_ignore'].keys()) == 1
        ## complex types
        assert_frame_equal(ex.dataframe, ex1.dataframe)
        assert_series_equal(ex.series, ex1.series)
        assert ex.clevdict == ex1.clevdict
        assert np.array_equal(ex['np.ndarray'], ex1['np.ndarray'])

    @staticmethod
    def test_flatten_and_restore_dict():
        test_dict = CleverDict({
            'A': 1,
            'B': {
                'nested1': 1,
                'nested2': 2
            },
            'C': {
                'nested1': {
                    'deep_nested': CleverDict({'key': [1,2,3]}) 
                }
            },
            'D': [1, 2],
            'F': [{'nested1': 1}]
        })
        flattened_dict = dict(FlatDict(test_dict, delimiter=Context._FLAT_DICT_DEPTH_SEPARATOR))
        assert test_dict == Context._restore_nested_dict(flattened_dict)

    @staticmethod
    def test_nested_context_store_and_load(environment):
        # create a context and store it on disk
        ex = Context({
            'A': 1,
            'B': {
                'nested1': pd.DataFrame({'a': [1, 2], 'b': [3.0, 4.0]}),
                'nested2': 2
            },
            'C': {
                'nested1': {
                    'deep_nested': '1' 
                }
            },
            'D': [1, 2],
            'F': [{'nested1': 1}]
        })
        test_location = test_temp_dir + sep + "test_nested_context_store_and_load"
        ex.store(test_location)
        
        # load the context from disk
        ex1 = Context.load(test_location)

        # compare the original context matches with the one loaded from memory
        ## complex type
        assert ex.keys() == ex1.keys()
        assert ex.B.keys() == ex1.B.keys()
        assert_frame_equal(ex.B['nested1'], ex1.B['nested1'])
        # remove complex types
        ex.B['nested1'] = None
        ex1.B['nested1'] = None
        assert ex == ex1

    @staticmethod
    def test_no_collections_of_complex_objs():
        # this dictionary must raise an error
        test_dict = Context({
            'a': [1,2,3],
            'b': [{'nested_simple': [1,2]}],
            'c': [CleverDict({1: 2})],
            'd': [
                    [CleverDict({1: 2})]
                ],
            'f': CleverDict({"nested": 1})
        })
        test_location = test_temp_dir + sep + "test_no_collections_of_complex_objs__1"
        try:
            test_dict.store(test_location)
        except AssertionError:
            pass
        else:
            raise AssertionError("Context._assert_no_collections_of_complex_obj should raise an exception for this input")
        
        # make the dictionary compliant (should not raise any error)
        del test_dict['c']
        del test_dict['d']
        test_location = test_temp_dir + sep + "test_no_collections_of_complex_objs__2"
        test_dict.store(test_location)


# class TestStep:

#     def create_steps():
#         # data
#         example_dataset = pd.DataFrame({'a': [1, 2], 'b': [3.0, 4.0]})

#         # processing
#         class Subtract(Step):
#             def run_on(self, input_step) -> pd.DataFrame:
#                 super().run_on(input_step)
#                 return input_step['output']
                
#             def run(self, data):
#                 return {"output": data - 1}           
       
#         class ScalarMultiply(Step):
#             def run_on(self, input_a, input_b) -> Iterable[pd.DataFrame]:
#                 super().run_on(input_a, input_b)
#                 return input_a['output'], input_b['output']
                
#             def run(self, input_a: pd.DataFrame, input_b: pd.DataFrame):
#                 output = {"output": input_a.mul(input_b)}
#                 return output


#         transform1 = Subtract()
#         merge = ScalarMultiply()

#         # final output

#         class Visualizer(Step):
#             def run_on(self, input_step) -> pd.DataFrame:
#                 super().run_on(input_step)
#                 return input_step['output']
            
#             def run(self, input_data):
#                 test_location = test_temp_dir + sep + "pipeline_out.png"
#                 plt.figure(figsize=(8,8))
#                 plt.scatter(x=input_data.a, y=input_data.b)
#                 plt.xlabel("a")
#                 plt.ylabel("b")
#                 plt.savefig(test_location)
                
#                 return {}
            
#         class FinalResult(Step):
#             def run_on(self, input_step) -> pd.DataFrame:
#                 super().run_on(input_step)
#                 return input_step['output']
            
#             def run(self, input_data):
#                 all_values = input_data.values.reshape(-1)
#                 return {"output": sum(all_values)}


#         plot = Visualizer()
#         result = FinalResult()

#         # pipeline
#         ## input
#         dataset1 = Step.from_data(example_dataset)
#         dataset2 = Step.from_data(example_dataset**2)

#         # transformations
#         transform1.run_on(dataset1)
#         merge.run_on(transform1, dataset2)

#         # ouptut steps
#         plot.run_on(merge)
#         result.run_on(merge)

#         result.run()
        

        
        
class TestStep:
    
    @staticmethod
    def test_call_hierarchy_without_context():
        """
          A
         / \\
        B   C
         \\ /
          D
        """

        class A(Step):
            def run(self, *args, **kwargs):
                print("I'm "+str(self)+". Received args:", args, "kwargs", kwargs)
                if not args:    
                    return str(self)                    # example: A
                elif len(args) == 1:
                    return args[0] + " > " + str(self)  # example: A > B
                else:
                    args = [f"({res})" for res in args]                 # example: from [A, B] to [(A), (B)]
                    joint_output_of_dependencies = ' + '.join(args)     # example: from [(A), (B)] to (A) + (B)
                    return "(" + joint_output_of_dependencies + ")" + " > " + str(self) # example: ((A) + (B)) > C

        a = A()
        b = A(depends_on=a, name_alias='B')
        c = A(depends_on=a, name_alias='C')
        d = A(depends_on=[b,c], name_alias='D')

        # Given the current DAG, the call order is fixed 
        # and independent from depth-first or breadh-first dependency exploration logics.
        # Only one output is possible

        # Check Steps that are going to be executed
        assert d._dry_run() == dict.fromkeys(["A", "B", "C", "D"], True)

        # Check final output
        assert d.materialize() == "((A > B) + (A > C)) > D"    
        
    @staticmethod
    def test_call_hierarchy_with_context():
        """
          A
         / \\
        B   C
         \\ /
          D
        """
        runs_record = []

        class A(Step):
            def run(self, *args, **kwargs):
                nonlocal runs_record
                runs_record.append(str(self))
                print("I'm "+str(self)+". Received args:", args, "kwargs", kwargs)
                if not args:    
                    return str(self)                    # example: A
                elif len(args) == 1:
                    return args[0] + " > " + str(self)  # example: A > B
                else:
                    args = [f"({res})" for res in args]                 # example: from [A, B] to [(A), (B)]
                    joint_output_of_dependencies = ' + '.join(args)     # example: from [(A), (B)] to (A) + (B)
                    return "(" + joint_output_of_dependencies + ")" + " > " + str(self) # example: ((A) + (B)) > C

        ct = Context()
        a = A(ctx=ct)
        b = A(depends_on=a, name_alias='B', ctx=ct)
        c = A(depends_on=a, name_alias='C', ctx=ct)
        d = A(depends_on=[b,c], name_alias='D', ctx=ct)
        
        # Check Steps that are going to be executed
        assert d._dry_run() == dict.fromkeys(["A", "B", "C", "D"], True)

        output = d.materialize()

        # Given the current DAG, the call order is fixed 
        # and independent from depth-first or breadh-first dependency exploration logics.
        # Only one output is possible

        # Check intermediate outputs
        assert ct.to_dict(ignore='context_meta') == {
            'A': 'A', 
            'B': 'A > B', 
            'C': 'A > C',
            'D': '((A > B) + (A > C)) > D'}
        
        # Check that repeated dependencies re-use previous outputs
        assert runs_record == ['A', 'B', 'C', 'D']

        # Check final output
        assert output == "((A > B) + (A > C)) > D"     

        # Check Steps that are going to be executed upon subsequent run (Context can provide cache)
        assert d._dry_run() == {"D": False}
    
    @staticmethod
    def test_step_from_data():
        df = pd.DataFrame({'a': [1, 2], 'b': [3.0, 4.0]})
        data_step = Step.from_data(df)
        assert_frame_equal(df, data_step.materialize().output)

