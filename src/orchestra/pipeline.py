# Context dependencies
from typing import Any, Dict, List, Optional
import cleverdict
from cleverdict import CleverDict
import pandas as pd
from os.path import sep
from pathvalidate import sanitize_filepath
import builtins
import numpy as np
from os import makedirs
from flatdict import FlatDict
import pickle
import sklearn
import warnings
import sys
from datetime import datetime

# Step dependencies
from typing import List, Optional, Tuple
import networkx as nx
from dagviz import dagre
from copy import copy


class Context(CleverDict):
    """
    A special dictionary whose values can be referenced by the syntax 'context[key]' and 'context.key'. 
    Methods .store() and .load() enable on-disk memory persistency. 
    As the dictionary is saved to a JSON file, the persistency of standard types is supported excluding sets and tuples. 
    Persistency operations have been implemented also for pd.DataFrames, pd.Series and cleverdict.Cleverdict.
    Keys prefixed with ".", at any nesting level, are not stored on disk. 

    Persistency through pickle is discouraged for maintainability reasons. 

    Due to dependency errors, a Context with the following shape is currently not suported. The problem is with the use of 
    numbers as keys for dictionaries at 1st nested level. If the {number: 'any_value'} is nested at 2nd level, 
    there should be no issues. For ease, avoid using numebrs as keys at any level.
    {
        'any_key': { 
            number: 'any_value'
        }
    }
    """

    _TYPE_STRING_FRAME = "_-_pd.DataFrame"
    _TYPE_STRING_SERIES = "_-_pd.Series"
    _TYPE_STRING_CLEVERDICT = "_-_Cleverdict"
    _TYPE_STRING_NUMPY = "_-_np.ndarray"
    _TYPE_STRING_PICKLE = "_-_pickle"
    _FLAT_DICT_DEPTH_SEPARATOR = ";;"
    _DONT_STORE_FLAG_KEY_PREFIX = "."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'context_meta' not in self:
            self['context_meta'] = {     
                'created_on': str(datetime.now().date()),
                'python': Context.python_version(), 
                'sklearn': Context.sklearn_version()
            }  
  
    # Saving/Loading operations
    def store(self, dir_path=None, fullcopy=False):
        """
        Methods .store() and .load() enable on-disk memory persistency. 
        As the dictionary is saved to a JSON file, the persistency of standard types is supported excluding sets and tuples. 
        Persistency operations have been implemented also for pd.DataFrames, pd.Series and cleverdict.Cleverdict.
        Keys prefixed with ".", at any nesting level, are not stored on disk. 

        Persistency through pickle is discouraged for maintainability reasons. 

        Due to dependency errors, a Context with the following shape is currently not suported. The problem is with the use of 
        numbers as keys for dictionaries at 1st nested level. If the {number: 'any_value'} is nested at 2nd level, 
        there should be no issues. For ease, avoid using numebrs as keys at any level.
        {
            'any_key': { 
                number: 'any_value'
            }
        }
        """
        # flatten the dictionary
        try:
            dict_with_flat_structure = dict(FlatDict(self, delimiter=Context._FLAT_DICT_DEPTH_SEPARATOR))
        except KeyError as e:
            raise NotImplementedError("The dependency 'flatdict' does not support the flattening of structures like {'any_key': {numeber: 'any_value'}}. " + 
                                    "The problem is with 'number'. " + f"In this case number was {str(e)} (may appear as a sting because flatdict does so). " + 
                                    "Please do not use numbers as dictionary keys.")
        # ignore keys prefixed with .
        dict_with_flat_structure = {k:v for k,v in dict_with_flat_structure.items() if k.split(Context._FLAT_DICT_DEPTH_SEPARATOR)[-1][0] !=  Context._DONT_STORE_FLAG_KEY_PREFIX}
        
        # check that values don't map to collection of complex types
        [self._assert_no_collections_of_complex_obj(k,v) for k,v in dict_with_flat_structure.items()]   

        # make a dir for all the files
        if dir_path[-1] != sep:
            dir_path += sep
        makedirs(dir_path, exist_ok=False)

        # parameter passed to downstream functions
        self.path = dir_path    # this path is used by internal functions to store complex objects
        
        # save complex objects (may transform the real value to a different type, or write it to disk and return a path in turn)
        dict2obj_paths = CleverDict(dict([self._get_or_save(k, v) for k,v in dict_with_flat_structure.items()]))

        # save the master record
        master_record_path = self.path + "master_record.json"
        dict2obj_paths.to_json(file_path=master_record_path, fullcopy=fullcopy)

        # clean-up parameters passed to upstream functions
        del self['path']
  
    @classmethod
    def load(cls, dir_path=None):
        """
        Methods .store() and .load() enable on-disk memory persistency. 
        As the dictionary is saved to a JSON file, the persistency of standard types is supported excluding sets and tuples. 
        Persistency operations have been implemented also for pd.DataFrames, pd.Series and cleverdict.Cleverdict.
        Keys prefixed with ".", at any nesting level, are not stored on disk. 

        Persistency through pickle is discouraged for maintainability reasons. 

        Due to dependency errors, a Context with the following shape is currently not suported. The problem is with the use of 
        numbers as keys for dictionaries at 1st nested level. If the {number: 'any_value'} is nested at 2nd level, 
        there should be no issues. For ease, avoid using numebrs as keys at any level.
        {
            'any_key': { 
                number: 'any_value'
            }
        }
        """
        # load the master record
        if dir_path[-1] != sep:
            dir_path += sep
        master_record_path = dir_path + "master_record.json"
        dict2obj_paths = CleverDict.from_json(None, master_record_path)

        # load complex objects (may replace the loaded value with a transformation or a file loaded from disk)
        dict_with_flat_structure = dict([cls._get_or_load(k, v) for k,v in dict2obj_paths.items()])

        # restore nested structure
        dict_with_nested_structure = cls._restore_nested_dict(dict_with_flat_structure)
                
        return cls(dict_with_nested_structure)
    
    @classmethod
    def _restore_nested_dict(cls, dict_with_flat_structure: dict):
        original_dict_structure = dict()
        for k, v in dict_with_flat_structure.items():
            depth_levels = k.split(Context._FLAT_DICT_DEPTH_SEPARATOR) # FlatDict uses "." as delimiter between depth levels
            current_dict = original_dict_structure
            while len(depth_levels) >= 2:
                # depth_levels = A, B, ...
                # check if dictionary B exists or create it
                outer_level = depth_levels[0]
                try:                                       #  A
                    next_dict = current_dict[outer_level]     # current_dict = B
                except KeyError:
                    current_dict[outer_level] = CleverDict()           # new empty dict 
                    next_dict = current_dict[outer_level]
                current_dict = next_dict
                depth_levels = depth_levels[1:]     # depth_levels = B, C, ...
            current_dict[depth_levels[0]] = v
        return original_dict_structure

    @staticmethod
    def _assert_no_collections_of_complex_obj(name, collection):
        if type(collection) == builtins.list:   # collection = [1, 2, 3, ComplexObj, ..]
            # search nested collections and check them recursively
            [Context._assert_no_collections_of_complex_obj(name, x) for x in collection if type(x) == builtins.list]
            # then check that types of elements in this collection are not complex
            assert all([type(x) in (builtins.int, builtins.str, builtins.float, builtins.bool, builtins.dict) for x in collection]), f"{name} key is mapped to a list of compelx objects that cannot be stored on disk. Save complex objects outside of a list."
    
    # Saving/Loading operations -- Helper methods
    def _get_or_save(self, name, value):
        match type(value):
            # all options in case must be expressed as "something.something"
            case builtins.int | builtins.str | builtins.float | builtins.bool | builtins.list | builtins.dict:
                pass
            case pd.DataFrame:
                name, value = self._save_df(name, value)
            case pd.Series:
                name, value = self._save_series(name, value)
            case np.array | np.ndarray:
                name, value = self._save_np(name, value)
            case cleverdict.CleverDict:
                name, value = self._save_cleverdict(name, value)
            case (sklearn.cluster._kmeans.KMeans | sklearn.decomposition.PCA):
                name, value = self._save_pickle(name, value)
            case _:
                raise TypeError(f"Unsupported store operation for variable {name} of type {type(value)}. If ")
        return name, value    
    
    @classmethod
    def _get_or_load(cls, name, value):
        try:
            type_string = name[name.rindex("_-_"):]
        except ValueError:  # no type_string, then it is a builtin type (CleverDict already handles all except set and tuple)
            return name, value
        else:
            match type_string:
                case Context._TYPE_STRING_FRAME:
                    name, value = cls._load_df(name, value)
                case Context._TYPE_STRING_SERIES:
                    name, value = cls._load_series(name, value)
                case Context._TYPE_STRING_CLEVERDICT:
                    name, value = cls._load_cleverdict(name, value)
                case Context._TYPE_STRING_NUMPY:
                    name, value = cls._load_np(name, value)
                case Context._TYPE_STRING_PICKLE:
                    name, value = cls._load_pickle(name, value)
                case _:
                    raise TypeError(f"Unsupported load operation for complex object {name} with type string {type_string}.")
        return name, value

    ## Data Frames        
    def _save_df(self, name: str, value: pd.DataFrame) -> Tuple[str, str]:
        name += Context._TYPE_STRING_FRAME
        path = sanitize_filepath(self.path + name + ".parquet")
        value.to_parquet(path)
        return name, path

    @classmethod
    def _load_df(cls, name: str, path: str) -> Tuple[str, pd.DataFrame]:
        return name.removesuffix(cls._TYPE_STRING_FRAME), pd.read_parquet(path)

    ## Series    
    def _save_series(self, name: str, value: pd.Series) -> Tuple[str, str]:
        name += Context._TYPE_STRING_SERIES
        path = sanitize_filepath(self.path + name + ".parquet")
        pd.DataFrame(value).to_parquet(path)
        return name, path
    
    @classmethod
    def _load_series(cls, name: str, path: str) -> Tuple[str, pd.Series]:
        name = name.removesuffix(cls._TYPE_STRING_SERIES)
        df = pd.read_parquet(path)
        return name, df[df.columns[0]]
    
    ## Cleverdict
    def _save_cleverdict(self, name: str, value: CleverDict) -> Tuple[str, dict]:
        return name + Context._TYPE_STRING_CLEVERDICT, value.to_dict()
    
    @staticmethod
    def _load_cleverdict(name: str, value: dict) -> Tuple[str, CleverDict]:
        return name.removesuffix(Context._TYPE_STRING_CLEVERDICT), CleverDict(value)
    
    ## Numpy
    def _save_np(self, name: str, value: np.ndarray) -> Tuple[str, str]:
        name += Context._TYPE_STRING_NUMPY
        path = sanitize_filepath(self.path + name + ".npy")
        np.save(path, value, allow_pickle=False)
        return name, path
    
    @classmethod
    def _load_np(cls, name: str, path: str) -> Tuple[str, np.ndarray]:
        name = name.removesuffix(cls._TYPE_STRING_NUMPY)
        value = np.load(path, allow_pickle=False)
        return name, value
    
    ## Pickle 
    def _save_pickle(self, name: str, value: Any) -> Tuple[str, str]:
        name += Context._TYPE_STRING_PICKLE
        path = sanitize_filepath(self.path + name + ".pickle")
        with open(path, "wb") as f:
            pickle.dump(value, f)
        return name, path

    @classmethod
    def _load_pickle(cls, name: str, path: str) -> Tuple[str, Any]:
        name = name.removesuffix(cls._TYPE_STRING_PICKLE)
        with open(path, "rb") as f:
            value = pickle.load(f)
        return name, value
    
    @staticmethod
    def python_version() -> str:
        vinfo = sys.version_info
        return f"{vinfo.major}.{vinfo.minor}.{vinfo.micro}"
    
    @staticmethod
    def sklearn_version() -> str:
        return sklearn.__version__
    
    # Suppressed methods
    def to_json(self, file_path=None, fullcopy=False, ignore=None, exclude=None, only=None):
        raise NotImplementedError("Use .store() instead")
    
    @classmethod
    def from_json(cls, *args, **kwargs):
        raise NotImplementedError("Use .load() instead")



class Step:
    """
    Define a step by subclassing Step and overriding the method run(). Finally, call .materialize() on the last step to execute the pipeline. 
    On materialization, The method .run() of each step is called with the output of its dependencies as arguments.
    Each Step should return None or a dictionary that include the output to be passed to the next step.
    To support the swapping of steps, use the same output encoding convetion, for example:
    {
        'output': pd.DataFrame(...),
        'feature_names': [...]
        ...
    }
    A Step is referenced by its class name. The class name must be unique inside a pipeline. However, the same Step can appear multiple times 
    in the same pipeline if given a unique alias with Step(name_alias=...). This helps reusing the same Step for different inputs. 
    """

    # Constructors

    def __init__(self, depends_on: List["Step"]|"Step" = [], ctx: Optional[Context] = None, name_alias: Optional[str]=None):
        self.dependencies = [depends_on] if type(depends_on) != list else depends_on
        self.ctx = ctx
        self.name_alias = name_alias

    @staticmethod
    def from_data(data, name_alias="Data"):
        s0 = Step(name_alias=name_alias)
        def return_data():
            return CleverDict({'output': data})
        s0.run = return_data
        return s0

    # External interface

    def run(self, *args, **kwargs) -> Any:
        """
        Override this method to compute and return the result expected from this Step. The method will receive the result of any dependency declared 
        in the constructor through the argument depends_on. It is up to you to know the type of the output of the dependencies and use it accordingly. 
        To help the interchangeability and modularity of the Steps, it is advised to define a convention for the result type; for example, the return 
        type could be a dictionary with some fixed keys that might be either present or not 
        {
            'output': '...main result goes here...', 
            'model': ..., 
            'feature_names': ...,
            ...
        }. In this case, defining a class ResultType of type TypedDict helps the type checker suggesting the predefined keys. 
        
        :param tuple args: Optional arguments
        :param dict kwargs: Optional keyword arguments
        :return the output of this Step is passed to any other Step requiring this Step as a dependency (through constructor argument 'depends_on')
        """
        # since .run() can return watherver type is needed, it is 
        raise NotImplementedError

    def materialize(self) -> Any:
        """
        Run this Step return its output. Any declared dependency needed by this Step and or its dependencies are run too. 
        """
        self._assert_unique_dependency_types()
        return self._run()

    def depending_steps_breadth_first(self):
        return list(dict.fromkeys(Step._map_dependencies_obj2names(self._dependencies_breadth_first())))    # unique step names sorted breadth first
    
    def depending_steps_depth_first(self):
        return list(dict.fromkeys(Step._map_dependencies_obj2names(self._dependencies_depth_first())))    # unique step names sorted depth first
    
    def dependency_graph(self, graph: Optional[nx.DiGraph] = None):
        """
        Plots a direct acyclic graph of connected Step. Each edge represents a dependency as 
        declared in Step(depends_on=...). Only the Steps that are directly or indirectly 
        required to compute this Step (regardless of the Context availability) are plotted. 
        The returned object is displayed in a Jupyter notebook as an interactive plot with 
        zoom and pan. See also dependency_graph_js() for custom options.
        """
        # start point
        head = False
        if not graph:
            head = True
            graph = nx.DiGraph()
            graph.add_nodes_from(self.depending_steps_breadth_first())
        
        # add edges        
        for dep in self.dependencies:
            graph.add_edge(str(dep), str(self))
            dep.dependency_graph(graph)             # add dependencies depth first
        
        if head:
            return dagre.Dagre(graph)
    
    def dependency_graph_js(self, enable_wheel_zoom=False):
        """
        Returns the raw js code rendereing the DAG from Step.dependency_graph(). It can be visualized in 
        a Jupyter notebook using IPython.display.display_javascript(..., raw=True). By default, this method 
        disables the zoom (not the pan) feature; call dependency_graph_js(True) to re-enable it.
        """
        dagre_zoom_template = dagre._template
        graph_renderer_obj = self.dependency_graph()
        if enable_wheel_zoom:
            return graph_renderer_obj._repr_javascript_()
        else:
            try:    
                # temporarily swap dagre._template (module's attribute)
                dagre._template = dagre_zoom_template.replace('svg.call(zoom);', 'svg.call(zoom).on("wheel.zoom", null);')
                return graph_renderer_obj._repr_javascript_()
            finally:
                dagre._template = dagre_zoom_template   # set it back

    def execution_plan(self):
        """
        Returns a list describing the Steps required for materialization. Each Step is decorated with: 
        - (1) a flag indicating whether its result is available from the context or not.
        - (2) a flag indicating which steps are being invoked if this Step is materialized, depending on the context availability.
        """
        legend = "Execution plan. Result availability: [*] available - [ ] n/a - [x] no context. Scheduled execution: (H) head - (>) scheduled run on materialize - ( ) no run required."
        step_names2obj: dict[str, Step] = {str(self): self} | {str(obj):obj for obj in self._dependencies_breadth_first()}
        step_names2exec = self._dry_run()
        output = ""
        for s_name, s_obj in step_names2obj.items():
            # availability of result from context
            ctx = s_obj.ctx
            if ctx is None:
                ctx_status = "[x]"
            elif ctx.get(s_name, None) is None:
                ctx_status = "[ ]"
            else:
                ctx_status = "[*]"
            # exec required
            if not s_name in step_names2exec:   # the invocation of this step superseded by a subsequent step cached in the dependency graph
                exec_status = "( )"
            elif str(self) == s_name:           # this step is the head (will always be executed if not cached)
                exec_status = "(H)" 
            elif step_names2exec[s_name] is False:  # the result of this step is cached (no exec needed)
                exec_status = "( )"
            else:                               # the result of this step must be computed (not available in cache)
                exec_status = "(>)"
            # exec_status = " " if step_names2exec.get(s_name, True) else ">"
            output = f" {ctx_status} {exec_status} {s_name}\n" + output
        print(legend + "\n" + output)

    @staticmethod
    def noop_step(dependencies: list["Step"], name_alias: Optional[str]=None) -> "Step":
        """
        A convenient function to force materialize a series of Steps. Returns a NoOp Step that depends on the argument dependencies. 
        Simply call .materialize() on the returned Step to materialize the arguments. 
        """
        class NoOpNode(Step):
            def run(self, *args):
                return args
        return NoOpNode(depends_on=dependencies, name_alias=name_alias)

    # Internal methods - data flow

    def _dry_run(self):
        """
        Returns a dictionary indicating the required Steps depending on the Context status. If the Context provides the output of this Step, the output is a dictionary 
        mapping this Step name to False and its dependencies won't be included (unless they appear as computed dependencies of other Steps). Otherwise, the Step name 
        is mapped to True and will include its dependencies. 
        """
        # context
        if self.ctx is not None and str(self) in self.ctx:
            return {str(self): False}
        else:
            # on before run
            previous_output = {name: value for d in self.dependencies for name, value in d._dry_run().items()}
            this_output = {str(self): True}
            return this_output  | previous_output

    def _on_before_run(self):
        """
        Computes output of dependencies
        """
        return [d._run() for d in self.dependencies]
    
    def _run(self):
        """
        Computes output of dependencies, run this Step, and passes the output to its dependencies. 
        If a context is available, repeated dependencies are computed only once. 
        """
        if self.ctx is not None and str(self) in self.ctx:
            return self.ctx[str(self)]
        else:
            previous_otput = self._on_before_run()
            try:
                output = self.run(*previous_otput)
            except:
                warnings.warn(f"Error encountered while executing step {str(self)}")
                raise
            return self._on_after_run(output)
    
    def _on_after_run(self, output):
        """
        Cache the result to a context (if available) and passes the output. If the output is a dictionary, it is promoted to a CleverDict.
        """
        if type(output) == builtins.dict:
            output = CleverDict(output)
        self._add_to_contex(output)
        return output
    
    def _add_to_contex(self, obj):
        """
        Saves the given object to the Context
        """
        if obj is not None and self.ctx is not None:
            self.ctx[str(self)] = obj

    # Internal methods - dependency management

    def _dependencies_depth_first(self) -> List["Step"]:
        """
        Warning, may very well contain duplicates as it reflects the call hierarchy from the last node to the first node(s)
        """
        return [l2 for step in self.dependencies for l2 in [l1 for l1 in step._dependencies_depth_first()] + [step]]
    
    def _dependencies_breadth_first(self) -> List["Step"]:
        """
        Warning, may very well contain duplicates as it reflects the call hierarchy from the last node to the first node(s)
        """
        return self.dependencies + [obj for step in self.dependencies for obj in step._dependencies_breadth_first()]
    
    @staticmethod
    def _map_dependencies_obj2names(obj: List["Step"]) -> List[str]:
        """
        Convenience method to map a list of Steps to their names. For a single Step, simply call str(step_obj) instead.
        """
        return [str(x) for x in obj]

    def _assert_unique_dependency_types(self):
        depending_objects = self._dependencies_depth_first()
        depending_object_names = Step._map_dependencies_obj2names(depending_objects)
        depending_object_ids = [id(x) for x in depending_objects]               # unique by design for each object in memory
        depending_unique_objects2names = dict(zip(depending_object_ids, depending_object_names))    # keys are unique memory objects, values are Step names
        # unique_objects must correspond to unique names
        assert len(depending_unique_objects2names.keys()) == len(set(depending_unique_objects2names.values())), (
            "Dependency(ies) " + 
            f"{sorted(set([x for x in depending_unique_objects2names.values() if list(depending_unique_objects2names.values()).count(x) > 1]))} " +
            "appear more than once in the dependency graph but correspond to different objects (i.e., receiving different input/producing different output). " + 
            "These should have separate names (also to avoid conflicts in Context). Remove incorrect dependencies or use Step(name_alias) to fix it."
            )

    def __str__(self):
        return self.name_alias or self.__class__.__name__
    
