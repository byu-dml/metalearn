from collections.abc import Mapping
import inspect
import itertools
from typing import List, Callable, Dict, Union, Optional, Any

from metalearn.metafeatures.constants import ProblemType, MetafeatureGroup


class ResourceComputer:
    """
    Decorates ``computer``, a resource computing function with metadata about that function.

    Parameters
    ----------
    computer
        The function that computes the resources.
    returns
        The names of the resources that ``computer`` returns, specified in the same order as ``computer`` returns
        them.
    argmap
        A custom map of ``computer``'s argument names to the global resource names that will be passed as
        ``computer``'s arguments when ``computer`` is called.
    """

    def __init__(
        self, computer: Callable, returns: List[str], argmap: Optional[Dict[str,Any]] = None
    ) -> None:
        argspec = inspect.getfullargspec(computer)
        # TODO: If needed, add support for `computer` functions that use these types of arguments.
        if (
            argspec.varargs is not None or argspec.varkw is not None or argspec.defaults is not None or
            len(argspec.kwonlyargs) > 0
        ):
            raise ValueError('`computer` must use only positional arguments with no default values')

        self.computer = computer
        self.returns = returns
        self.argmap = {arg_name: arg_name for arg_name in argspec.args}

        if argmap is not None:
            # override computer arg value with developer provided values
            # Note each value in `argmap` is a global resource name (e.g. `"XSample"`) or a literal value (e.g. `5`)
            self.argmap.update(argmap)
        
    
    def __call__(self, *args, **kwargs):
        """
        Allows a ``ResourceComputer`` instance to be callable. Just forwards all arguments on to self.computer.
        """
        return self.computer(*args, **kwargs)

    @property
    def name(self) -> str:
        """Returns the function name of self.computer"""
        return self.computer.__name__


class MetafeatureComputer(ResourceComputer):

    def __init__(
        self,
        computer: Callable,
        returns: List[str], # TODO: Add support for passing just a string, not a list?
        problem_type: ProblemType,
        groups: List[MetafeatureGroup],
        argmap: Optional[Dict[str,str]] = {}
    ) -> None:
        """
        Decorates ``computer``, a metafeature computing function
        with metadata about that function.
        
        Parameters
        ----------
        computer
            The function that computes the metafeatures.
        returns
            The names of the metafeatures that ``computer`` returns, specified in
            the same order as ``computer`` returns them.
        problem_type
            The type of ML problem `computer`'s metafeatures can be computed for.
        groups
            The metafeature groups this computer's returned metafeatures belong to.
            e.g. statistical, info-theoretic, simple, etc.
        argmap
            A custom map of ``computer``'s argument names to the global resource names
            that will be passed as ``computer``'s arguments when ``computer`` is called.
        """
        super(MetafeatureComputer, self).__init__(computer, returns, argmap)
        self.groups = groups
        self.problem_type = problem_type


class collectordict(Mapping):
    """
    A partially mutable mapping in which keys can be set at most one time.
    A LookupError is raised if a key is set more than once.
    For simplicity, all values must be set manually.
    """

    dict_cls = dict

    def __init__(self):
        self._dict = self.dict_cls()

    def __getitem__(self, key):
        return self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __setitem__(self, key, value):
        if key in self._dict:
            raise LookupError(f'{key} already exists')
        self._dict[key] = value

    def update(self, mapping: Mapping):
        for key, value in mapping.items():
            self[key] = value


def build_resources_info(*computers: ResourceComputer) -> collectordict:
    resources_info = collectordict()
    for computer in computers:
        for resource_name in computer.returns:
            resources_info[resource_name] = computer
    return resources_info
