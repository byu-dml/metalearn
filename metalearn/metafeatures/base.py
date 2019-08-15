from collections.abc import MutableMapping
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


class ResourceComputerMap(MutableMapping):
    def __init__(self, computers: Union[ResourceComputer,List[ResourceComputer],None] = None) -> None:
        """
        Wraps a dictionary map of resource names to their computers.
        Includes visibility into whether duplicate computers
        are trying to become associated with a resource in the map e.g.
        if a package developer has accidentally declared two computers
        that return the same resource.
        """
        self._map: Dict[str,ResourceComputer] = {}
        if computers is not None:
            self.add(computers)
    
    def add(self, computers: Union[ResourceComputer,List[ResourceComputer]]) -> None:
        """
        Adds more resource name/resource computer key/value
        pairs to a resource map, throwing an error on duplicates.
        """
        if isinstance(computers, list):
            for computer in computers:
                self._add_one(computer)
        elif isinstance(computers, ResourceComputer):
            self._add_one(computers)
        else:
            raise ValueError("computers must be ResourceComputer or List[ResourceComputer]")
    
    def __getitem__(self, key: str = None) -> ResourceComputer:
        """Used for getting a resource from the map."""
        return self._map[key]

    def _add_one(self, computer: ResourceComputer) -> None:
        if not isinstance(computer, ResourceComputer):
            raise ValueError(f"computer is not a ResourceComputer; it is a {type(computer)}")
            
        for resource_name in computer.returns:
            self.__setitem__(resource_name, computer)
    
    def __setitem__(self, resource_name: str, computer: ResourceComputer):
        if resource_name in self._map:
            raise ValueError(
                f"duplicate computer '{computer.name}' provided for resource '{resource_name}', "
                f"which is already present in the resouce map, registered "
                f"by computer '{self._map[resource_name].name}'"
            )
        self._map[resource_name] = computer
    
    def __iter__(self):
        return iter(self._map)
    
    def __len__(self):
        return len(self._map)
    
    def __delitem__(self, key: str):
        raise TypeError("ResourceComputerMap does not support deletion of its ResourceComputers")