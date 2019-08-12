import inspect
from typing import List, Callable, Dict, Union, Optional, Any
import itertools

from metalearn.metafeatures.constants import ProblemType, MetafeatureGroup

class ResourceComputer:

    def __init__(
        self,
        computer: Callable,
        returns: List[str],
        argmap: Optional[Dict[str,Any]] = None
    ) -> None:
        """
        Decorates ``computer``, a resource computing function
        with metadata about that function.
        
        Parameters
        ----------
        computer
            The function that computes the resources.
        returns
            The names of the resources that ``computer`` returns, specified in
            the same order as ``computer`` returns them.
        argmap
            A custom map of ``computer``'s argument names to the global resource names
            that will be passed as ``computer``'s arguments when ``computer`` is called.
        """
        self._computer = computer
        self.returns = returns

        self.argmap = {}

        # reversing is needed because `self.defaults` gives the default
        # argument values corresponding to the *last* `n` arguments in the
        # function signature.
        reversed_args = self.args[::-1]
        reversed_defaults = self.defaults[::-1]
        arg_default_pairs = itertools.zip_longest(reversed_args, reversed_defaults)

        for local_name, default in arg_default_pairs:
            # By default, just use the `computer` function's
            # normal local argument names in the argmap,
            # making sure to preserve default argument values
            # when they are supplied.
            if default is not None:
                # The function has a default value for this arg;
                # use that.
                self.argmap[local_name] = default
            else:
                # This function has no default. Tell the system
                # to pass in the global resource identified by
                # this arg's ``local_name`` when calling this
                # ``computer``.
                self.argmap[local_name] = local_name

        if argmap is not None:
            # Now include any argument name or value overrides
            # the developer has provided. Note: each value in `argmap`
            # may be a global resource name (e.g. `"XSample"`) or
            # a direct value for the argument (e.g. `5`)
            self.argmap.update(argmap)
    
    def __call__(self, *args, **kwargs):
        """
        Allows a ``ResourceComputer`` instance to be callable.
        Just forwards all arguments on to self._computer.
        """
        return self._computer(*args, **kwargs)
    
    @property
    def args(self) -> list:
        """Returns a list of the positional parameter names of self._computer"""
        return inspect.getfullargspec(self._computer).args
    
    @property
    def defaults(self) -> list:
        """
        From https://docs.python.org/3/library/inspect.html#inspect.getfullargspec
        [Returns] an n-tuple of default argument values corresponding to the last `n`
        positional parameters [of self._computer].
        """
        defaults = inspect.getfullargspec(self._computer).defaults
        return [] if defaults is None else defaults

    @property
    def name(self) -> str:
        """Returns the function name of self._computer"""
        return self._computer.__name__


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


class ResourceComputerMap:
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
    
    def __contains__(self, key):
        """Called to implement membership test operators. e.g. `key in my_resouce_map`."""
        return key in self._map
    
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
    
    def get(self, key: str = None) -> Union[Dict[str,ResourceComputer],ResourceComputer]:
        """Used for getting the resource map."""
        if key is not None:
            return self._map[key]
        return self._map

    def _add_one(self, computer: ResourceComputer) -> None:
        if not isinstance(computer, ResourceComputer):
            raise ValueError(f"computer is not a ResourceComputer; it is a {type(computer)}")
            
        for resource_name in computer.returns:
            if resource_name in self._map:
                raise ValueError(
                    f"duplicate computer '{computer.name}' provided for resource '{resource_name}', "
                    f"which is already present in the resouce map, registered "
                    f"by computer '{self.get(resource_name).name}'"
                )
            self._map[resource_name] = computer