# Contributing to `metalearn`

## Adding New Metafeatures

`metalearn` uses a caching mechanism to cache expensive computations that may need to be used again within the package by another function. Both resources (e.g. the dataset itself or a preprocessed version of it) and metafeatures (e.g. entropy, number of features) are cached by the system.

When adding a new metafeature to the package, the function that computes that metafeature needs to be registered in the `resources_info` variable in [./metalearn/metafeatures/resources.py](./metalearn/metafeatures/resources.py), and passed to the call made to `_get_metafeature_ids` in that module as well. Before the function can be registered and passed though, it needs to be decorated with metadata by being passed through the `MetafeatureComputer` constructor (see example below). This allows the metafeatures returned by the function to be used intelligently by the package.

Follow the example below to know how to write and register new metafeature(s). Note that a metafeature-computing function (e.g. `get_dataset_stats` as seen below) can compute and return more than one meta-feature.

```python
# Import needed utilities
from metalearn.metafeatures.base import MetafeatureComputer
from metalearn.metafeatures.constants import ProblemType, MetafeatureGroup

# Declare the function that computes the metafeatures.
def get_dataset_stats(X, column_types):
    # Calculate metafeatures.
    number_of_instances = X.shape[0]
    number_of_features = X.shape[1]
    # Return a tuple (here it's two metafeatures).
    return (number_of_instances, number_of_features)

# Decorate the metafeature-computing function with data
# the package will use.
get_dataset_stats = MetafeatureComputer(
    # Pass the function into the `MetafeatureComputer`
    # decorator.
    computer=get_dataset_stats,
    # Give each metafeature returned by the function a
    # name for the cache to use (order here must match the
    # order they are returned in by `computer`).
    returns=[
        "NumberOfInstances",
        "NumberOfFeatures"
    ],
    # Associate a problem type with the new metafeatures.
    problem_type=ProblemType.ANY,
    # Associate one or more metafeature groups.
    groups=[MetafeatureGroup.SIMPLE],
    # Specify which values to pass to the function
    # when calling it to compute the metafeatures.
    # Here we are passing the cached resource called
    # "X_raw" as the value for this function's "X" argument.
    argmap={ "X": "X_raw" }
)
```

By convention, all the decorated metafeature-computing functions in a module are aggregated at the bottom of the module into a list called `metafeature_computers`, which is then imported by [./metalearn/metafeatures/resources.py](./metalearn/metafeatures/resources.py) and added to that module's `resources_info` variable.