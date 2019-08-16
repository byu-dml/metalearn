from typing import List

from metalearn.metafeatures.base import ResourceComputerMap, MetafeatureComputer, ResourceComputer

from metalearn.metafeatures.decision_tree_metafeatures import resource_computers as dt_resources
from metalearn.metafeatures.general_resource_computers import resource_computers as util_resources
from metalearn.metafeatures.text_metafeatures import resource_computers as text_resources

from metalearn.metafeatures.simple_metafeatures import metafeature_computers as simple_metafeatures
from metalearn.metafeatures.statistical_metafeatures import metafeature_computers as statistical_metafeatures
from metalearn.metafeatures.information_theoretic_metafeatures import metafeature_computers as info_theoretic_metafeatures
from metalearn.metafeatures.landmarking_metafeatures import metafeature_computers as landmarking_metafeatures
from metalearn.metafeatures.text_metafeatures import metafeature_computers as text_metafeatures
from metalearn.metafeatures.decision_tree_metafeatures import metafeature_computers as dt_metafeatures


def _get_metafeature_ids(metafeature_computers: List[MetafeatureComputer]) -> List[str]:
    """Returns a list of all metafeature IDs found in `metafeature_computers`"""
    metafeature_ids = set()
    for computer in metafeature_computers:
        for name in computer.returns:
            if name in metafeature_ids:
                raise ValueError("there is already a MetafeatureComputer that returns the {name} metafeature.")
            metafeature_ids.add(name)
    return list(metafeature_ids)


resources_info = ResourceComputerMap()

# Add all the ResourceComputers
resources_info.add(dt_resources)
resources_info.add(util_resources)
resources_info.add(text_resources)

# Add noop resource computers for the base resources.
# Since they'll always be in the Metafeatures resource hash,
# they'll never be needed to be computed by a ResourceComputer,
# but they need to be in `resources_info` since `Metafeatures._get_arguments`
# and `Metafeatures._resource_is_target_dependent` requires them to be.
resources_info.add(ResourceComputer(lambda _: None, ["X_raw"]))
resources_info.add(ResourceComputer(lambda _: None, ["X"]))
resources_info.add(ResourceComputer(lambda _: None, ["Y"]))
resources_info.add(ResourceComputer(lambda _: None, ["column_types"]))
resources_info.add(ResourceComputer(lambda _: None, ["sample_shape"]))
resources_info.add(ResourceComputer(lambda _: None, ["seed_base"]))
resources_info.add(ResourceComputer(lambda _: None, ["n_folds"]))

# Add all the MetafeatureComputers
resources_info.add(simple_metafeatures)
resources_info.add(statistical_metafeatures)
resources_info.add(info_theoretic_metafeatures)
resources_info.add(landmarking_metafeatures)
resources_info.add(text_metafeatures)
resources_info.add(dt_metafeatures)

# Get all the metafeature ids
metafeature_ids = _get_metafeature_ids(
    simple_metafeatures +
    statistical_metafeatures +
    info_theoretic_metafeatures +
    landmarking_metafeatures +
    text_metafeatures +
    dt_metafeatures
)
