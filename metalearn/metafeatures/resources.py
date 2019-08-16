from typing import List

from metalearn.metafeatures.base import collectordict, MetafeatureComputer, ResourceComputer

from metalearn.metafeatures.decision_tree_metafeatures import resources_info as dt_resources
from metalearn.metafeatures.general_resource_computers import resources_info as util_resources
from metalearn.metafeatures.text_metafeatures import resources_info as text_resources

from metalearn.metafeatures.simple_metafeatures import metafeatures_info as simple_metafeatures
from metalearn.metafeatures.statistical_metafeatures import metafeatures_info as statistical_metafeatures
from metalearn.metafeatures.information_theoretic_metafeatures import metafeatures_info as info_theoretic_metafeatures
from metalearn.metafeatures.landmarking_metafeatures import metafeatures_info as landmarking_metafeatures
from metalearn.metafeatures.text_metafeatures import metafeatures_info as text_metafeatures
from metalearn.metafeatures.decision_tree_metafeatures import metafeatures_info as dt_metafeatures


resources_info = collectordict()

# Add all the ResourceComputers
resources_info.update(dt_resources)
resources_info.update(util_resources)
resources_info.update(text_resources)

# Add noop resource computers for the base resources.
# Since they'll always be in the Metafeatures resource hash,
# they'll never be needed to be computed by a ResourceComputer,
# but they need to be in `resources_info` since `Metafeatures._get_arguments`
# and `Metafeatures._resource_is_target_dependent` requires them to be.
resources_info["X_raw"] = ResourceComputer(lambda _: None, ["X_raw"])
resources_info["X"] = ResourceComputer(lambda _: None, ["X"])
resources_info["Y"] = ResourceComputer(lambda _: None, ["Y"])
resources_info["column_types"] = ResourceComputer(lambda _: None, ["column_types"])
resources_info["sample_shape"] = ResourceComputer(lambda _: None, ["sample_shape"])
resources_info["seed_base"] = ResourceComputer(lambda _: None, ["seed_base"])
resources_info["n_folds"] = ResourceComputer(lambda _: None, ["n_folds"])

# Add all the MetafeatureComputers
resources_info.update(simple_metafeatures)
resources_info.update(statistical_metafeatures)
resources_info.update(info_theoretic_metafeatures)
resources_info.update(landmarking_metafeatures)
resources_info.update(text_metafeatures)
resources_info.update(dt_metafeatures)

# Get all the metafeature ids
metafeature_ids = [
    mf_id for mfs_info in [
        simple_metafeatures,
        statistical_metafeatures,
        info_theoretic_metafeatures,
        landmarking_metafeatures,
        text_metafeatures,
        dt_metafeatures,
    ] for mf_id in mfs_info.keys()
]
