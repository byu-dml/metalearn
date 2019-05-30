"""
Exposes all the package's bundled data files as file path strings.
Needed to allow the data files to be successfully accessed across platforms 
and install types.
"""

import pkg_resources

METAFEATURE_CONFIG = pkg_resources.resource_filename('metalearn', 'metafeatures/metafeatures.json')
METAFEATURES_JSON_SCHEMA = pkg_resources.resource_filename('metalearn', 'metafeatures/metafeatures_schema.json')