import pkg_resources

# Expose the `metafeatures_schema.json` file as a file path string.
# Needed to allow the file to be successfully accessed across platforms 
# and install types.
METAFEATURES_JSON_SCHEMA_PATH = pkg_resources.resource_filename('metalearn', 'metafeatures/metafeatures_schema.json')
