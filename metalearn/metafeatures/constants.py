from enum import Enum

# Constant Enums
class ProblemType(Enum):
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'
    ANY = 'any'

class MetafeatureGroup(Enum):
    ALL = 'all'
    SIMPLE = 'simple'
    TEXT = 'text'
    STATISTICAL = 'statistical'
    INFO_THEORETIC = 'info_theoretic'
    LANDMARKING = 'landmarking'
    MODEL_BASED = 'model_based'
    TARGET_DEPENDENT = 'target_dependent'

# Constant strings
VALUE_KEY = 'value'
COMPUTE_TIME_KEY = 'compute_time'
NUMERIC = 'NUMERIC'
TEXT = 'TEXT'
CATEGORICAL = 'CATEGORICAL'
NO_TARGETS = 'NO_TARGETS'
NUMERIC_TARGETS = 'NUMERIC_TARGETS'
TIMEOUT = 'TIMEOUT'
