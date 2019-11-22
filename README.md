| **Develop**                                                                                                                                  | **Master**                                                                                                                                 |
|----------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| [![Build Status](https://travis-ci.org/byu-dml/metalearn.svg?branch=develop)](https://travis-ci.org/byu-dml/metalearn)                       | [![Build Status](https://travis-ci.org/byu-dml/metalearn.svg?branch=master)](https://travis-ci.org/byu-dml/metalearn)                      |
| [![codecov](https://codecov.io/gh/byu-dml/metalearn/branch/develop/graph/badge.svg)](https://codecov.io/gh/byu-dml/metalearn/branch/develop) | [![codecov](https://codecov.io/gh/byu-dml/metalearn/branch/master/graph/badge.svg)](https://codecov.io/gh/byu-dml/metalearn/branch/master) |

# Metalearn

The Data Mining Lab of the Computer Science Department of Brigham Young University (BYU-DML) python3 library of meta-learning tools.
Extracts **general**, **statistical**, **information-theoretic**, **landmarking** and **model-based** meta-features from tabular datasets for use in meta-learning applications.

## Installation

### Using pip:  

`pip install metalearn`

### From source:

```bash
git clone https://github.com/byu-dml/metalearn.git
cd metalearn
python3 setup.py install
```

Note that this project follows the versioning scheme defined by [Semantic Versioning 2.0.0](https://semver.org). This means (among other things) that a given version of the package has _zero guarantee of backwards compatibility_ with previous major versions.

## Example Usage

### Simple Example
```python
from metalearn import Metafeatures
import pandas as pd
import numpy as np

# X and Y must be a pandas DataFrame and a pandas Series respectively
X = pd.DataFrame(np.random.rand(8,2))
Y = pd.Series(['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'], name='targets')

metafeatures = Metafeatures()
mfs = metafeatures.compute(X, Y)
```

### Full Example
```python
import pandas as pd
import numpy as np
from metalearn import Metafeatures

data = pd.DataFrame({
    'cat': np.random.choice(['a', 'b'], size=20),
    'num': np.random.rand(20),
    'targets': np.random.choice(['x', 'y'], size=20)
})

X = data.drop('targets', axis=1)
Y = data['targets']

metafeatures = Metafeatures()
mfs = metafeatures.compute(
    X,
    Y=Y,
    column_types={'cat': 'CATEGORICAL', 'num': 'NUMERIC', 'targets': 'CATEGORICAL'},
    metafeature_ids=['RatioOfNumericFeatures'],
    exclude=None,
    sample_shape=(8, None),
    seed=0,
    n_folds=2,
    verbose=True,
    timeout=10,
    return_times=True,
)

print(mfs)

# RatioOfNumericFeatures
# {'RatioOfNumericFeatures': {'value': 0.5, 'compute_time': 3.9138991269283e-05}}
```
**Warning:** Metafeatures are timed as if each dependency has to be recomputed whenever it is needed.
This means that the returned times may not be accurate for a particular application, especially if a 
metafeature depends on a computationally intensive resource in multiple places.

## Using the Test Suite

Using this cloned or downloaded repository, the tests can be run with:
```
pip install -r requirements.txt
python3 run_tests.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgements

Many of the metafeatures in this package were inspired by the work done in the R project [mfe](https://github.com/rivolli/mfe)
