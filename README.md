[![Build Status](https://api.travis-ci.org/byu-dml/metalearn.png)](https://travis-ci.org/byu-dml/metalearn)

# Metalearn

The Data Mining Lab of the Computer Science Department of Brigham Young
University (BYU-DML) python library of meta-learning tools. Extracts **general**, **statistical**, **information-theoretic**, **landmarking** and **model-based** meta-features from tabular datasets for use in meta-learning applications.

## Installing

Metalearn's dependencies will be installed with this package, including:
- pandas
- numpy
- scikit-learn

### Using pip:  

`pip install metalearn`

### From source:

```bash
git clone https://github.com/byu-dml/metalearn.git
cd metalearn
python setup.py install
```
## Using Metalearn
  
```python
from metalearn import Metafeatures
import pandas as pd
import numpy as np

X = pd.DataFrame(np.arange(16).reshape(4,4))
Y = pd.Series(['a', 'a', 'b', 'b'])

metafeatures = Metafeatures()  # instantiate an instance of the base class Metafeatures
mfs = metafeatures.compute(X, Y)
```

## Using the Test Suite

In order for the test suite to run properly, our package requires Python3.6.  
You can install the requirements for the test suite using:  
`pip install -r requirements.txt`
  
The repository also contains code to compare our metafeatures against ones computed by [OpenML](https://github.com/openml/OpenML).  
In order to use it, first install the requirements for OpenML by running:  
`pip install -r openml_requirements.txt`  

All test code is run from the file `run_tests.py` with the command:  
`python run_tests.py`

### Unit Tests

To use the automated unit tests, uncomment these lines before running:  
```python
runner = unittest.TextTestRunner(verbosity=1)
tests = unittest.TestLoader().discover('tests')
if not runner.run(tests).wasSuccessful():
    sys.exit(1)
```

### OpenML Comparison

To compare our computed metafeatures against the ones computed by OpenML, uncomment these lines before running:  
```python
from tests.compare_with_openml import compare_with_openml

compare_with_openml(n_datasets=10)
```

### Benchmarking

to benchmark the metafeatures before and after making changes, uncomment these lines before running:  
```python
from tests.data.compute_dataset_metafeatures import compute_dataset_metafeatures
from tests.benchmark_metafeatures import (
    run_metafeature_benchmark, compare_metafeature_benchmarks
)

compute_dataset_metafeatures()
run_metafeature_benchmark("start")
run_metafeature_benchmark("end")
compare_metafeature_benchmarks("start", "end")
```
