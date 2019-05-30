[![Build Status](https://api.travis-ci.org/byu-dml/metalearn.png)](https://travis-ci.org/byu-dml/metalearn)

# Metalearn

The Data Mining Lab of the Computer Science Department of Brigham Young University (BYU-DML) python3 library of meta-learning tools.
Extracts **general**, **statistical**, **information-theoretic**, **landmarking** and **model-based** meta-features from tabular datasets for use in meta-learning applications.

## Installation

Dependencies installed with this package include:
- pandas
- numpy
- scikit-learn

### Using pip:  

`pip install metalearn`

### From source:

```bash
git clone https://github.com/byu-dml/metalearn.git
cd metalearn
python3 setup.py install
```
## Example Usage
  
```python
from metalearn import Metafeatures
import pandas as pd
import numpy as np

# X and Y must be a pandas DataFrame and a pandas Series respectively
X = pd.DataFrame(np.random.rand(8,2))
Y = pd.Series(['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'])

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
`python3 run_tests.py`

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

Tracks the difference in speed of computation before and after changes. Make sure the imports are uncommented before running:
```python
from tests.data.compute_dataset_metafeatures import compute_dataset_metafeatures
from tests.benchmark_metafeatures import (
    run_metafeature_benchmark, compare_metafeature_benchmarks
)
```
Before making any changes, uncomment the line:
```python
run_metafeature_benchmark("start")
```
to save computation times.

To compare, recomment out the previous line and uncomment out the following.
```python
run_metafeature_benchmark("end")
compare_metafeature_benchmarks("start", "end")
```
This will compare the two files and create a new file with the differences and their standard deviations

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/byu-dml/metalearn/blob/develop/LICENSE) file for details

## Acknowledgements

Many of the metafeatures in this package were inspired by the work done in the R project [mfe](https://github.com/rivolli/mfe)
