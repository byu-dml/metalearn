[![Build Status](https://api.travis-ci.org/byu-dml/metalearn.png)](https://travis-ci.org/byu-dml/metalearn)

# metalearn
The Data Mining Lab of the Computer Science Department of Brigham Young
University (BYU-DML) python library of meta-learning tools.

Currently extracts meta-features from tabular datasets with categorical
targets.

This package is installable from pypi using:  
`pip install metalearn`

Example:  
```python
from metalearn import Metafeatures
import pandas as pd

X = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
Y = pd.Series(['a', 'b', 'c'])

metafeatures = Metafeatures()  # instantiate an instance of the base class Metafeatures
mfs = Metafeatures.compute(X, Y)
```

  
In order for the test suite to run properly, our package requires Python3.6.  
You can install the requirements for the test suite using:  
`pip install -r requirements.txt`
  
  
The repository also contains code to compare our metafeatures against ones computed by [OpenML](https://github.com/openml/OpenML).  
In order to use it, first install the requirements by running:  
`pip install -r openml_requirements.txt`  
