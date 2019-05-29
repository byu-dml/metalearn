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

  
In order for the test suite to run properly, our package requires Python3.6.  
You can install the requirements for the test suite using:  
`pip install -r requirements.txt`
  
  
The repository also contains code to compare our metafeatures against ones computed by [OpenML](https://github.com/openml/OpenML).  
In order to use it, first install the requirements by running:  
`pip install -r openml_requirements.txt`  
