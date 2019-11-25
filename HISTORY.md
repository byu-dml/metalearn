## 0.6.0

* Added selecting and filter of metafeatures by semantic groups
* Updated dependency versions for scikit-learn, numpy, and pandas
* Identified a bug in calculating metafeature compute time, so compute time is not returned by default
* Reorganized test suite
* Added custom function decorators for easier internal maintainability of computable metafeatures that replaced the metafeatures.json file
* Added usage examples to README
* Added build status to README
* Updated license to 2019
* Added releasing documentation
* Added contributing documentation
* Added mypy check to CI

## 0.5.3

* Added Travis-CI
* Handle empty X inputs
* Handle X and y shape mismatch
* Use SVD solver for Linear Discriminant Analysis landmarker to fix index out of bounds error
