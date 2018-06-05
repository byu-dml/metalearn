import unittest

from test.metalearn.metafeatures.test_metafeatures import metafeatures_suite
from test.data.compute_dataset_metafeatures import compute_dataset_metafeatures

if __name__ == '__main__':
    # compute_dataset_metafeatures()
    unittest.TextTestRunner().run(metafeatures_suite())
