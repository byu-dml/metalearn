import unittest

# from test.data.compute_dataset_metafeatures import compute_dataset_metafeatures
from test.metalearn.metafeatures.test_metafeatures import metafeatures_suite

if __name__ == '__main__':
    # compute_dataset_metafeatures()
    unittest.TextTestRunner().run(metafeatures_suite())
