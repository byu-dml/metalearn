import unittest
from test.metalearn.metafeatures.test_metafeatures import metafeatures_suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(metafeatures_suite())
