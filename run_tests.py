import unittest

# import test.data
# from test.metalearn.metafeatures.compare_with_openml import compare_with_openml
from test.metalearn.metafeatures.test_metafeatures import metafeatures_suite
# from test.metalearn.metafeatures.benchmark_metafeatures import (
#     run_metafeature_benchmark, compare_metafeature_benchmarks
# )


if __name__ == '__main__':
    # test.data.initialize()
    # compare_with_openml(10)
    unittest.TextTestRunner().run(metafeatures_suite())
    # run_metafeature_benchmark("start")
    # run_metafeature_benchmark("end")
    # compare_metafeature_benchmarks("start", "end")
