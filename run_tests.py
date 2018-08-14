import unittest

# from test.data.compute_dataset_metafeatures import compute_dataset_metafeatures
from test.metalearn.metafeatures.test_metafeatures import metafeatures_suite
# from test.metalearn.metafeatures.benchmark_metafeatures import (
#     run_metafeature_benchmark, compare_metafeature_benchmarks
# )

if __name__ == '__main__':
    # compute_dataset_metafeatures()
    unittest.TextTestRunner().run(metafeatures_suite())
    # run_metafeature_benchmark("start")
    # run_metafeature_benchmark("end")
    # compare_metafeature_benchmarks("start", "end")
