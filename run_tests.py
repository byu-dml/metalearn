import sys
import unittest

# from tests.compare_with_openml import compare_with_openml
# from tests.data.compute_dataset_metafeatures import compute_dataset_metafeatures
# from tests.benchmark_metafeatures import (
#     run_metafeature_benchmark, compare_metafeature_benchmarks
# )

if __name__ == '__main__':
    # Compare our mfs to openml's mfs
    # compare_with_openml(10)

    # Run benchmarks
    # compute_dataset_metafeatures()
    # run_metafeature_benchmark("start")
    # run_metafeature_benchmark("end")
    # compare_metafeature_benchmarks("start", "end")

    # Run unit tests
    runner = unittest.TextTestRunner(verbosity=1)
    tests = unittest.TestLoader().discover('tests')
    if not runner.run(tests).wasSuccessful():
        sys.exit(1)
