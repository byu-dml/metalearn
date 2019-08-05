import sys
import unittest

# from tests.compare_with_openml import compare_with_openml
# from tests.data.compute_dataset_metafeatures import compute_dataset_metafeatures
# from tests.benchmark_metafeatures import (
#     run_metafeature_benchmark, compare_metafeature_benchmarks
# )

if __name__ == '__main__':
    # Compare our mfs to openml's mfs
    # compare_with_openml(n_datasets=10)

    # Benchmark against another version of the code
    # run_metafeature_benchmark("start") # Run this on the branch or commit you want to benchmark against...
    # run_metafeature_benchmark("end") # ...then run this on the branch or commit you've been developing
    # compare_metafeature_benchmarks("start", "end")

    # Run unit tests
    runner = unittest.TextTestRunner(verbosity=1)
    tests = unittest.TestLoader().discover('tests')
    if not runner.run(tests).wasSuccessful():
        sys.exit(1)

    # # Compute Metafeatures
    # compute_dataset_metafeatures()
