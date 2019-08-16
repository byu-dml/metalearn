"""
Use to run various testing modules of the package.
The default is to run the unit tests alone. To use,
run this script from the terminal, appending the names
of whatever modules you want to run. Appending no names
will run the unit tests by default. Example usage:

    `python run_tests.py openml benchmark-start`

This will run both the `compare_with_openml` (openml)
and `run_metafeature_benchmark('start')` (benchmark-start) modules.
"""

import sys
import unittest
import argparse


def get_modules_to_run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'modules',
        nargs = '*',
        choices=[
            'unittests',
            'benchmark-start',
            'benchmark-end',
            'benchmark-compare',
            'openml',
            'compute-mfs'
        ],
        default='unittests',
        help = 'Which modules to run'
    )

    modules = parser.parse_args().modules
    if isinstance(modules, str):
        modules = [modules]
    return modules


if __name__ == '__main__':
    chosen_modules = get_modules_to_run()

    if 'openml' in chosen_modules:
        print('Comparing our mfs to openml\'s mfs...')
        from tests.compare_with_openml import compare_with_openml
        compare_with_openml(n_datasets=10)

    # Run this on a branch or commit you want to benchmark against...
    if 'benchmark-start' in chosen_modules:
        print('Starting a benchmark...')
        from tests.benchmark_metafeatures import run_metafeature_benchmark
        run_metafeature_benchmark('start') 
    
    # ...then run this on the branch or commit you've been developing
    if 'benchmark-end' in chosen_modules:
        print('Ending a benchmark...')
        from tests.benchmark_metafeatures import run_metafeature_benchmark
        run_metafeature_benchmark('end') 
         
    if 'benchmark-compare' in chosen_modules:
        print('Comparing benchmarks...')
        from tests.benchmark_metafeatures import compare_metafeature_benchmarks
        compare_metafeature_benchmarks('start', 'end')

    if 'compute-mfs' in chosen_modules:
        print('Computing the dataset metafeatures...')
        from tests.data.compute_dataset_metafeatures import compute_dataset_metafeatures
        compute_dataset_metafeatures()

    if 'unittests' in chosen_modules:
        print('Running unit tests...')
        runner = unittest.TextTestRunner(verbosity=1)
        tests = unittest.TestLoader().discover('tests')
        if not runner.run(tests).wasSuccessful():
            sys.exit(1)
