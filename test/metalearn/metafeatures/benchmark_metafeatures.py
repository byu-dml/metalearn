import json
import time

import numpy as np

from metalearn import Metafeatures
from test.data.dataset import read_dataset
from test.config import CORRECTNESS_SEED, METADATA_PATH


def get_benchmark_path(benchmark_name):
    return f"./{benchmark_name}.json"

def write_benchmark_data(benchmark_name, benchmark_data):
    json.dump(
        benchmark_data, open(get_benchmark_path(benchmark_name), "w"),
        indent=4, sort_keys=True
    )

def read_benchmark_data(benchmark_name):
    return json.load(open(get_benchmark_path(benchmark_name), "r"))

def run_metafeature_benchmark(benchmark_name, iters=100):
    """
    Computes metafeatures `iters` times over the test datasets and stores
    comparable information in ./<benchmark_name>.json.
    """
    with open(METADATA_PATH, "r") as f:
        dataset_descriptions = json.load(f)
    benchmark_data = {}
    for dataset_metadata in dataset_descriptions:
        print(dataset_metadata["filename"])
        X, Y, column_types = read_dataset(dataset_metadata)
        init_times = []
        total_compute_times = []
        metafeature_compute_times = {
            mf_id: [] for mf_id in Metafeatures.IDS
        }
        for i in range(iters):
            print(f"iter {i}")
            start_timestamp = time.time()
            mf = Metafeatures()
            init_timestamp = time.time()
            computed_mfs = mf.compute(
                X=X, Y=Y, column_types=column_types, seed=CORRECTNESS_SEED
            )
            compute_timestamp = time.time()
            init_times.append(init_timestamp - start_timestamp)
            total_compute_times.append(compute_timestamp - init_timestamp)
            for mf_id, result in computed_mfs.items():
                metafeature_compute_times[mf_id].append(
                    result[Metafeatures.COMPUTE_TIME_KEY]
                )
        benchmark_data[dataset_metadata["filename"]] = {
            "init_time": {
                "mean": np.mean(init_times),
                "std_dev": np.std(init_times)
            },
            "total_compute_time": {
                "mean": np.mean(total_compute_times),
                "std_dev": np.std(total_compute_times)
            },
            "metafeature_compute_time": {
                mf_id: {
                    "mean": np.mean(mf_times),
                    "std_dev": np.std(mf_times)
                } for mf_id, mf_times in metafeature_compute_times.items()
            }
        }
    write_benchmark_data(benchmark_name, benchmark_data)

def compare_metafeature_benchmarks(bm_1_name, bm_2_name, n_std_dev=3):
    """
    Compares two benchmark tests from the run_metafeature_benchmark function.
    Identifies significant differences in individual metafeature computation
    time and aggregate computation time. Reports the items that are more than
    n_std_dev standard deviations different from each other. Does not test for
    correctness.
    """

    def compare(item_name, bm_1, bm_2):
        if (bm_2["mean"] - bm_1["mean"]) < -bm_1["std_dev"] * n_std_dev:
            rel_imp = (bm_2["mean"] - bm_1["mean"]) / -bm_1["std_dev"]
            print(f"{item_name} faster by {rel_imp} standard deviations")
        if (bm_1["mean"] - bm_2["mean"]) < -bm_2["std_dev"] * n_std_dev:
            rel_imp = (bm_1["mean"] - bm_2["mean"]) / -bm_2["std_dev"]
            print(f"{item_name} slower by {rel_imp} standard deviations")

    bm_1_data = read_benchmark_data(bm_1_name)
    bm_2_data = read_benchmark_data(bm_2_name)
    for dataset_filename, benchmark_data_1 in bm_1_data.items():
        print(f"{dataset_filename} benchmarks beginning")
        benchmark_data_2 = bm_2_data[dataset_filename]
        compare(
            "init_time", benchmark_data_1["init_time"],
            benchmark_data_2["init_time"]
        )
        compare(
            "total_compute_time", benchmark_data_1["total_compute_time"],
            benchmark_data_2["total_compute_time"]
        )
        for mf_id, bm_1 in benchmark_data_1["metafeature_compute_time"].items():
            compare(
                mf_id, bm_1,
                benchmark_data_2["metafeature_compute_time"][mf_id]
            )
        print(f"{dataset_filename} benchmarks finished")
