import json
import os
import time

import pandas as pd

from metalearn.metafeatures.metafeatures import Metafeatures
from test.config import CORRECTNESS_SEED, METAFEATURES_DIR, METADATA_PATH
from .dataset import read_dataset


def get_dataset_metafeatures_path(dataset_filename):
    dataset_name = dataset_filename.rsplit(".", 1)[0]
    return os.path.join(METAFEATURES_DIR, dataset_name+"_mf.json")

def compute_dataset_metafeatures():
    metadata = json.load(open(METADATA_PATH, "r"))
    for dataset_metadata in metadata:
        dataset_filename = dataset_metadata["filename"]
        target_class_name = dataset_metadata["target_class_name"]
        index_col_name = dataset_metadata.get("index_col_name", None)

        choice = None
        while not choice in ["y", "v", "n"]:
            choice = input(dataset_filename + " [(y)es, (v)erbose, (n)o]: ")

        if choice == "n":
            continue

        X, Y, column_types = read_dataset(dataset_filename, index_col_name, target_class_name)

        start_time = time.time()
        metafeatures = Metafeatures().compute(X=X, Y=Y, seed=CORRECTNESS_SEED)
        run_time = time.time() - start_time

        if choice == "v":
            print(json.dumps(metafeatures, sort_keys=True, indent=4))
        print("Runtime: " + str(run_time))

        choice = None
        while not choice in ["y", "n"]:
            choice = input(f"Update {dataset_filename} metafeatures? [(y)es, (n)o]: ")
        if choice == "y":
            mf_file_path = get_dataset_metafeatures_path(dataset_filename)
            json.dump(metafeatures, open(mf_file_path, "w"), sort_keys=True, indent=4)
