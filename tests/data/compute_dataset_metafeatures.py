import json
import os
import time

import numpy as np
import pandas as pd

from metalearn import Metafeatures
from tests.config import CORRECTNESS_SEED, METAFEATURES_DIR, METADATA_PATH
from .dataset import read_dataset


def get_dataset_metafeatures_path(dataset_filename):
    dataset_name = dataset_filename.rsplit(".", 1)[0]
    return os.path.join(METAFEATURES_DIR, dataset_name+"_mf.json")


def is_close(computed_value, known_value):
    if type(known_value) is str:
        correct = known_value == computed_value
    else:
        correct = np.array(np.isclose(known_value, computed_value, equal_nan=True)).all()
    return correct


def compute_dataset_metafeatures():
    metadata = json.load(open(METADATA_PATH, "r"))
    for dataset_metadata in metadata:
        dataset_filename = dataset_metadata["filename"]

        choice = None
        while not choice in ["y", "v", "n"]:
            choice = input(dataset_filename + " [(y)es, (v)erbose, (n)o]: ")

        if choice == "n":
            continue

        X, Y, column_types = read_dataset(dataset_metadata)

        start_time = time.time()
        computed_mfs = Metafeatures().compute(X=X, Y=Y, column_types=column_types, seed=CORRECTNESS_SEED)
        run_time = time.time() - start_time

        if choice == "v":
            known_mf_path = get_dataset_metafeatures_path(dataset_filename)
            with open(known_mf_path, 'r') as fp:
                known_mfs = json.load(fp)

            new_mfs = {}
            deleted_mfs = {}
            updated_mfs = {}
            same_mfs = {}
            all_mf_names = set(list(computed_mfs.keys()) + list(known_mfs.keys()))
            for mf in all_mf_names:
                if mf not in known_mfs.keys():
                    new_mfs[mf] = computed_mfs[mf]
                elif mf not in computed_mfs.keys():
                    deleted_mfs[mf] = known_mfs[mf]
                elif is_close(computed_mfs[mf]['value'], known_mfs[mf]['value']):
                    same_mfs[mf] = computed_mfs[mf]
                else:
                    updated_mfs[mf] = {'known': known_mfs[mf], 'computed': computed_mfs[mf]}

            print('UNCHANGED METAFEATURES')
            print(json.dumps(same_mfs, sort_keys=True, indent=4))
            print('DELETED METAFEATURES')
            print(json.dumps(deleted_mfs, sort_keys=True, indent=4))
            print('NEW METAFEATURES')
            print(json.dumps(new_mfs, sort_keys=True, indent=4))
            print('UPDATED METAFEATURES')
            print(json.dumps(updated_mfs, sort_keys=True, indent=4))

        print("Runtime: " + str(run_time))

        choice = None
        while not choice in ["y", "n"]:
            choice = input(f"Update {dataset_filename} metafeatures? [(y)es, (n)o]: ")
        if choice == "y":
            mf_file_path = get_dataset_metafeatures_path(dataset_filename)
            with open(mf_file_path, 'w') as fp:
                json.dump(computed_mfs, fp, sort_keys=True, indent=4)
