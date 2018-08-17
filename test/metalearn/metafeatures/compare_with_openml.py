import json
import os
import sys
import traceback

import numpy as np
import openml

from test.data.dataset import _read_arff_dataset
from metalearn import Metafeatures
from test.config import OPENML_COMPARE_RESULTS_DIR


def compare_with_openml(
    n_datasets=10, max_dataset_shape=(50000, 200), tol=.01, verbose=False
):
    """
    Compares the metafeatures computed by metalearn with those of OpenML.
    Samples `n_datasets` which are no larger than `max_dataset_shape` for this
    comparison.
    Parameters:
    -----------
    n_datasets: int, the number of openml datasets to compare
    max_dataset_shape: tuple, the maximum number of rows and columns in the
        openml datasets for which metafeatures will be compared
    verbose: bool, passed to Metafeatures.compute. Prints the name of each
        metafeature before it is computed.
    """
    dataset_ids = _list_dataset_ids(max_dataset_shape)
    np.random.shuffle(dataset_ids)

    inconsistencies = False
    successful_runs = 0
    for dataset_id in dataset_ids:
        print(f"Dataset_id {dataset_id}: ", end="")
        sys.stdout.flush()
        try:
            dataset = _download_dataset(dataset_id)
            if dataset is None:
                print(f"Invalid dataset type")
            else:
                compare_results = _compare_metafeatures(dataset, tol, verbose)
                _write_results(compare_results, dataset_id)
                if len(
                    compare_results["INCONSISTENT SHARED METAFEATURES"]
                ) > 0:
                    inconsistencies = True
                successful_runs += 1
                print(f"success #{successful_runs}.")
        except KeyboardInterrupt:
            raise
        except (
            openml.exceptions.OpenMLServerException, ValueError
        ):
            message = traceback.format_exc()
            print(f"Error:\n{message}")
            _write_results(message, dataset_id)
        if successful_runs >= n_datasets:
            break

    if inconsistencies:
        print(
            f"\nNot all metafeatures matched results from OpenML.\nResults "
            f"written to {OPENML_COMPARE_RESULTS_DIR}"
        )

def _list_dataset_ids(max_dataset_shape):
    datasets = openml.datasets.list_datasets()
    dataset_ids = []
    for d_id, mfs in datasets.items():
        n_instances = mfs.get("NumberOfInstances")
        n_features = mfs.get("NumberOfFeatures")
        if not (
            n_instances is None or n_instances > max_dataset_shape[0] or
            n_features is None or n_features > max_dataset_shape[1]
        ):
            dataset_ids.append(d_id)
    return dataset_ids

def _download_dataset(dataset_id):
    raw_dataset = openml.datasets.get_dataset(dataset_id)
    df = _read_arff_dataset(raw_dataset.data_file)
    targets = str(raw_dataset.default_target_attribute).split(",")
    if len(targets) <= 1:
        if targets[0] == "None":
            X = df
            Y = None
        else:
            X = df.drop(columns=targets, axis=1)
            Y = df[targets].squeeze()
        return {"X": X, "Y": Y, "metafeatures": raw_dataset.qualities}
    else:
        return None

def _write_results(results, dataset_id):
    if not os.path.exists(OPENML_COMPARE_RESULTS_DIR):
        os.makedirs(OPENML_COMPARE_RESULTS_DIR)
    report_name = 'openml_comparison_' + str(dataset_id) + '.json'
    with open(OPENML_COMPARE_RESULTS_DIR+report_name,'w') as fh:
        json.dump(results, fh, indent=4)

def _compare_metafeatures(oml_dataset, tol, verbose):
    # get metafeatures from dataset using our metafeatures
    our_mfs = Metafeatures().compute(
        X=oml_dataset["X"], Y=oml_dataset["Y"], verbose=verbose
    )
    oml_mfs = oml_dataset["metafeatures"]
    mf_id_map = json.load(
        open("./test/metalearn/metafeatures/oml_metafeature_map.json", "r")
    )

    oml_exclusive_mfs = {x: v for x,v in oml_dataset["metafeatures"].items()}
    our_exclusive_mfs = {}
    consistent_mfs = {}
    inconsistent_mfs = {}

    for our_mf_id, our_mf_result in our_mfs.items():
        our_mf_value = our_mf_result[Metafeatures.VALUE_KEY]
        if our_mf_id in mf_id_map:
            oml_mf_id = mf_id_map[our_mf_id]["openmlName"]
            if oml_mf_id in oml_mfs:
                oml_exclusive_mfs.pop(oml_mf_id)
                oml_mf_value = oml_mfs[oml_mf_id]
                if type(our_mf_value) is str:
                    diff = None
                else:
                    mf_multiplier = mf_id_map[our_mf_id]["multiplier"]
                    diff = abs(our_mf_value - mf_multiplier*oml_mf_value)
                comparison = {
                    our_mf_id: {
                        "openml": mf_multiplier*oml_mf_value,
                        "metalearn": our_mf_value
                    }
                }
                if diff is None or diff > tol:
                    inconsistent_mfs.update(comparison)
                else:
                    consistent_mfs.update(comparison)
            else:
                our_exclusive_mfs[our_mf_id] = our_mf_value
        else:
            our_exclusive_mfs[our_mf_id] = our_mf_value

    return {
        "INCONSISTENT SHARED METAFEATURES": inconsistent_mfs,
        "CONSISTENT SHARED METAFEATURES": consistent_mfs,
        "OUR EXCLUSIVE METAFEATURES": our_exclusive_mfs,
        "OPENML EXCLUSIVE METAFEATURES": oml_exclusive_mfs
    }
