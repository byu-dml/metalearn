import openml
import json
import os
import arff
import numpy as np
import traceback
import sys

from test.data.dataset import _read_arff_dataset
from metalearn import Metafeatures

FILE_PATH = './test/metalearn/metafeatures/openmlComparisons/'

def _list_dataset_ids(max_features, max_instances):
    datasets_dict = openml.datasets.list_datasets()
    datasets = [
        k for k,v in datasets_dict.items() if 
        ("NumberOfInstances" in v.keys() and v["NumberOfInstances"] <= max_instances) and 
        ("NumberOfFeatures" in v.keys() and v["NumberOfFeatures"] <= max_features)
    ]
    return datasets

def _download_dataset(dataset_id):
    raw_dataset = openml.datasets.get_dataset(dataset_id)
    target = str(raw_dataset.default_target_attribute).split(",")
    df = _read_arff_dataset(raw_dataset.data_file)
    if len(target) <= 1:
        if target[0] == "None":
            X = df
            Y = None
        else:
            X = df.drop(columns=target, axis=1)
            Y = df[target].squeeze()
        dataset_metafeatures = {x: (float(v) if v is not None else v) for x,v in raw_dataset.qualities.items()}
        dataset = {"X": X, "Y": Y, "metafeatures": dataset_metafeatures}
        return dataset

def _write_results(results, dataset_id):
    if not os.path.exists(FILE_PATH):
        os.makedirs(FILE_PATH)
    report_name = 'openml_comparison_' + str(dataset_id) + '.json'
    with open(FILE_PATH+report_name,'w') as fh:
        json.dump(results, fh, indent=4)

def compare_metafeatures(oml_dataset, dataset_id, verbose):
    # get metafeatures from dataset using our metafeatures
    ourMetafeatures = Metafeatures().compute(X=oml_dataset["X"], Y=oml_dataset["Y"], verbose=verbose)

    mfNameMap = json.load(open("test/metalearn/metafeatures/oml_metafeature_map.json", "r"))

    omlExclusiveMf = {x: v for x,v in oml_dataset["metafeatures"].items()}
    ourExclusiveMf = {}
    consistentSharedMf = []
    inconsistentSharedMf = []

    for metafeatureName, metafeatureValue in ourMetafeatures.items():
        metafeatureValue = metafeatureValue["value"]
        if 'int' in str(type(metafeatureValue)):
            metafeatureValue = int(metafeatureValue)
        elif 'float' in str(type(metafeatureValue)):
            metafeatureValue = float(metafeatureValue)

        if mfNameMap.get(metafeatureName) is None:
            ourExclusiveMf[metafeatureName] = metafeatureValue
        else:
            openmlName = mfNameMap[metafeatureName]["openmlName"]
            if oml_dataset["metafeatures"].get(openmlName) is None:
                ourExclusiveMf[metafeatureName] = metafeatureValue
            else:
                omlExclusiveMf.pop(openmlName)
                omlMetafeatureValue = oml_dataset["metafeatures"][openmlName]
                multiplier = mfNameMap[metafeatureName]["multiplier"]
                if metafeatureValue == Metafeatures().NUMERIC_TARGETS or metafeatureValue == Metafeatures().NO_TARGETS:
                    diff = float("NaN")
                else:
                    diff = abs(omlMetafeatureValue/multiplier - metafeatureValue)
                singleMfDict = {metafeatureName: {"OpenML Value": omlMetafeatureValue/multiplier,
                                                  "Our Value": metafeatureValue, "Difference": diff}
                                }
                if diff <= .05:
                    consistentSharedMf.append(singleMfDict)
                elif diff > .05 or diff == np.isnan(diff):
                    inconsistentSharedMf.append(singleMfDict)

    # write results to json file
    openmlData = { "INCONSISTENT SHARED METAFEATURES": inconsistentSharedMf,
                   "CONSISTENT SHARED METAFEATURES": consistentSharedMf,
                   "OUR EXCLUSIVE METAFEATURES": ourExclusiveMf,
                   "OPENML EXCLUSIVE METAFEATURES": omlExclusiveMf}

    _write_results(openmlData, dataset_id)

    if len(inconsistentSharedMf) > 0:
        return True
    else:
        return False

def compare_with_openml(runs=1, max_features=200, max_instances=50000, verbose=False):
    # get a list of datasets from openml
    datasets = _list_dataset_ids(max_features, max_instances)

    inconsistencies = False
    successful_runs = 0
    sample_size = runs
    while successful_runs < sample_size:
        dataset_id = np.random.choice(datasets, replace = False)
        print(f"Dataset_id {dataset_id}: ", end="")
        sys.stdout.flush()
        try:
            dataset = _download_dataset(dataset_id)
            if dataset is None:
                print(f"Invalid dataset type")
            else:
                if compare_metafeatures(dataset, dataset_id, verbose):
                    inconsistencies = True
                successful_runs += 1
                print(f"Succeeded. Runs = {successful_runs}")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Error: {e}")
            _write_results(traceback.format_exc(), dataset_id)
                
    if inconsistencies:
        print(f"\nNot all metafeatures matched results from OpenML.\nResults written to {FILE_PATH}")       
