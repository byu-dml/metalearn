import json
import os
import time

import pandas as pd

from metalearn.metafeatures.metafeatures import Metafeatures
from test.config import CORRECTNESS_SEED, METAFEATURES_DIR, METADATA_PATH
from .dataset import read_dataset


def extract_metafeatures(X, Y, seed=None):
    metafeatures = {}
    features_df = Metafeatures().compute(X=X, Y=Y, seed=seed)
    for feature in features_df.columns:
        metafeatures[feature] = features_df[feature].values[0]
    return metafeatures

def get_dataset_metafeatures_path(dataset_filename):
    dataset_name = dataset_filename.rsplit('.', 1)[0]
    return os.path.join(METAFEATURES_DIR, dataset_name+'_mf.json')

def compute_dataset_metafeatures():
    metadata = json.load(open(METADATA_PATH, 'r'))
    for dataset_metadata in metadata:
        dataset_filename = dataset_metadata['filename']
        target_class_name = dataset_metadata['target_class_name']
        index_col_name = dataset_metadata.get('index_col_name', None)

        choice = None
        while not choice in ['y', 'v', 'n']:
            choice = input(dataset_filename + ' [(y)es, (v)erbose, (n)o]: ')

        if choice == 'n':
            continue

        X, Y, column_types = read_dataset(dataset_filename, index_col_name, target_class_name)

        start_time = time.time()
        metafeatures = extract_metafeatures(X=X, Y=Y, seed=CORRECTNESS_SEED)
        run_time = time.time() - start_time

        # metafeatures = {key: value for key, value in metafeatures.items() if not '_Time' in key}

        for key, value in metafeatures.items():
            if 'int' in str(type(value)):
                metafeatures[key] = int(value)
            elif 'float' in str(type(value)):
                metafeatures[key] = float(value)
            else:
                raise Exception('unhandled type: {}'.format(type(value)))

        if choice == 'v':
            print(json.dumps(metafeatures, sort_keys=True, indent=4))
        print('Runtime: ' + str(run_time))

        choice = None
        while not choice in ['y', 'n']:
            choice = input(f'Update {dataset_filename} metafeatures? [(y)es, (n)o]: ')
        if choice == 'y':
            mf_file_path = get_dataset_metafeatures_path(dataset_filename)
            json.dump(metafeatures, open(mf_file_path, 'w'), sort_keys=True, indent=4)
