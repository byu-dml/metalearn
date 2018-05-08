import os
import json
import time

import pandas as pd

from metalearn.metafeatures.metafeatures import Metafeatures

def extract_metafeatures(X, Y, random_seed=0):
    metafeatures = {}
    features_df = Metafeatures().compute(X=X, Y=Y, seed=random_seed)
    for feature in features_df.columns:
        metafeatures[feature] = features_df[feature].as_matrix()[0]
    return metafeatures

directory_path = './data/'
datasets_path = os.path.join(directory_path, 'test_datasets.json')
datasets_file = open(datasets_path, 'r')
datasets = json.load(datasets_file)
datasets_file.close()
for dataset in datasets:
	file_path = dataset['path']
	target_name = dataset['target_name']
	name = file_path[:file_path.rfind('.')]
	print('\nDataset: ' + name)
	choice = input('y - run / n - don\'t run / v - run with output\n')
	if choice == 'y' or choice == 'v':
		dataframe = pd.read_csv(os.path.join(directory_path, file_path))
		if "d3mIndex" in dataframe.columns:
			dataframe.drop(columns="d3mIndex", inplace=True)
		X = dataframe.drop(columns=[target_name], axis=1)
		Y = dataframe[target_name]
		start_time = time.time()
		metafeatures = extract_metafeatures(X=X,Y=Y)
		run_time = time.time() - start_time
		if choice == 'v':
			print(json.dumps(metafeatures, sort_keys=True, indent=4))
		print('Dataset: ' + name)
		print('Runtime: ' + str(run_time))
		choice = input('y - update known metafeatures / n - don\'t \n')
		if choice == 'y':
			mf_file = open(os.path.join(directory_path, name + '_mf.json'), 'w')
			json.dump(metafeatures, mf_file, sort_keys=True, indent=4)
			mf_file.close()
