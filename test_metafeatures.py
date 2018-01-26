import json

import numpy as np
import pandas as pd

from arff2pandas import a2p

from metalearn.metafeatures.simple_metafeatures import SimpleMetafeatures
from metalearn.metafeatures.statistical_metafeatures import StatisticalMetafeatures
from metalearn.metafeatures.information_theoretic_metafeatures import InformationTheoreticMetafeatures
from metalearn.metafeatures.landmarking_metafeatures import LandmarkingMetafeatures


def load_arff(infile_path):
    f = open(infile_path)
    dataframe = a2p.load(f)
    column_name = [name for name in list(dataframe.columns) if 'class@' in name][0]
    dataframe = dataframe.rename(index=str, columns={column_name: 'target'})
    return dataframe

def extract_metafeatures(dataframe):
    metafeatures = {}
    features_df = SimpleMetafeatures().compute(dataframe)
    for feature in features_df.columns:
        metafeatures[feature] = features_df[feature].as_matrix()[0]
    features_df = StatisticalMetafeatures().compute(dataframe)
    for feature in features_df.columns:
        metafeatures[feature] = features_df[feature].as_matrix()[0]
    features_df = InformationTheoreticMetafeatures().compute(dataframe)
    for feature in features_df.columns:
        metafeatures[feature] = features_df[feature].as_matrix()[0]
    features_df = LandmarkingMetafeatures().compute(dataframe)
    for feature in features_df.columns:
        metafeatures[feature] = features_df[feature].as_matrix()[0]
    return metafeatures

def main():
    for filename in ["./data/iris.arff", "./data/38_sick_train_data.csv"]:
    # for filename in ["./data/38_sick_train_data.csv"]:
    # for filename in ["./data/iris.arff"]:
        ext = filename.split(".")[-1]
        if ext == "arff":
            dataframe = load_arff(filename)
        elif ext == "csv":
            dataframe = pd.read_csv(filename)
            dataframe.rename(columns={"Class": "target"}, inplace=True)
        else:
            raise ValueError("file type '{}' not implemented")

        if "d3mIndex" in dataframe.columns:
            dataframe.drop(columns="d3mIndex", inplace=True)

        metafeatures = extract_metafeatures(dataframe)
        print(json.dumps(metafeatures, sort_keys=True, indent=4))
    print("tests finished")

if __name__ == "__main__":
    main()
