import json

import numpy as np

import openml
import pandas as pd

from arff2pandas import a2p


from metalearn.metafeatures.simple_metafeatures import SimpleMetafeatures
from metalearn.metafeatures.statistical_metafeatures import StatisticalMetafeatures
from metalearn.metafeatures.information_theoretic_metafeatures import InformationTheoreticMetafeatures
from metalearn.metafeatures.landmarking_metafeatures import LandmarkingMetafeatures


def import_openml_dataset(id=1):
    # get a dataset from openml using a dataset id
    dataset = openml.datasets.get_dataset(id)
    # get the metafeatures from the dataset
    omlMetafeatures = dataset.qualities
    # get X, Y, and attributes from the dataset
    X, Y, attributes = dataset.get_data(target=dataset.default_target_attribute, return_attribute_names=True)

    # format attributes
    for i in range(len(X[0])):
        attributes[i] = (attributes[i], str(type(X[0][i])))

    return X, Y, attributes, omlMetafeatures

def compare_with_openml(X,Y,attributes,omlMetafeatures):
    # get metafeatures from dataset using our metafeatures

    print("\nopenml Metafeatures: \n",omlMetafeatures,"\n\n")
    metafeatures = extract_metafeatures(X,Y,attributes)
    print("\nOur Metafeatures: \n",metafeatures,"\n\n")


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
    # print(compute_metafeatures("./iris.arff"))
    print("\n\n\n")
    X,Y,attributes,omlMetafeatures = import_openml_dataset()
    compare_with_openml(X,Y,attributes,omlMetafeatures)
    main()

