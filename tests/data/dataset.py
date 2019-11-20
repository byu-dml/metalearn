import os

from arff2pandas import a2p
import pandas as pd


DATASET_DIR = './tests/data/datasets/'


def get_dataset_path(dataset_filename):
    return os.path.join(DATASET_DIR, dataset_filename)

def _read_arff_dataset(infile_path):
    """ Loads and ARFF file to a pandas dataframe and drops meta-info on column type. """
    with open(infile_path) as fh:
        df = a2p.load(fh)
    # Default column names follow ARFF, e.g. petalwidth@REAL, class@{a,b,c}
    df.columns = [col.split('@')[0] for col in df.columns]
    return df

def _read_csv_dataset(path, index_col_name):
    return pd.read_csv(path, index_col=index_col_name)

def read_dataset(dataset_metadata):
    """ Loads a csv or arff file (provided they are named *.{csv|arff}) """
    dataset_filename = dataset_metadata["filename"]
    target_class_name = dataset_metadata["target_class_name"]
    index_col_name = dataset_metadata.get("index_col_name", None)
    column_types = dataset_metadata.get("column_types", None)
    dataset_path = get_dataset_path(dataset_filename)
    ext = dataset_path.split(".")[-1]
    if ext == "arff":
        dataframe = _read_arff_dataset(dataset_path)
    elif ext == "csv":
        dataframe = _read_csv_dataset(dataset_path, index_col_name)
    else:
        raise ValueError("load file type '{}' not implemented".format(ext))

    X = dataframe.drop(columns=[target_class_name], axis=1)
    Y = dataframe[target_class_name]
    return X, Y, column_types
