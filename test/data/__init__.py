import hashlib
import itertools as it
import json
import os
import random
import string

import pandas as pd
import numpy as np

from metalearn.metafeatures.metafeatures import Metafeatures

from test.config import (
    CORRECTNESS_SEED, DATASET_SEED, DATASET_DIR, METADATA_PATH,
    METAFEATURES_DIR
)


def get_dataset_path(dataset_id):
    return os.path.join(DATASET_DIR, dataset_id+'.csv')

def read_dataset(metadata):
    dataset_id = metadata['id']
    path = get_dataset_path(dataset_id)
    data = pd.read_csv(path)
    y = None
    X = data
    if 'target_column' in metadata:
        target_column = metadata['target_column']
        y = X[target_column]
        X.drop(target_column, axis=1, inplace=True)
    return X, y, metadata['column_types']

class InvalidStateError(Exception):
    pass


class PandasChainedAssignmentContextManager:

    def __init__(self, chained_assignment_mode=None):
        self._local_setting = chained_assignment_mode

    def __enter__(self):
        self._global_setting = pd.options.mode.chained_assignment
        pd.options.mode.chained_assignment = self._local_setting

    def __exit__(self, *args):
        pd.options.mode.chained_assignment = self._global_setting


class DatasetGenerator:

    NUMERIC = 'NUMERIC'
    CATEGORICAL = 'CATEGORICAL'
    TEXT = 'TEXT'
    COLUMN_TYPES = (NUMERIC, CATEGORICAL, TEXT)

    def __init__(self):
        self.datasets = None

    def generate(
        self, n_rows=32, max_n_categories=4, max_text_length=8,
        max_n_cols_per_type=8, sparsity=.125, *, seed=DATASET_SEED
    ):
        self.n_rows = n_rows
        self.max_n_categories = max_n_categories
        self.max_text_length = max_text_length
        self.max_n_cols_per_type = max_n_cols_per_type
        self.sparsity = sparsity
        self.seed = seed

        self.rng = random.Random()
        self.rng.seed(seed)
        self.datasets = []

        with PandasChainedAssignmentContextManager():
            self._generate(1, 1)
            self._generate(n_rows, max_n_cols_per_type)
            self._generate_sparse(n_rows, max_n_cols_per_type)
            self._generate_empty_rows(n_rows, max_n_cols_per_type, 1)
            self._generate_empty_rows(n_rows, max_n_cols_per_type)
            self._generate_empty_cols(n_rows, max_n_cols_per_type, 1)
            self._generate_empty_cols(n_rows, max_n_cols_per_type)

    def save(self, dataset_dir, metadata_path):
        if self.datasets is None:
            raise InvalidStateError('Datasets not generated')

        if not os.path.isdir(dataset_dir):
            os.makedirs(dataset_dir)

        metadatas = []
        for dataset in self.datasets:
            dataset['metadata'].pop('id', None)
            dataset_id = self._hash_id(dataset)
            dataset['metadata']['id'] = dataset_id
            dataset_path = get_dataset_path(dataset_id)
            dataset['data'].to_csv(dataset_path, index=False)
            metadatas.append(dataset['metadata'])

        metadata_dir = os.path.dirname(metadata_path)
        if not os.path.isdir(metadata_dir):
            os.makedirs(metadata_dir)

        with open(os.path.join(metadata_path), 'w') as f:
            json.dump(metadatas, f, sort_keys=True)

    def _generate(self, n_rows, max_n_cols_per_type):
        Xs, X_metadatas = self._generate_Xs(n_rows, max_n_cols_per_type)

        for X, X_metadata in zip(Xs, X_metadatas):
            self._add_dataset(X, X_metadata)
            for y_column_type in self.COLUMN_TYPES:
                data, metadata = self._generate_y(X, X_metadata, y_column_type, n_rows)
                self._add_dataset(data, metadata)

    def _generate_Xs(self, n_rows, max_n_cols_per_type):
        metadatas = []
        Xs = []
        for unique_column_types in self._enumerate_column_type_combinations():
            metadata = self._sample_X_metadata(unique_column_types, max_n_cols_per_type)
            X = self._sample_X(metadata, n_rows)
            metadatas.append(metadata)
            Xs.append(X)
        return Xs, metadatas

    def _enumerate_column_type_combinations(self):
        return it.chain(
            *map(
                lambda r: it.combinations(self.COLUMN_TYPES, r),
                range(1, 1 + len(self.COLUMN_TYPES))
            )
        )

    def _sample_X_metadata(self, unique_column_types, max_n_cols_per_type):
        metadata = {
            'unique_column_types': sorted(unique_column_types),
        }
        column_types = []
        for col_type in unique_column_types:
            column_types += [col_type] * self.rng.randint(1, max_n_cols_per_type)
        self.rng.shuffle(column_types)
        metadata['column_types'] = {
            f'x{i}': col_type for i, col_type in enumerate(column_types)
        }
        return metadata

    def _sample_X(self, X_metadata, n_rows):
        X = {}
        for col_name, col_type in sorted(X_metadata['column_types'].items()):
            X[col_name] = self._sample_column(col_type, n_rows)
        return pd.DataFrame(X)

    def _sample_column(self, column_type, n_rows):
        if column_type == self.NUMERIC:
            return [self._sample_numeric_value() for _ in range(n_rows)]

        elif column_type == self.CATEGORICAL:
            n_categories = self.rng.randint(2, self.max_n_categories)
            return [
                self._sample_categorical_value(n_categories) for _ in range(n_rows)
            ]

        elif column_type == self.TEXT:
            return [self._sample_text_value() for _ in range(n_rows)]

        elif column_type is None:
            return [None] * n

        else:
            self._unknown_column_type(column_type)

    def _unknown_column_type(self, column_type):
        raise ValueError('Unknown column type: {}'.format(column_type))

    def _sample_numeric_value(self):
        return 2 * self.rng.random() - 1

    def _sample_categorical_value(self, n_categories):
        return str(self.rng.randint(1, n_categories))

    def _sample_text_value(self):
        text_length = self.rng.randint(0, self.max_text_length)
        return ''.join(
            self.rng.choice(string.ascii_letters) for _ in range(text_length)
        )

    def _add_dataset(self, data, metadata):
        self.datasets.append({
            'data': data, 'metadata': metadata
        })

    def _generate_y(self, X, X_metadata, y_column_type, n_rows):
        data = X.copy(deep=True)
        metadata = {
            'unique_column_types': sorted(list(set(
                X_metadata['unique_column_types'] + [y_column_type]
            ))),
            'column_types': {
                key: value for key, value in sorted(X_metadata['column_types'].items())
            }
        }
        metadata['column_types']['y'] = y_column_type
        metadata['target_column'] = 'y'

        y = pd.Series(np.zeros(n_rows), name='y')
        for col_name in sorted(data.columns):
            x = data[col_name]
            x_column_type = metadata['column_types'][col_name]

            if x_column_type == self.NUMERIC:
                y += self._sample_numeric_value() * x

            elif x_column_type == self.CATEGORICAL:
                x_dummies = pd.get_dummies(x)
                for dummy_col_name in sorted(x_dummies.columns):
                    dummy_x = x_dummies[dummy_col_name]
                    y += self._sample_numeric_value() * dummy_x

            elif x_column_type == self.TEXT:
                x_lens = x.str.len()
                y += (self._sample_numeric_value() / x_lens.max()) * x_lens

            else:
                self._unknown_column_type(column_type)

        if y_column_type == self.CATEGORICAL:
            y = y.round().abs().astype(int)
        elif y_column_type == self.TEXT:
            y = (10*y).round().astype(int) % len(string.ascii_letters)
            for i, value in enumerate(y):
                y[i] = string.ascii_letters[value]

        data = pd.concat([data, y], axis=1)

        return data, metadata

    def _generate_sparse(self, n_rows, max_n_cols_per_type):
        X_metadata = self._sample_X_metadata(self.COLUMN_TYPES, max_n_cols_per_type)
        X = self._sample_X(X_metadata, n_rows)
        data, metadata = self._generate_y(X, X_metadata, self.CATEGORICAL, n_rows)

        for col_name in sorted(X.columns):
            for i in range(n_rows):
                if self.rng.random() < self.sparsity:
                    data[col_name][i] = None

        self._add_dataset(data, metadata)

    def _generate_empty_rows(self, n_rows, max_n_cols_per_type, n_empty_rows=None):
        X_metadata = self._sample_X_metadata(self.COLUMN_TYPES, max_n_cols_per_type)
        X = self._sample_X(X_metadata, n_rows)
        data, metadata = self._generate_y(X, X_metadata, self.CATEGORICAL, n_rows)

        if n_empty_rows is None:
            n_empty_rows = max(2, int(round((self.sparsity * n_rows))))

        empty_row_indices = self.rng.choices(range(n_rows), k=n_empty_rows)
        for col_name in sorted(X.columns):
            for i in empty_row_indices:
                data[col_name][i] = None

        self._add_dataset(data, metadata)

    def _generate_empty_cols(self, n_rows, max_n_cols_per_type, n_empty_cols=None):
        X_metadata = self._sample_X_metadata(self.COLUMN_TYPES, max_n_cols_per_type)
        X = self._sample_X(X_metadata, n_rows)
        data, metadata = self._generate_y(X, X_metadata, self.NUMERIC, n_rows)

        if n_empty_cols is None:
            n_empty_cols = max(2, int(round((self.sparsity * len(X.columns)))))

        empty_col_names = self.rng.choices(X.columns, k=n_empty_cols)
        for empty_col_name in empty_col_names:
            data[empty_col_name] = [None] * n_rows

        self._add_dataset(data, metadata)

    @staticmethod
    def _hash_id(dataset):
        return hashlib.sha256(DatasetGenerator._dataset_to_json(dataset).encode('utf8')).hexdigest()

    @staticmethod
    def _dataset_to_json(dataset):
        json_structure = {
            'data': dataset['data'].to_json(),
            'metadata': dataset['metadata']
        }
        return json.dumps(json_structure, sort_keys=True)


def get_dataset_metafeatures_path(dataset_id):
    return os.path.join(METAFEATURES_DIR, dataset_id+"_mf.json")

def compute_metafeatures(X, y, column_types, dataset_id):
    try:
        metafeatures = Metafeatures().compute(X=X, Y=y, column_types=column_types, seed=CORRECTNESS_SEED)
        for mf_name, mf_info in metafeatures.items():
            if 'compute_time' in mf_info:
                del mf_info['compute_time']

        mf_file_path = get_dataset_metafeatures_path(dataset_id)
        if not os.path.isdir(METAFEATURES_DIR):
            os.makedirs(METAFEATURES_DIR)
        json.dump(metafeatures, open(mf_file_path, "w"), sort_keys=True)
    except ValueError: # todo fix value error
        pass


def initialize():
    dg = DatasetGenerator()
    dg.generate()
    dg.save(DATASET_DIR, METADATA_PATH)
    for dataset in dg.datasets:
        X, y, column_types = read_dataset(dataset['metadata'])
        dataset_id = dataset['metadata']['id']
        compute_metafeatures(X, y, column_types, dataset_id)
