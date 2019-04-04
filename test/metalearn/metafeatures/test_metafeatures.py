""" Contains unit tests for the Metafeatures class. """
import inspect
import json
import jsonschema
import math
import os
import random
import time
import copy
import unittest

import pandas as pd
import numpy as np

from metalearn import Metafeatures
from test.config import CORRECTNESS_SEED, METADATA_PATH
from test.data.dataset import read_dataset
from test.data.compute_dataset_metafeatures import get_dataset_metafeatures_path

FAIL_MESSAGE = "message"
FAIL_REPORT = "report"
TEST_NAME = "test_name"


class MetafeaturesWithDataTestCase(unittest.TestCase):
    """ Contains tests for Metafeatures that require loading data first. """

    def setUp(self):
        self.datasets = {}
        with open(METADATA_PATH, "r") as fh:
            dataset_descriptions = json.load(fh)
        for dataset_description in dataset_descriptions:
            X, Y, column_types = read_dataset(dataset_description)
            filename = dataset_description["filename"]
            known_dataset_metafeatures_path = get_dataset_metafeatures_path(
                filename
            )
            if os.path.exists(known_dataset_metafeatures_path):
                with open(known_dataset_metafeatures_path) as fh:
                    metafeatures = json.load(fh)
                self.datasets[filename] = {
                    "X": X, "Y": Y, "column_types": column_types,
                    "known_metafeatures": metafeatures,
                    "known_metafeatures_path": known_dataset_metafeatures_path
                }
            else:
                raise FileNotFoundError(f"{known_dataset_metafeatures_path} does not exist")

    def tearDown(self):
        del self.datasets

    def _report_test_failures(self, test_failures, test_name):
        if test_failures != {}:
            report_path = f"./failures_{test_name}.json"
            with open(report_path, "w") as fh:
                json.dump(test_failures, fh, indent=4)
            message = next(iter(test_failures.values()))[FAIL_MESSAGE]
            self.fail(
                f"{message} Details have been written in {report_path}."
            )

    def _check_correctness(self, computed_mfs, known_mfs, filename):
        """
        Tests whether computed_mfs are close to previously computed metafeature
        values. This assumes that the previously computed values are correct
        and allows testing for changes in metafeature computation. Only checks
        the correctness of the metafeatures passed in--does not test that all
        computable metafeatures were computed.
        """
        test_failures = {}
        fail_message = "Not all metafeatures matched previous results."

        for mf_id, result in computed_mfs.items():
            computed_value = result[Metafeatures.VALUE_KEY]
            known_value = known_mfs[mf_id][Metafeatures.VALUE_KEY]
            correct = True
            if known_value is None:
                correct = False
            elif type(known_value) is str:
                correct = known_value == computed_value
            else:
                correct = np.array(np.isclose(known_value, computed_value, equal_nan=True)).all()
            if not correct:
                test_failures[mf_id] = {
                    "known_value": known_value,
                    "computed_value": computed_value
                }

        return self._format_check_report(
            "correctness", fail_message, test_failures, filename
        )

    def _format_check_report(
        self, test_name, fail_message, test_failures, filename
    ):
        if test_failures == {}:
            return test_failures
        else:
            return {
                filename: {
                    TEST_NAME: test_name,
                    FAIL_MESSAGE: fail_message,
                    FAIL_REPORT: test_failures
                }
            }

    def _check_compare_metafeature_lists(self, computed_mfs, known_mfs, filename):
        """
        Tests whether computed_mfs matches the list of previously computed metafeature
        names as well as the list of computable metafeatures in Metafeatures.list_metafeatures
        """
        test_failures = {}
        fail_message = "Metafeature lists do not match."

        with open("./metalearn/metafeatures/metafeatures.json") as f:
            master_mf_ids = json.load(f)["metafeatures"].keys()
        master_mf_ids_set = set(master_mf_ids)

        known_mf_ids_set = set({
            x for x in known_mfs.keys() if "_Time" not in x
        })
        computed_mf_ids_set = set(computed_mfs.keys())

        intersect_mf_ids_set = master_mf_ids_set.intersection(known_mf_ids_set
            ).intersection(computed_mf_ids_set)

        master_diffs = master_mf_ids_set - intersect_mf_ids_set
        if len(master_diffs) > 0:
            test_failures["master_differences"] = list(master_diffs)
        known_diffs = known_mf_ids_set - intersect_mf_ids_set
        if len(known_diffs) > 0:
            test_failures["known_differences"] = list(known_diffs)
        computed_diffs = computed_mf_ids_set - intersect_mf_ids_set
        if len(computed_diffs) > 0:
            test_failures["computed_differences"] = list(computed_diffs)

        return self._format_check_report(
            "metafeature_lists", fail_message, test_failures, filename
        )

    def _perform_checks(self, functions):
        check = {}
        for function, args in functions:
            check = function(*args)
            if check != {}:
                break
        return check

    def test_run_without_exception(self):
        try:
            for dataset_filename, dataset in self.datasets.items():
                Metafeatures().compute(
                    X=dataset["X"], Y=dataset["Y"],
                    column_types=dataset["column_types"]
                )
        except Exception as e:
            exc_type = type(e).__name__
            self.fail(f"computing metafeatures raised {exc_type} unexpectedly")

    def test_correctness(self):
        """Tests that metafeatures are computed correctly, for known datasets.
        """
        test_failures = {}
        test_name = inspect.stack()[0][3]
        for dataset_filename, dataset in self.datasets.items():
            computed_mfs = Metafeatures().compute(
                X=dataset["X"], Y=dataset["Y"], seed=CORRECTNESS_SEED,
                column_types=dataset["column_types"]
            )
            known_mfs = dataset["known_metafeatures"]
            required_checks = [
                (self._check_correctness,
                 [computed_mfs, known_mfs, dataset_filename]),
                (self._check_compare_metafeature_lists,
                 [computed_mfs, known_mfs, dataset_filename])
            ]
            test_failures.update(self._perform_checks(required_checks))

        self._report_test_failures(test_failures, test_name)

    def test_individual_metafeature_correctness(self):
        test_failures = {}
        test_name = inspect.stack()[0][3]
        for dataset_filename, dataset in self.datasets.items():
            known_mfs = dataset["known_metafeatures"]
            for mf_id in Metafeatures.IDS:
                computed_mfs = Metafeatures().compute(
                    X=dataset["X"], Y=dataset["Y"], seed=CORRECTNESS_SEED,
                    metafeature_ids=[mf_id],
                    column_types=dataset["column_types"]
                )
                required_checks = [
                    (self._check_correctness,
                     [computed_mfs, known_mfs, dataset_filename])
                ]
                test_failures.update(self._perform_checks(required_checks))

        self._report_test_failures(test_failures, test_name)

    def test_no_targets(self):
        """ Test Metafeatures().compute() without targets
        """
        test_failures = {}
        test_name = inspect.stack()[0][3]
        for dataset_filename, dataset in self.datasets.items():
            metafeatures = Metafeatures()
            computed_mfs = metafeatures.compute(
                X=dataset["X"], Y=None, seed=CORRECTNESS_SEED,
                column_types=dataset["column_types"]
            )
            known_mfs = dataset["known_metafeatures"]
            target_dependent_metafeatures = Metafeatures.list_metafeatures(
                "target_dependent"
            )
            for mf_name in target_dependent_metafeatures:
                known_mfs[mf_name] = {
                    Metafeatures.VALUE_KEY: Metafeatures.NO_TARGETS,
                    Metafeatures.COMPUTE_TIME_KEY: 0.
                }

            required_checks = [
                (self._check_correctness,
                 [computed_mfs, known_mfs, dataset_filename]),
                (self._check_compare_metafeature_lists,
                 [computed_mfs, known_mfs, dataset_filename])
            ]
            test_failures.update(self._perform_checks(required_checks))

        self._report_test_failures(test_failures, test_name)

    def test_numeric_targets(self):
        """ Test Metafeatures().compute() with numeric targets
        """
        test_failures = {}
        test_name = inspect.stack()[0][3]
        for dataset_filename, dataset in self.datasets.items():
            metafeatures = Metafeatures()
            column_types = dataset["column_types"].copy()
            column_types[dataset["Y"].name] = metafeatures.NUMERIC
            computed_mfs = metafeatures.compute(
                X=dataset["X"], Y=pd.Series(np.random.rand(dataset["Y"].shape[0]),
                name=dataset["Y"].name), seed=CORRECTNESS_SEED, 
                column_types=column_types
            )
            known_mfs = dataset["known_metafeatures"]
            target_dependent_metafeatures = Metafeatures.list_metafeatures(
                "target_dependent"
            )
            for mf_name in target_dependent_metafeatures:
                known_mfs[mf_name] = {
                    Metafeatures.VALUE_KEY: Metafeatures.NUMERIC_TARGETS,
                    Metafeatures.COMPUTE_TIME_KEY: 0.
                }

            required_checks = [
                (self._check_correctness,
                 [computed_mfs, known_mfs, dataset_filename]),
                (self._check_compare_metafeature_lists,
                 [computed_mfs, known_mfs, dataset_filename])
            ]
            test_failures.update(self._perform_checks(required_checks))

        self._report_test_failures(test_failures, test_name)

    def test_request_metafeatures(self):
        SUBSET_LENGTH = 20
        test_failures = {}
        test_name = inspect.stack()[0][3]
        for dataset_filename, dataset in self.datasets.items():
            metafeature_ids = random.sample(Metafeatures.IDS, SUBSET_LENGTH)
            computed_mfs = Metafeatures().compute(
                X=dataset["X"],Y=dataset["Y"], seed=CORRECTNESS_SEED,
                metafeature_ids=metafeature_ids,
                column_types=dataset["column_types"]
            )
            known_metafeatures = dataset["known_metafeatures"]
            required_checks = [
                (self._check_correctness,
                 [computed_mfs, known_metafeatures, dataset_filename])
            ]

            test_failures.update(self._perform_checks(required_checks))
            self.assertEqual(
                metafeature_ids, list(computed_mfs.keys()),
                "Compute did not return requested metafeatures"
            )
        self._report_test_failures(test_failures, test_name)

    def test_exclude_metafeatures(self):
        SUBSET_LENGTH = 20
        test_failures = {}
        test_name = inspect.stack()[0][3]
        for dataset_filename, dataset in self.datasets.items():
            metafeature_ids = random.sample(Metafeatures.IDS, SUBSET_LENGTH)
            computed_mfs = Metafeatures().compute(
                X=dataset["X"], Y=dataset["Y"], seed=CORRECTNESS_SEED,
                exclude=metafeature_ids,
                column_types=dataset["column_types"]
            )
            known_metafeatures = dataset["known_metafeatures"]
            required_checks = [
                (self._check_correctness,
                 [computed_mfs, known_metafeatures, dataset_filename])
            ]

            test_failures.update(self._perform_checks(required_checks))
            if any(mf_id in computed_mfs.keys() for mf_id in metafeature_ids):
                self.assertTrue(False, "Metafeatures computed an excluded metafeature")

        self._report_test_failures(test_failures, test_name)

    def test_compute_effects_on_dataset(self):
        """
        Tests whether computing metafeatures has any side effects on the input
        X or Y data. Fails if there are any side effects.
        """
        for dataset in self.datasets.values():
            X_copy, Y_copy = dataset["X"].copy(), dataset["Y"].copy()
            Metafeatures().compute(
                X=dataset["X"],Y=dataset["Y"],
                column_types=dataset["column_types"]
            )
            if not (
                X_copy.equals(dataset["X"]) and Y_copy.equals(dataset["Y"])
            ):
                self.assertTrue(
                    False, "Input data has changed after Metafeatures.compute"
                )

    def test_compute_effects_on_compute(self):
        """
        Tests whether computing metafeatures has any side effects on the
        instance metafeatures object. Fails if there are any side effects.
        """
        required_checks = []
        test_failures = {}
        test_name = inspect.stack()[0][3]
        for dataset_filename, dataset in self.datasets.items():
            metafeatures_instance = Metafeatures()
            # first run
            metafeatures_instance.compute(
                X=dataset["X"],Y=dataset["Y"],seed=CORRECTNESS_SEED,
                column_types=dataset["column_types"]
            )
            # second run
            computed_mfs = metafeatures_instance.compute(
                X=dataset["X"],Y=dataset["Y"],seed=CORRECTNESS_SEED,
                column_types=dataset["column_types"]
            )

            known_mfs = dataset["known_metafeatures"]
            required_checks.append(
                (self._check_correctness,
                 [computed_mfs, known_mfs, dataset_filename])
            )
            test_failures.update(self._perform_checks(required_checks))
        self._report_test_failures(test_failures, test_name)

    def test_output_format(self):
        with open("./metalearn/metafeatures/metafeatures_schema.json") as f:
            mf_schema = json.load(f)
        for dataset_filename, dataset in self.datasets.items():
            computed_mfs = Metafeatures().compute(
                X=dataset["X"],Y=dataset["Y"],
                column_types=dataset["column_types"]
            )
            try:
                jsonschema.validate(computed_mfs, mf_schema)
            except jsonschema.exceptions.ValidationError as e:
                self.fail(
                    f"Metafeatures computed from {dataset_filename} do not "+
                    "conform to schema"
                )

    def test_output_json_compatibility(self):
        with open("./metalearn/metafeatures/metafeatures_schema.json") as f:
            mf_schema = json.load(f)
        for dataset_filename, dataset in self.datasets.items():
            computed_mfs = Metafeatures().compute(
                X=dataset["X"],Y=dataset["Y"],
                column_types=dataset["column_types"]
            )
            try:
                json_computed_mfs = json.dumps(computed_mfs)
            except Exception as e:
                self.fail(
                    f"Failed to convert metafeature output to json: {str(e)}"
                )

    def test_soft_timeout(self):
        """Tests Metafeatures().compute() with timeout set"""   
        test_name = inspect.stack()[0][3]   
        test_failures = {} 
        for dataset_filename, dataset in self.datasets.items():
            metafeatures = Metafeatures()

            start_time = time.time()
            metafeatures.compute(
                X=dataset["X"], Y=dataset["Y"], seed=CORRECTNESS_SEED,
                column_types=dataset["column_types"]
            )
            full_compute_time = time.time() - start_time

            start_time = time.time()
            computed_mfs = metafeatures.compute(
                X=dataset["X"], Y=dataset["Y"], seed=CORRECTNESS_SEED,
                column_types=dataset["column_types"], timeout=full_compute_time/2
            )
            limited_compute_time = time.time() - start_time

            self.assertGreater(
                full_compute_time, limited_compute_time,
                f"Compute metafeatures exceeded timeout on '{dataset_filename}'"
            )
            computed_mfs_timeout = {k: v for k, v in computed_mfs.items()
                                    if v[Metafeatures.VALUE_KEY] != Metafeatures.TIMEOUT}
            known_mfs = dataset["known_metafeatures"]
            required_checks = [
                (self._check_correctness,
                 [computed_mfs_timeout, known_mfs, dataset_filename]),
                (self._check_compare_metafeature_lists,
                 [computed_mfs, known_mfs, dataset_filename])
            ]

        test_failures.update(self._perform_checks(required_checks))
        self._report_test_failures(test_failures, test_name)


class MetafeaturesTestCase(unittest.TestCase):
    """ Contains tests for Metafeatures that can be executed without loading data. """

    def setUp(self):
        self.dummy_features = pd.DataFrame(np.random.rand(50, 50))
        self.dummy_target = pd.Series(np.random.randint(2, size=50), name="target").astype("str")

        self.invalid_requested_metafeature_message_start = "One or more requested metafeatures are not valid:"
        self.invalid_excluded_metafeature_message_start = "One or more excluded metafeatures are not valid:"
        self.invalid_metafeature_message_start_fail_message = "Error message indicating invalid metafeatures did not start with expected string."
        self.invalid_metafeature_message_contains_fail_message = "Error message indicating invalid metafeatures should include names of invalid features."

    def test_dataframe_input_error(self):
        """ Tests if `compute` gives a user-friendly error when a TypeError or ValueError occurs. """

        expected_error_message1 = "X must be of type pandas.DataFrame"
        fail_message1 = "We expect a user friendly message when the features passed to compute is not a Pandas.DataFrame."
        expected_error_message2 = "X must not be empty"
        fail_message2 = "We expect a user friendly message when the features passed to compute are empty."
        expected_error_message3 = "Y must be of type pandas.Series"
        fail_message3 = "We expect a user friendly message when the target column passed to compute is not a Pandas.Series."
        expected_error_message4 = "Y must have the same number of rows as X"
        fail_message4 = "We expect a user friendly message when the target column passed to compute has a number of rows different than X's."
        # We don't check for the Type of TypeError explicitly as any other error would fail the unit test.

        with self.assertRaises(TypeError) as cm:
            Metafeatures().compute(X=None, Y=self.dummy_target)
        self.assertEqual(str(cm.exception), expected_error_message1, fail_message1)

        with self.assertRaises(TypeError) as cm:
            Metafeatures().compute(X=np.zeros((500, 50)), Y=pd.Series(np.zeros(500)))
        self.assertEqual(str(cm.exception), expected_error_message1, fail_message1)

        with self.assertRaises(ValueError) as cm:
            Metafeatures().compute(X=pd.DataFrame(np.zeros((0, 50))), Y=pd.Series(np.zeros(500)))
        self.assertEqual(str(cm.exception), expected_error_message2, fail_message2)

        with self.assertRaises(ValueError) as cm:
            Metafeatures().compute(X=pd.DataFrame(np.zeros((500, 0))), Y=pd.Series(np.zeros(500)))
        self.assertEqual(str(cm.exception), expected_error_message2, fail_message2)

        with self.assertRaises(TypeError) as cm:
            Metafeatures().compute(X=pd.DataFrame(np.zeros((500, 50))), Y=np.random.randint(2, size=500).astype("str"))
        self.assertEqual(str(cm.exception), expected_error_message3, fail_message3)

        with self.assertRaises(ValueError) as cm:
            Metafeatures().compute(X=pd.DataFrame(np.zeros((500, 50))), Y=pd.Series(np.random.randint(2, size=0), name="target").astype("str"))
        self.assertEqual(str(cm.exception), expected_error_message4, fail_message4)

    def _check_invalid_metafeature_exception_string(self, exception_str, expected_str, invalid_metafeatures):
        """ Checks if the exception message starts with the right string, and contains all of the invalid metafeatures expected. """
        self.assertTrue(
            exception_str.startswith(expected_str),
            self.invalid_metafeature_message_start_fail_message
        )

        for invalid_mf in invalid_metafeatures:
            self.assertTrue(
                invalid_mf in exception_str,
                self.invalid_metafeature_message_contains_fail_message
            )

    def test_metafeatures_input_all_invalid(self):
        """ Test cases where all requested and excluded metafeatures are invalid. """

        invalid_metafeatures = ["ThisIsNotValid", "ThisIsAlsoNotValid"]

        with self.assertRaises(ValueError) as cm:
            Metafeatures().compute(X=self.dummy_features, Y=self.dummy_target, metafeature_ids=invalid_metafeatures)
        self._check_invalid_metafeature_exception_string(str(cm.exception),
                                                         self.invalid_requested_metafeature_message_start,
                                                         invalid_metafeatures)

        with self.assertRaises(ValueError) as cm:
            Metafeatures().compute(X=self.dummy_features, Y=self.dummy_target, exclude=invalid_metafeatures)
        self._check_invalid_metafeature_exception_string(str(cm.exception),
                                                         self.invalid_excluded_metafeature_message_start,
                                                         invalid_metafeatures)

    def test_metafeatures_input_partial_invalid(self):
        """ Test case where only some requested and excluded metafeatures are invalid. """

        invalid_metafeatures = ["ThisIsNotValid", "ThisIsAlsoNotValid"]
        valid_metafeatures = ["NumberOfInstances", "NumberOfFeatures"]

        with self.assertRaises(ValueError) as cm:
            Metafeatures().compute(X=self.dummy_features, Y=self.dummy_target,
                                   metafeature_ids=invalid_metafeatures + valid_metafeatures)
        self._check_invalid_metafeature_exception_string(str(cm.exception),
                                                         self.invalid_requested_metafeature_message_start,
                                                         invalid_metafeatures)

        with self.assertRaises(ValueError) as cm:
            Metafeatures().compute(X=self.dummy_features, Y=self.dummy_target,
                                   exclude=invalid_metafeatures + valid_metafeatures)
        self._check_invalid_metafeature_exception_string(str(cm.exception),
                                                         self.invalid_excluded_metafeature_message_start,
                                                         invalid_metafeatures)

        # Order should not matter
        with self.assertRaises(ValueError) as cm:
            Metafeatures().compute(X=self.dummy_features, Y=self.dummy_target,
                                   metafeature_ids=valid_metafeatures + invalid_metafeatures)
        self._check_invalid_metafeature_exception_string(str(cm.exception),
                                                         self.invalid_requested_metafeature_message_start,
                                                         invalid_metafeatures)

        with self.assertRaises(ValueError) as cm:
            Metafeatures().compute(X=self.dummy_features, Y=self.dummy_target,
                                   exclude=valid_metafeatures + invalid_metafeatures)
        self._check_invalid_metafeature_exception_string(str(cm.exception),
                                                         self.invalid_excluded_metafeature_message_start,
                                                         invalid_metafeatures)

    def test_request_and_exclude_metafeatures(self):
        expected_exception_string = "metafeature_ids and exclude cannot both be non-null"

        with self.assertRaises(ValueError) as cm:
            Metafeatures().compute(X=self.dummy_features, Y=self.dummy_target,
                                   metafeature_ids=[], exclude=[])

        self.assertEqual(str(cm.exception), expected_exception_string)

    def test_column_type_input(self):
        column_types = {col: "NUMERIC" for col in self.dummy_features.columns}
        column_types[self.dummy_features.columns[2]] = "CATEGORICAL"
        column_types[self.dummy_target.name] = "CATEGORICAL"
        # all valid
        try:
            Metafeatures().compute(
                self.dummy_features, self.dummy_target, column_types
            )
        except Exception as e:
            exc_type = type(e).__name__
            self.fail(f"computing metafeatures raised {exc_type} unexpectedly")
        # some valid
        column_types[self.dummy_features.columns[0]] = "NUMBER"
        column_types[self.dummy_features.columns[1]] = "CATEGORY"
        with self.assertRaises(ValueError) as cm:
            Metafeatures().compute(
                self.dummy_features, self.dummy_target, column_types
            )
        self.assertTrue(
            str(cm.exception).startswith(
                "Invalid column types:"
            ),
            "Some invalid column types test failed"
        )
        # all invalid
        column_types = {feature: "INVALID_TYPE" for feature in self.dummy_features.columns}
        column_types[self.dummy_target.name] = "INVALID"
        with self.assertRaises(ValueError) as cm:
            Metafeatures().compute(
                self.dummy_features, self.dummy_target, column_types
            )
        self.assertTrue(
            str(cm.exception).startswith(
                "Invalid column types:"
            ),
            "All invalid column types test failed"
        )
        # invalid number of column types
        del column_types[self.dummy_features.columns[0]]
        with self.assertRaises(ValueError) as cm:
            Metafeatures().compute(
                self.dummy_features, self.dummy_target, column_types
            )
        self.assertTrue(
            str(cm.exception).startswith(
                "Column type not specified for column"
            ),
            "Invalid number of column types test failed"
        )

    def test_sampling_shape_no_exception(self):
        try:
            Metafeatures().compute(
                self.dummy_features, self.dummy_target, sample_shape=(10,10)
            )
        except Exception as e:
            exc_type = type(e).__name__
            self.fail(f"computing metafeatures raised {exc_type} unexpectedly")

    def test_sampling_shape_correctness(self):
        sample_shape = (7,13)
        metafeatures = Metafeatures()
        dummy_mf_df = metafeatures.compute(
            self.dummy_features, self.dummy_target, sample_shape=sample_shape
        )
        X_sample = metafeatures._resources["XSample"]["value"]
        self.assertEqual(
            X_sample.shape, sample_shape,
            f"Sampling produced incorrect shape {X_sample.shape}; should have" +
            f" been {sample_shape}."
        )

    def test_sampling_shape_invalid_input(self):
        error_tests = [
            {
                "sample_shape": "bad_shape",
                "message": "`sample_shape` must be of type `tuple` or `list`"
            },
            {
                "sample_shape": {0:"bad", 1:"shape"},
                "message": "`sample_shape` must be of type `tuple` or `list`"
            },
            {
                "sample_shape": (2,2,2),
                "message": "`sample_shape` must be of length 2"
            },
            {
                "sample_shape": [1],
                "message": "`sample_shape` must be of length 2"
            },
            {
                "sample_shape": (0,1),
                "message": "Cannot sample less than one row"
            },
            {
                "sample_shape": (1,0),
                "message": "Cannot sample less than 1 column"
            },
            {
                "sample_shape": (3,10),
                # 4 based on self.dummy_target
                "message": "Cannot sample less than 4 rows from Y"
            }
        ]
        for test in error_tests:
            with self.assertRaises(ValueError) as cm:
                Metafeatures().compute(
                    self.dummy_features, self.dummy_target,
                    sample_shape=test["sample_shape"]
                )
            self.assertEqual(
                str(cm.exception),
                test["message"]
            )

    def test_n_folds_invalid_input(self):
        tests = [
            {
                "n_folds": 0,
                "message": "`n_folds` must be >= 2, but was 0"
            },
            {
                "n_folds": 1,
                "message": "`n_folds` must be >= 2, but was 1"
            },
            {
                "n_folds": 2.1,
                "message": "`n_folds` must be an integer, not 2.1"
            },
            {
                "n_folds": "hello",
                "message": "`n_folds` must be an integer, not hello"
            },
            {
                "n_folds": [3],
                "message": "`n_folds` must be an integer, not [3]"
            },
            {
                "n_folds": {5:7},
                "message": "`n_folds` must be an integer, not {5: 7}"
            }
        ]
        for test in tests:
            with self.assertRaises(ValueError) as cm:
                Metafeatures().compute(
                    self.dummy_features, self.dummy_target,
                    n_folds=test["n_folds"]
                )
            self.assertEqual(str(cm.exception), test["message"])

    def test_n_folds_with_small_dataset(self):
        # should raise error with small (few instances) dataset
        # unless not computing landmarking mfs
        X_small = pd.DataFrame(np.random.rand(3, 7))
        Y_small = pd.Series([0,1,0], name="target").astype("str")
        metafeatures = Metafeatures()

        with self.assertRaises(ValueError) as cm:
            metafeatures.compute(X_small, Y_small, n_folds=2)
        self.assertEqual(
            str(cm.exception),
            "The minimum number of instances in each class of Y is n_folds=2." +
            " Class 1 has 1."
        )

    def test_n_folds_with_small_dataset_no_landmarkers(self):
        # should raise error with small (few instances) dataset
        # unless not computing landmarking mfs
        X_small = pd.DataFrame(np.random.rand(3, 7))
        Y_small = pd.Series([0,1,0], name="target").astype("str")
        metafeature_ids = [
            "NumberOfInstances", "NumberOfFeatures", "NumberOfClasses",
            "NumberOfNumericFeatures", "NumberOfCategoricalFeatures"
        ]
        try:
            Metafeatures().compute(
                X_small, Y_small, metafeature_ids=metafeature_ids, n_folds=2
            )
        except Exception as e:
           exc_type = type(e).__name__
           self.fail(f"computing metafeatures raised {exc_type} unexpectedly")

    def test_target_column_with_one_unique_value(self):
        # should not raise an error
        X = pd.DataFrame(np.random.rand(100, 7))
        Y = pd.Series(np.random.randint(0, 1, 100), name="target").astype("str")
        try:
            Metafeatures().compute(X, Y)
        except Exception as e:
           exc_type = type(e).__name__
           self.fail(f"computing metafeatures raised {exc_type} unexpectedly")

    def test_list_metafeatures(self):
        mf_list = Metafeatures.list_metafeatures()
        mf_list_copy = copy.deepcopy(mf_list)
        mf_list.clear()
        if Metafeatures.list_metafeatures() != mf_list_copy:
            mf_list.extend(mf_list_copy)
            self.assertTrue(False, "Metafeature list has been mutated")


def metafeatures_suite():
    test_cases = [MetafeaturesTestCase, MetafeaturesWithDataTestCase]
    return unittest.TestSuite(map(unittest.TestLoader().loadTestsFromTestCase, test_cases))

