""" Contains unit tests for the MetaFeatures class. """
import json
import math
import os
import random
import unittest
import copy
import time
import operator

import openml
import pandas as pd
import numpy as np
from arff2pandas import a2p
import arff

from metalearn.metafeatures.metafeatures import Metafeatures
from test.data.dataset import read_dataset
from test.data.dataset import _read_arff_dataset
from test.data.compute_dataset_metafeatures import get_dataset_metafeatures_path


class MetaFeaturesWithDataTestCase(unittest.TestCase):
    """ Contains tests for MetaFeatures that require loading data first. """

    def setUp(self):
        self.datasets = {}
        self.data_folder = './test/data/'
        with open(self.data_folder + "test_dataset_metadata.json", "r") as fh:
            dataset_descriptions = json.load(fh)

        for dataset_metadata in dataset_descriptions:
            filename = dataset_metadata["filename"]
            target_class_name = dataset_metadata["target_class_name"]
            index_col_name = dataset_metadata.get('index_col_name', None)
            X, Y, column_types = read_dataset(filename, index_col_name, target_class_name)
            self.datasets[filename] = {"X": X, "Y": Y}

    def tearDown(self):
        del self.datasets

    def test_run_without_fail(self):
        for filename, dataset in self.datasets.items():
            metafeatures_df = Metafeatures().compute(X=dataset["X"],Y=dataset["Y"])
            metafeatures_dict = metafeatures_df.to_dict('records')[0]
            # print(json.dumps(metafeatures_dict, sort_keys=True, indent=4))

    def test_correctness(self):
        """ For each dataset that has a corresponding mf (metafeature) file present,
            check differences in columns we do not expect to change.
        """
        random_seed = 0

        fails = {}
        for filename, dataset in self.datasets.items():
            known_dataset_metafeatures_path = get_dataset_metafeatures_path(filename)
            if os.path.exists(known_dataset_metafeatures_path):
                with open(known_dataset_metafeatures_path) as fh:
                    known_mfs = json.load(fh)

                # Explicitly create empty dict because this provides information about successful tests.
                fails[known_dataset_metafeatures_path] = {}

                metafeatures_df = Metafeatures().compute(X=dataset["X"],Y=dataset["Y"],seed=random_seed)
                computed_mfs = metafeatures_df.to_dict('records')[0]
                for key, value in computed_mfs.items():
                    if 'int' in str(type(value)):
                        computed_mfs[key] = int(value)
                    elif 'float' in str(type(value)):
                        computed_mfs[key] = float(value)
                    else:
                        raise Exception('unhandled type: {}'.format(type(value)))

                for mf, computed_value in computed_mfs.items():
                    if '_Time' in mf:
                        # Timing metafeatures will always differ anyway.
                        # For now we pay no mind, no matter how big a difference may be.
                        continue

                    known_value = known_mfs.get(mf)
                    if not math.isclose(known_value, computed_value) and not (np.isnan(known_value) and np.isnan(computed_value)):
                        fails[known_dataset_metafeatures_path][mf] = (known_value, computed_value)

        self.assertGreater(len(fails), 0, "No known results could be loaded, correctness could not be verified.")
        if not all(f == {} for f in fails.values()):
            # Results are no longer correct. Because multiple results that can be wrong are calculated at once,
            # we want to output all of the wrong results so it might be easier to find out what went wrong.
            fails = {k:v for (k,v) in fails.items() if v != {}}
            fail_report_file = './test/metalearn/metafeatures/correctness_fails.json'
            with open(fail_report_file,'w') as fh:
                json.dump(fails, fh, indent=4)

            self.assertTrue(False, "Not all metafeatures matched previous results, output written to {}.".format(fail_report_file))

    # @unittest.expectedFailure
    def test_compare_openml(self):

        def import_openml_datasets():

            # get a list of datasets from openml
            datasets_dict = openml.datasets.list_datasets()
            datasets = list([k for k,v in datasets_dict.items() if v["NumberOfInstances"] <= 50000 and
                             v["NumberOfFeatures"] <= 200])
            # get a list of filtered dataset ids
            # rand_dataset_ids = datasets
            rand_dataset_ids = datasets
            # rand_dataset_ids = [564]

            # get X, Y, and metafeatures from the datasets
            inconsistencies = False
            runs = 0
            sample_size = 3
            while runs < sample_size:
                try:
                    # dataset_id = np.random.choice(datasets, replace = False)
                    dataset_id = 471
                    dataset = openml.datasets.get_dataset(dataset_id)
                    target = str(dataset.default_target_attribute).split(",")
                    df = _read_arff_dataset(dataset.data_file)
                    if len(target) <= 1:
                        if target[0] == "None":
                            X = df
                            Y = None
                        else:
                            X = df.drop(columns=target, axis=1)
                            Y = df[target].squeeze()
                        dataset_metafeatures = {x: (float(v) if v is not None else v) for x,v in dataset.qualities.items()}
                        dataset = {"X": X, "Y": Y, "metafeatures": dataset_metafeatures}
                        print(dataset_id)
                        if compare_with_openml(dataset, dataset_id):
                            inconsistencies = True
                        runs = runs + 1
                        print("Runs: " + str(runs) + "\tid: " + str(dataset_id))
                except arff.BadNominalValue:
                    continue
                except TypeError as t:
                    print(t)
                except ValueError as v:
                    print(v)
                    continue
                except IndexError as i:
                    print(i)
            self.assertFalse(inconsistencies, "Not all metafeatures matched results from OpenML.")

        def compare_with_openml(oml_dataset, dataset_id):
            # get metafeatures from dataset using our metafeatures
            ourMetafeatures = Metafeatures().compute(X=oml_dataset["X"], Y=oml_dataset["Y"])
            ourMetafeatures = ourMetafeatures.to_dict(orient="records")[0]

            mfNameMap = json.load(open("test/metalearn/metafeatures/oml_metafeature_map.json", "r"))

            omlExclusiveMf = {x: v for x,v in oml_dataset["metafeatures"].items()}
            ourExclusiveMf = {}
            consistentSharedMf = []
            inconsistentSharedMf = []

            for metafeatureName, metafeatureValue in ourMetafeatures.items():
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
                        print(metafeatureName)
                        print(f"Oml value: {omlMetafeatureValue} Our value: {metafeatureValue}")
                        print()
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

            file_path = './test/metalearn/metafeatures/openmlComparisons/'
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            report_name = 'openml_comparison_' + str(dataset_id) + '.json'
            with open(file_path+report_name,'w') as fh:
                json.dump(openmlData, fh, indent=4)

            if len(inconsistentSharedMf) > 0:
                return True
            else:
                return False

        import_openml_datasets()

    def test_compare_metafeature_lists(self):
        inconsistencies = {}
        with open("./metalearn/metafeatures/metafeatures.json") as fh:
            master_list = json.load(fh)
        master_names = set(master_list["metafeatures"].keys())
        for filename, dataset in self.datasets.items():
            known_dataset_metafeatures_path = get_dataset_metafeatures_path(filename)

            if os.path.exists(known_dataset_metafeatures_path):
                with open(known_dataset_metafeatures_path) as fh:
                    known_mfs = json.load(fh)

                inconsistencies[known_dataset_metafeatures_path] = {}

                metafeatures_df = Metafeatures().compute(X=dataset["X"],Y=dataset["Y"])
                computed_mfs = metafeatures_df.to_dict('records')[0]

                known_names_t = set({x for x in known_mfs.keys() if "_Time" in x})
                computed_names_t = set({x for x in computed_mfs.keys() if "_Time" in x})
                intersect_t = known_names_t.intersection(computed_names_t)

                known_names_t_unique = known_names_t - intersect_t
                computed_names_t_unique = computed_names_t - intersect_t

                known_names_no_t = set({x for x in known_mfs.keys() if "_Time" not in x})
                computed_names_no_t = set({x for x in computed_mfs.keys() if "_Time" not in x})
                intersect = master_names.intersection(computed_names_no_t.intersection(known_names_no_t))

                master_names_unique = master_names - intersect
                known_names_unique = (known_names_no_t - intersect).union(known_names_t_unique)
                computed_names_unique = (computed_names_no_t - intersect).union(computed_names_t_unique)

                if len(known_names_unique) > 0:
                    inconsistencies[known_dataset_metafeatures_path]["Known Metafeatures"] = list(known_names_unique)
                if len(computed_names_unique) > 0:
                    inconsistencies[known_dataset_metafeatures_path]["Computed Metafeatures"] = list(computed_names_unique)
                if len(master_names_unique) > 0:
                    inconsistencies[known_dataset_metafeatures_path]["Master List Metafeatures"] = list(master_names_unique)

        self.assertGreater(len(inconsistencies), 0, "No known results could be loaded, metafeature lists could not be compared.")
        if not all(i == {} for i in inconsistencies.values()):
            inconsistencies = {k:v for (k,v) in inconsistencies.items() if v != {}}
            inconsistency_report_file = './test/metalearn/metafeatures/mf_inconsistencies.json'
            with open(inconsistency_report_file, 'w') as fh:
                json.dump(inconsistencies, fh, indent=4)
            self.assertTrue(False, "Metafeature lists do not match, output written to {}.".format(inconsistency_report_file))

    def _is_target_dependent(self, resource_name):
        if resource_name=='Y':
            return True
        elif resource_name=='XSample':
            return False
        else:
            resource_info = self.resource_info_dict[resource_name]
            parameters = resource_info.get('parameters', [])
            for parameter in parameters:
                if self._is_target_dependent(parameter):
                    return True
            function = resource_info['function']
            parameters = self.function_dict[function]['parameters']
            for parameter in parameters:
                if self._is_target_dependent(parameter):
                    return True
            return False

    def _get_target_dependent_metafeatures(self):
        self.resource_info_dict = {}
        metafeatures_list = []
        mf_info_file_path = './metalearn/metafeatures/metafeatures.json'
        with open(mf_info_file_path, 'r') as f:
            mf_info_json = json.load(f)
            self.function_dict = mf_info_json['functions']
            json_metafeatures_dict = mf_info_json['metafeatures']
            json_resources_dict = mf_info_json['resources']
            metafeatures_list = list(json_metafeatures_dict.keys())
            combined_dict = {**json_metafeatures_dict, **json_resources_dict}
            for key in combined_dict:
                self.resource_info_dict[key] = combined_dict[key]
        target_dependent_metafeatures = []
        for mf in metafeatures_list:
            if self._is_target_dependent(mf):
                target_dependent_metafeatures.append(mf)
        return target_dependent_metafeatures

    def test_no_targets(self):
        random_seed = 0
        fails = {}
        inconsistencies = {}
        for filename, dataset in self.datasets.items():
            known_dataset_metafeatures_path = get_dataset_metafeatures_path(filename)
            if os.path.exists(known_dataset_metafeatures_path):
                with open(known_dataset_metafeatures_path) as fh:
                    known_mfs = json.load(fh)

                # Explicitly create empty dicts because this provides information about successful tests.
                fails[known_dataset_metafeatures_path] = {}
                inconsistencies[known_dataset_metafeatures_path] = {}

                metafeatures_df = Metafeatures().compute(X=dataset["X"],Y=None,seed=random_seed)
                computed_mfs = metafeatures_df.to_dict('records')[0]
                self.assertEqual(len(known_mfs), len(computed_mfs), "Computed metafeature list does not match correct metafeature list for no_targets test.")
                
                target_dependent_metafeatures = self._get_target_dependent_metafeatures()
                for mf, computed_value in computed_mfs.items():
                    if '_Time' in mf:
                        # Timing metafeatures will always differ anyway.
                        # For now we pay no mind, no matter how big a difference may be.
                        continue
                    if mf in target_dependent_metafeatures:
                        if not computed_value == 'NO_TARGETS':
                            fails[known_dataset_metafeatures_path][mf] = ('NO_TARGETS', computed_value)
                    else:
                        known_value = known_mfs.get(mf)
                        if not math.isclose(known_value, computed_value) and not (np.isnan(known_value) and np.isnan(computed_value)):
                            fails[known_dataset_metafeatures_path][mf] = (known_value, computed_value)
        self.assertGreater(len(fails), 0, "No known results could be loaded, correctness for no_targets test could not be verified.")
        if not all(f == {} for f in fails.values()):
            # Results are no longer correct. Because multiple results that can be wrong are calculated at once,
            # we want to output all of the wrong results so it might be easier to find out what went wrong.
            fails = {k:v for (k,v) in fails.items() if v != {}}
            fail_report_file = './test/metalearn/metafeatures/no_targets_correctness_fails.json'
            with open(fail_report_file,'w') as fh:
                json.dump(fails, fh, indent=4)
            self.assertTrue(False, "Not all metafeatures matched correct results for no_targets test, output written to {}.".format(fail_report_file))

    # temporarily remove timeout due to broken pipe bug
    def _test_timeout(self):
        '''Tests whether the Metafeatures.compute function returns within the allotted time.'''
        for filename, dataset in self.datasets.items():
            known_mfs = None
            known_dataset_metafeatures_path = get_dataset_metafeatures_path(filename)
            if os.path.exists(known_dataset_metafeatures_path):
                with open(known_dataset_metafeatures_path) as fh:
                    known_mfs = json.load(fh)
            for timeout in [3,5,10]:
                mf = Metafeatures()
                start_time = time.time()
                df = mf.compute(X=dataset["X"], Y=dataset["Y"], timeout=timeout, seed=0)
                compute_time = time.time() - start_time
                if not known_mfs is None:
                    for mf_name, mf_value in df.to_dict('records')[0].items():
                        if not '_Time' in mf_name and mf_value != 'TIMEOUT':
                            self.assertTrue(math.isclose(mf_value, known_mfs[mf_name]), f'Metafeature {mf_name} not computed correctly with timeout enabled')
                self.assertGreater(timeout, compute_time, "computing metafeatures exceeded max time. dataset: '{}', max time: {}, actual time: {}".format(filename, timeout, compute_time))
                self.assertEqual(df.shape[1], 2*len(Metafeatures().list_metafeatures()), "Some metafeatures were not returned...")

class MetaFeaturesTestCase(unittest.TestCase):
    """ Contains tests for MetaFeatures that can be executed without loading data. """

    def setUp(self):
        self.dummy_features = pd.DataFrame(np.random.rand(50,50))
        self.dummy_target = pd.Series(np.random.randint(2, size=50), name="target").astype("str")

        self.invalid_metafeature_message_start = "One or more requested metafeatures are not valid:"
        self.invalid_metafeature_message_start_fail_message = "Error message indicating invalid metafeatures did not start with expected string."
        self.invalid_metafeature_message_contains_fail_message = "Error message indicating invalid metafeatures should include names of invalid features."

    def test_dataframe_input_error(self):
        """ Tests if `compute` gives a user-friendly error when a TypeError occurs. """

        expected_error_message1 = "X must be of type pandas.DataFrame"
        fail_message1 = "We expect a user friendly message when the features passed to compute is not a Pandas.DataFrame."
        expected_error_message2 = "Y must be of type pandas.Series"
        fail_message2 = "We expect a user friendly message when the target column passed to compute is not a Pandas.Series."
        expected_error_message3 = "Regression problems are not supported (target feature is numeric)"
        fail_message3 = "We expect a user friendly message when the DataFrame passed to compute is a regression problem"
        # We don't check for the Type of TypeError explicitly as any other error would fail the unit test.

        with self.assertRaises(TypeError) as cm:
            Metafeatures().compute(X=None, Y=self.dummy_target)
        self.assertEqual(str(cm.exception), expected_error_message1, fail_message1)

        with self.assertRaises(TypeError) as cm:
            Metafeatures().compute(X=np.zeros((500,50)), Y=pd.Series(np.zeros(500)))
        self.assertEqual(str(cm.exception), expected_error_message1, fail_message1)

        with self.assertRaises(TypeError) as cm:
            Metafeatures().compute(X=pd.DataFrame(np.zeros((500,50))), Y=np.zeros(500))
        self.assertEqual(str(cm.exception), expected_error_message2, fail_message2)

        with self.assertRaises(TypeError) as cm:
            Metafeatures().compute(X=self.dummy_features, Y=self.dummy_target.astype("float32"))
        self.assertEqual(str(cm.exception), expected_error_message3, fail_message3)

    def _check_invalid_metafeature_exception_string(self, exception_str, invalid_metafeatures):
        """ Checks if the exception message starts with the right string, and contains all of the invalid metafeatures expected. """
        self.assertTrue(
                exception_str.startswith(self.invalid_metafeature_message_start),
                self.invalid_metafeature_message_start_fail_message
                )

        for invalid_mf in invalid_metafeatures:
            self.assertTrue(
                    invalid_mf in exception_str,
                    self.invalid_metafeature_message_contains_fail_message
                    )

    def test_metafeatures_input_all_invalid(self):
        """ Test case where all requested metafeatures are invalid. """

        invalid_metafeatures = ["ThisIsNotValid", "ThisIsAlsoNotValid"]

        with self.assertRaises(ValueError) as cm:
            Metafeatures().compute(X=self.dummy_features, Y=self.dummy_target, metafeature_ids = invalid_metafeatures)

        self._check_invalid_metafeature_exception_string(str(cm.exception), invalid_metafeatures)

    def test_metafeatures_input_partial_invalid(self):
        """ Test case where only some requested metafeatures are invalid. """

        invalid_metafeatures = ["ThisIsNotValid", "ThisIsAlsoNotValid"]
        valid_metafeatures = ["NumberOfInstances", "NumberOfFeatures"]

        with self.assertRaises(ValueError) as cm:
            Metafeatures().compute(X=self.dummy_features, Y=self.dummy_target, metafeature_ids = invalid_metafeatures+valid_metafeatures)

        self._check_invalid_metafeature_exception_string(str(cm.exception), invalid_metafeatures)

        # Order should not matter
        with self.assertRaises(ValueError) as cm:
            Metafeatures().compute(X = self.dummy_features, Y = self.dummy_target, metafeature_ids = valid_metafeatures+invalid_metafeatures)
        self._check_invalid_metafeature_exception_string(str(cm.exception), invalid_metafeatures)

    def test_column_type_input(self):
        column_types = {feature: "NUMERIC" for feature in self.dummy_features.columns}
        column_types[self.dummy_features.columns[2]] = "CATEGORICAL"
        column_types[self.dummy_target.name] = "CATEGORICAL"
        # all valid
        Metafeatures().compute(
            self.dummy_features, self.dummy_target, column_types
        )
        # some valid
        column_types[self.dummy_features.columns[0]] = "NUMBER"
        column_types[self.dummy_features.columns[1]] = "CATEGORY"
        with self.assertRaises(ValueError) as cm:
            Metafeatures().compute(
                self.dummy_features, self.dummy_target, column_types
            )
            self.assertTrue(
                str(cm.exception).startswith(
                    "One or more input column types are not valid:"
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
                    "One or more input column types are not valid:"
                ),
                "All invalid column types test failed"
            )
        # invalid number of column types
        del column_types[self.dummy_features.columns[0]]
        with self.assertRaises(ValueError) as cm:
            Metafeatures().compute(
                self.dummy_features, self.dummy_target, column_types
            )
            self.assertEqual(
                str(cm.exception),
                "The number of column_types does not match the number of" +
                "features plus the target",
                "Invalid number of column types test failed"
            )

def metafeatures_suite():
    # test_cases = [MetaFeaturesTestCase, MetaFeaturesWithDataTestCase]
    # return unittest.TestSuite(map(unittest.TestLoader().loadTestsFromTestCase, test_cases))
    suite = unittest.TestSuite()
    suite.addTest(MetaFeaturesWithDataTestCase("test_compare_openml"))
    return suite

