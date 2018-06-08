""" Contains unit tests for the MetaFeatures class. """
import json
import math
import os
import random
import unittest
import time

# import openml
import pandas as pd
import numpy as np

from metalearn.metafeatures.metafeatures import Metafeatures
from test.data.dataset import read_dataset
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

    def test_timeout(self):
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
    test_cases = [MetaFeaturesTestCase, MetaFeaturesWithDataTestCase]
    return unittest.TestSuite(map(unittest.TestLoader().loadTestsFromTestCase, test_cases))

""" === Anything under is line is currently not in use. === """
def import_openml_dataset(id=4):
    # get a dataset from openml using a dataset id
    dataset = openml.datasets.get_dataset(id)
    # get the metafeatures from the dataset
    omlMetafeatures = {x: float(v) for x, v in dataset.qualities.items()}

    # get X, Y, and attributes from the dataset
    X, Y, attributes = dataset.get_data(target=dataset.default_target_attribute, return_attribute_names=True)

    # create dataframe object from X,Y, and attributes
    dataframe = pd.DataFrame(X, columns=attributes)
    dataframe = dataframe.assign(target=pd.Series(Y))

    # format attributes
    # TODO: find out if pandas infers type correctly (remove this code after)
    for i in range(len(X[0])):
        attributes[i] = (attributes[i], str(type(X[0][i])))
        # set types of attributes (column headers) as well as the names

    return dataframe, omlMetafeatures

def compare_with_openml(dataframe, omlMetafeatures):
    # get metafeatures from dataset using our metafeatures
    ourMetafeatures = extract_metafeatures(dataframe)
    # todo use nested dictionary instead of tuple to make values more descriptive
    mfDict = json.load(open("oml_metafeature_map.json", "r"))

    omlExclusiveMf = {}
    ourExclusiveMf = ourMetafeatures
    sharedMf = []
    sharedMf.append(("OML Metafeature Name", "OML Metafeature Value", "Our Metafeature Name", "Our Metafeature Value", "Similar?"))
    for omlMetafeature in omlMetafeatures :
        # compare shared metafeatures
        if (ourMetafeatures.get(omlMetafeature) != None
            or ourMetafeatures.get("" if omlMetafeature not in mfDict else mfDict.get(omlMetafeature)[0]) != None) :
            omlMetafeatureName= ""
            omlMetafeatureValue= ""
            ourMetafeatureName= ""
            ourMetafeatureValue= ""
            similarityString= ""
            diff = 0
            similarityQualifier = 0.05

            # compare metafeatures with the same name
            if (ourMetafeatures.get(omlMetafeature) != None):
                omlMetafeatureName = omlMetafeature
                omlMetafeatureValue = float(omlMetafeatures.get(omlMetafeature))
                ourMetafeatureName = omlMetafeature
                ourMetafeatureValue = float(ourMetafeatures.get(ourMetafeatureName))
                # similarityQualifier = omlMetafeatureValue * .05
                diff = omlMetafeatureValue - ourMetafeatureValue
            # compare equivalent metafeatures with different names
            elif (ourMetafeatures.get(mfDict.get(omlMetafeature)[0]) != None):
                ourMetafeatureName, multiplier = mfDict.get(omlMetafeature)
                ourMetafeatureValue = float(ourMetafeatures.get(ourMetafeatureName))
                omlMetafeatureName = omlMetafeature
                omlMetafeatureValue = float(omlMetafeatures.get(omlMetafeature))
                # similarityQualifier = omlMetafeatureValue * .05
                diff = omlMetafeatureValue - (ourMetafeatureValue * multiplier)

            # determine if the metafeatures are similar
            if (abs(diff) <= similarityQualifier):
                similarityString = "yes"
            else:
                # compare oml value with our value, get diff between the two
                diff = abs(omlMetafeatures[openmlName] - metafeatureValue)
                if diff > .05:
                    similarityString = "No"
                else:
                    similarityString = "Yes"

                # sharedMfList is a pandas dataframe. We add a row consisting of the following values:
                # "OML Metafeature Name", "OML Metafeature Value", "Our Metafeature Name", "Our Metafeature Value", "Similar?"
                sharedMfList.append(
                    [openmlName, omlMetafeatures[openmlName], metafeatureName, metafeatureValue, similarityString])

                omlExclusiveMf.pop(openmlName)


    for index, row in enumerate(sharedMfList):
        sharedMf.loc[index] = row


    # print shared metafeature comparison
    print("Shared metafeature comparison")
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    sharedMf.sort_values("Similar?", ascending=False, axis=0, inplace=True)

    print(sharedMf)

    # print metafeatures calculate by our primitive exclusively
    print("\nMetafeatures calculated by our primitive exclusively:")
    print(json.dumps(ourExclusiveMf, sort_keys=True, indent=4))

def sort_by_compute_time(metafeatures):
    metafeature_times = {}
    for key in metafeatures:
        if "_Time" in key:
            metafeature_times[key] = metafeatures[key]
    return dict(sorted(metafeature_times.items(), key=lambda x: x[1], reverse=True))

#if __name__ == "__main__":
# dataframe, omlMetafeatures = import_openml_dataset()
# compare_with_openml(dataframe,omlMetafeatures)
