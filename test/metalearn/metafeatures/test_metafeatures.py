""" Contains unit tests for the MetaFeatures class. """
import json
import math
import os
import random
import unittest
import copy
import time

import openml
import pandas as pd
import numpy as np
from arff2pandas import a2p
import arff

from metalearn.metafeatures.metafeatures import Metafeatures

def load_arff(infile_path):
    """ Loads and ARFF file to a pandas dataframe and drops meta-info on column type. """
    with open(infile_path) as fh:
        df = a2p.load(fh)
    # Default column names follow ARFF, e.g. petalwidth@REAL, class@{a,b,c}
    df.columns = [col.split('@')[0] for col in df.columns]
    return df

def load_data(filename):
    """ Loads a csv or arff file (provided they are named *.{csv|arff}) """
    ext = filename.split(".")[-1]
    if ext == "arff":
        dataframe = load_arff(filename)
    elif ext == "csv":
        dataframe = pd.read_csv(filename)
    else:
        raise ValueError("load file type '{}' not implemented".format(ext))
    return dataframe

class MetaFeaturesWithDataTestCase(unittest.TestCase):
    """ Contains tests for MetaFeatures that require loading data first. """

    def setUp(self):
        self.datasets = {}
        self.data_folder = './data/'
        with open(self.data_folder + "test_datasets.json", "r") as fh:
            dataset_descriptions = json.load(fh)

        for obj in dataset_descriptions:
            filename = obj["path"]
            target_name = obj["target_name"]

            data = load_data(self.data_folder + filename)
            X = data.drop(columns=[target_name], axis=1)
            X.drop(columns=["d3mIndex"], axis=1, inplace=True, errors="ignore")
            Y = data[target_name]
            self.datasets[filename] = {"X": X, "Y": Y}

    def tearDown(self):
        del self.datasets
        # print(json.dumps(sort_by_compute_time(metafeatures), indent=4))
        # print(len(metafeatures), "metafeatures")

    def test_run_without_fail(self):
        for filename, dataset in self.datasets.items():
            metafeatures_df = Metafeatures().compute(X=dataset["X"],Y=dataset["Y"])
            metafeatures_dict = metafeatures_df.to_dict('records')[0]
            #print(json.dumps(metafeatures_dict, sort_keys=True, indent=4))

    def _get_last_mf_results_filename(self, dataset_filename):
        ext = dataset_filename.rsplit(".", 1)[1]
        search_str = "." + ext
        return dataset_filename.replace(search_str, "_mf.json")

    def test_correctness(self):
        """ For each dataset that has a corresponding mf (metafeature) file present,
            check differences in columns we do not expect to change.
        """
        random_seed = 0

        fails = {}
        for filename, dataset in self.datasets.items():
            last_results_file = self._get_last_mf_results_filename(filename)
            if os.path.exists(self.data_folder + last_results_file):
                with open(self.data_folder + last_results_file) as fh:
                    known_mfs = json.load(fh)

                # Explicitly create empty dict because this provides information about successful tests.
                fails[last_results_file] = {}

                metafeatures_df = Metafeatures().compute(X=dataset["X"],Y=dataset["Y"],seed=random_seed)
                computed_mfs = metafeatures_df.to_dict('records')[0]

                for mf, computed_value in computed_mfs.items():
                    if '_Time' in mf:
                        # Timing metafeatures will always differ anyway.
                        # For now we pay no mind, no matter how big a difference may be.
                        continue

                    known_value = known_mfs[mf]
                    if not math.isclose(known_value, computed_value) and not (np.isnan(known_value) and np.isnan(computed_value)):
                        fails[last_results_file][mf] = (known_value, computed_value)

        self.assertGreater(len(fails), 0, "No known results could be loaded, correctness could not be verified.")
        if not all(f == {} for f in fails.values()):
            # Results are no longer correct. Because multiple results that can be wrong are calculated at once,
            # we want to output all of the wrong results so it might be easier to find out what went wrong.
            fails = {k:v for (k,v) in fails.items() if v != {}}
            fail_report_file = './test/metalearn/metafeatures/correctness_fails.json'
            with open(fail_report_file,'w') as fh:
                json.dump(fails, fh, indent=4)

            self.assertTrue(False, "Not all metafeatures matched previous results, output written to {}.".format(fail_report_file))

    #@unittest.expectedFailure
    def test_compare_openml(self):

        def import_openml_datasets():

            # get a list of datasets from openml
            datasets_dict = openml.datasets.list_datasets()
            datasets = [8, 1547, 1552, 1553, 1554, 40693, 4551, 1490, 230, 189, 198, 561, 562, 203,]
            datasets = list([k for k,v in datasets_dict.items() if v["NumberOfInstances"] < 100000])

            random.shuffle(datasets)
            #get a list of filtered dataset ids
            dataset_indices = [ 40601]
            #randomly sample (2-5) of these datasets
            num_datasets = random.randint(2,5)
            rand_dataset_ids = random.sample(datasets, num_datasets)
            #rand_dataset_ids = datasets
            rand_dataset_ids = datasets
            #rand_dataset_ids = [564]

            # get X, Y, and metafeatures from the datasets
            oml_datasets = []
            inconsistencies = False
            for dataset_id in rand_dataset_ids:
                try:
                    print(dataset_id)
                    dataset = openml.datasets.get_dataset(dataset_id)
                    target = str(dataset.default_target_attribute).split(",")
                    df = load_arff(dataset.data_file)
                    X = df.drop(columns=target, axis=1)
                    if len(target) <= 1:
                        Y = df[target].squeeze()
                        if Y.dtype == "object":
                            dataset_metafeatures = {x: (float(v) if v is not None else v) for x,v in dataset.qualities.items()}
                            dataset = {"X": X, "Y": Y, "metafeatures": dataset_metafeatures}
                            if compare_with_openml(dataset, dataset_id):
                                inconsistencies = True
                    else:
                        print("multi label")
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except arff.BadNominalValue:
                    continue

            self.assertFalse(inconsistencies, "Not all metafeatures matched results from OpenML.")

        def compare_with_openml(oml_dataset, dataset_id):
            # get metafeatures from dataset using our metafeatures
            ourMetafeatures = Metafeatures().compute(X=oml_dataset["X"], Y=oml_dataset["Y"])
            ourMetafeatures = ourMetafeatures.to_dict(orient="records")[0]

            mfNameMap = json.load(open("oml_metafeature_map.json", "r"))

            omlExclusiveMf = {x: v for x,v in oml_dataset["metafeatures"].items()}
            ourExclusiveMf = {}
            consistentSharedMf = []
            inconsistentSharedMf = []

            similarityQualifier = .05
            for metafeatureName, metafeatureValue in ourMetafeatures.items():
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
                        diff = abs(omlMetafeatureValue/multiplier - metafeatureValue)
                        singleMfDict = {metafeatureName: {"OpenML Value": omlMetafeatureValue/multiplier, "Our Value": metafeatureValue, "Difference": diff}}
                        if diff <= .05:
                            consistentSharedMf.append(singleMfDict)
                        elif diff > .05 or diff == np.isnan(diff):
                            inconsistentSharedMf.append(singleMfDict)


            #write results to json file
            openmlData = { "INCONSISTENT SHARED METAFEATURES": inconsistentSharedMf, "CONSISTENT SHARED METAFEATURES": consistentSharedMf,
                "OUR EXCLUSIVE METAFEATURES": ourExclusiveMf, "OPENML EXCLUSIVE METAFEATURES": omlExclusiveMf}
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


    def test_timeout(self):
        '''Tests whether the Metafeatures.compute function returns within the allotted time.'''
        for filename, dataset in self.datasets.items():
            for timeout in [3, 5, 10]:
                mf = Metafeatures()
                start_time = time.time()
                mf.compute(X=dataset["X"], Y=dataset["Y"], timeout=timeout)
                compute_time = time.time() - start_time
                self.assertGreater(timeout, compute_time, "computing metafeatures exceeded max time. dataset: '{}', max time: {}, actual time: {}".format(filename, timeout, compute_time))


class MetaFeaturesTestCase(unittest.TestCase):
    """ Contains tests for MetaFeatures that can be executed without loading data. """

    def setUp(self):
        self.dummy_features = pd.DataFrame(np.random.rand(50,50))
        self.dummy_target = pd.Series(np.random.randint(2, size=50)).rename("target")

        self.invalid_metafeature_message_start = "One or more requested metafeatures are not valid:"
        self.invalid_metafeature_message_start_fail_message = "Error message indicating invalid metafeatures did not start with expected string."
        self.invalid_metafeature_message_contains_fail_message = "Error message indicating invalid metafeatures should include names of invalid features."

    def test_dataframe_input_error(self):
        """ Tests if `compute` gives a user-friendly error when a TypeError occurs. """

        expected_error_message1 = "X must be of type pandas.DataFrame"
        fail_message1 = "We expect a user friendly message when the features passed to compute is not a Pandas.DataFrame."
        expected_error_message2 = "Y must be of type pandas.Series"
        fail_message2 = "We expect a user friendly message when the target column passed to compute is not a Pandas.Series."
        # We don't check for the Type of TypeError explicitly as any other error would fail the unit test.

        with self.assertRaises(TypeError) as cm:
            Metafeatures().compute(X=None, Y=self.dummy_target)
        self.assertEqual(str(cm.exception), expected_error_message1, fail_message1)

        with self.assertRaises(TypeError) as cm:
            Metafeatures().compute(X=self.dummy_features, Y=None)
        self.assertEqual(str(cm.exception), expected_error_message2, fail_message2)

        with self.assertRaises(TypeError) as cm:
            Metafeatures().compute(X=np.zeros((500,50)), Y=pd.Series(np.zeros(500)))
        self.assertEqual(str(cm.exception), expected_error_message1, fail_message1)

        with self.assertRaises(TypeError) as cm:
            Metafeatures().compute(X=pd.DataFrame(np.zeros((500,50))), Y=np.zeros(500))
        self.assertEqual(str(cm.exception), expected_error_message2, fail_message2)

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
