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
            # print(json.dumps(metafeatures_dict, sort_keys=True, indent=4))

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

    def test_compare_openml(self):


        def import_openml_datasets():

            # get a list of datasets from openml
            datasets = openml.datasets.list_datasets()
            datasets = pd.DataFrame.from_dict(datasets, orient='index')

            #get a listed of filtered dataset ids
            dataset_indices = [31, 1464, 334, 50, 333, 1570, 1504, 1494, 3, 1510, 1489, 37, 1479, 1063, 1471, 1467, 1487, 44, 1067, 1493, 1480, 1492, 1068, 1491, 1050, 1462, 1046, 335, 151, 1049, 1116, 312, 1485, 1457, 1220, 1038, 1120, 1461, 6, 1486, 4534, 300, 183, 4134, 42, 1515, 4135, 40536, 28, 16, 18, 22, 32, 20, 12, 14, 1501, 1466, 1459, 375, 36, 1468, 469, 2, 188, 182, 307, 377, 54, 29, 23380, 11, 1549, 1555, 458, 15, 1476, 1478, 1475, 4538, 23, 1233, 60, 470, 451, 24, 61, 1497, 46, 23512, 4, 554, 6332, 1554, 1552, 38, 1053, 1114, 1112, 23381, 40496, 1553, 1590, 5, 1548, 40499, 43, 1547, 53, 1128, 1137, 1138, 1166, 1158, 1134, 1165, 1130, 1145, 1161, 179, 30, 59, 9, 40, 40668, 56, 181, 13, 55, 48, 52, 10, 26, 27, 782, 184, 51, 41, 39, 49, 34, 7, 35, 137, 172, 40900, 336, 885, 867, 171, 313, 163, 875, 186, 736, 916, 895, 187, 754, 974, 1013, 969, 829, 448, 726, 337, 464, 346, 921, 890, 119, 784, 811, 747, 902, 714, 461, 955, 444, 783, 748, 719, 255, 338, 789, 878, 762, 808, 278, 860, 40910, 277, 276, 814, 1075, 251, 685, 343, 450, 339, 342, 1069, 340, 803, 976, 730, 776, 733, 275, 911, 57, 1026, 925, 744, 886, 918, 879, 900, 1056, 943, 1011, 931, 896, 794, 949, 880, 932, 994, 889, 937, 792, 926, 933, 820, 795, 970, 871, 888, 788, 948, 936, 807, 1020, 796, 868, 995, 996, 774, 793, 185, 909, 934, 935, 779, 775, 906, 876, 1061, 884, 869, 877, 805, 1018, 1064, 756, 830, 766, 1021, 770, 973, 893, 908, 951, 870, 1014, 979, 997, 1037, 749, 812, 947, 873, 804, 763, 922, 923, 962, 819, 824, 841, 863, 958, 1005, 894, 724, 920, 838, 768, 1065, 950, 764, 753, 716, 978, 1054, 1066, 980, 746, 772, 907, 991, 834, 850, 1002, 752, 778, 816, 941, 725, 761, 741, 945, 1017, 1071, 750, 971, 832, 847, 790, 818, 1025, 915, 769, 717, 946, 481, 735, 1003, 765, 874, 817, 1059, 898, 732, 855, 882, 1045, 720, 857, 737, 773, 848, 833, 1015, 993, 1073, 1006, 1048, 721, ]

            #randomly sample (2-5) of these datasets
            num_datasets = random.randint(2,5)
            rand_dataset_ids = random.sample(dataset_indices,num_datasets)
            #rand_dataset_ids = [470]

            # get X, Y, and metafeatures from the datasets
            oml_datasets = []
            for raw_dataset_id in rand_dataset_ids:
                raw_dataset = openml.datasets.get_dataset(raw_dataset_id)
                dataset_metafeatures = raw_dataset.qualities.items()
                dataset_metafeatures = dict(dataset_metafeatures)
                X_raw, Y_raw, attributes= raw_dataset.get_data(target=raw_dataset.default_target_attribute, return_attribute_names=True)
                X = pd.DataFrame(data=X_raw, columns=attributes)
                Y = pd.Series(data=Y_raw, name="target")
                dataset = {"X": X, "Y": Y, "metafeatures": dataset_metafeatures, "attributes": attributes}
                oml_datasets.append(dataset)
                compare_with_openml(X, Y, dataset_metafeatures, raw_dataset_id)

            return oml_datasets

        def compare_with_openml(X,Y, omlMetafeatures, dataset_id):
            # get metafeatures from dataset using our metafeatures
            ourMetafeatures = Metafeatures().compute(X=X,Y=Y, sample_rows=False, sample_columns=False)
            ourMetafeatures = ourMetafeatures.to_dict(orient='records')
            ourMetafeatures = dict(ourMetafeatures[0])
            # todo use nested dictionary instead of tuple to make values more descriptive
            mfDict = json.load(open("oml_metafeature_map.json", "r"))
            omlExclusiveMf = omlMetafeatures.copy()
            ourExclusiveMf = ourMetafeatures.copy()
            consistentSharedMf = []
            inconsistentSharedMf = []
            #sharedMf = pd.DataFrame(columns=("OML Metafeature Name", "OML Metafeature Value", "Our Metafeature Name", "Our Metafeature Value", "Similar?"))
            similarityQualifier = .05
            for omlMetafeature in omlMetafeatures :
                # compare shared metafeatures
                if (ourMetafeatures.get(omlMetafeature) != None 
                    or ourMetafeatures.get("" if omlMetafeature not in mfDict else mfDict.get(omlMetafeature)["ourName"]) != None) :

                    # compare metafeatures with the same name
                    if (ourMetafeatures.get(omlMetafeature) != None):
                        omlMetafeatureName = omlMetafeature + "_oml"
                        omlMetafeatureValue = float(omlMetafeatures.get(omlMetafeature))
                        ourMetafeatureName = omlMetafeature
                        ourMetafeatureValue = ourMetafeatures.get(ourMetafeatureName)
                        diff = omlMetafeatureValue - ourMetafeatureValue
                    # compare equivalent metafeatures with different names
                    elif (ourMetafeatures.get(mfDict.get(omlMetafeature)["ourName"]) != None):
                        ourMetafeatureName = mfDict.get(omlMetafeature)["ourName"]
                        multiplier = mfDict.get(omlMetafeature)["multiplier"]
                        ourMetafeatureValue = ourMetafeatures.get(ourMetafeatureName)
                        omlMetafeatureName = omlMetafeature
                        omlMetafeatureValue = float(omlMetafeatures.get(omlMetafeature))
                        diff = omlMetafeatureValue - (ourMetafeatureValue * multiplier)
                        
                    # determine if the metafeatures are similar. Add to respective shared dictionary
                    tempMfDict = {omlMetafeatureName: omlMetafeatureValue, ourMetafeatureName: ourMetafeatureValue, "difference": diff}
                    if (abs(diff) <= similarityQualifier):
                        similarityString = "Yes"
                        consistentSharedMf.append(tempMfDict)
                    else:
                        similarityString = "No"
                        inconsistentSharedMf.append(tempMfDict)

                    #update exclusive dictionaries
                    omlExclusiveMf.pop(omlMetafeatureName, None)
                    ourExclusiveMf.pop(ourMetafeatureName, None)

            #write results to json file
            openmlData = { "Inconsistent Shared Metafeatures": inconsistentSharedMf, "Consistent Shared Metafeatures": consistentSharedMf,
                "Our Exclusive Metafeatures": ourExclusiveMf, "OpenML Exclusive Metafeatures": omlExclusiveMf}
            report_file = './test/metalearn/metafeatures/openmlComparisons/openml_comparison_' + str(dataset_id) + '.json'
            with open(report_file,'w') as fh:
                json.dump(openmlData, fh, indent=4)

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
