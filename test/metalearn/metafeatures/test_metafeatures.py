""" Contains unit tests for the MetaFeatures class. """
import json
import math
import os
import random
import unittest

import numpy as np
import pandas as pd
from arff2pandas import a2p

import openml

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
        self.dataframes = {}
        self.data_folder = './data/'
        with open(self.data_folder + "test_datasets.json", "r") as fh:
            datasets = json.load(fh)

        for obj in datasets:
            filename = obj["path"]
            target_name = obj["target_name"]

            dataframe = load_data(self.data_folder + filename)
            dataframe.rename(columns={target_name: "target"}, inplace=True)

            if "d3mIndex" in dataframe.columns:
                dataframe.drop(columns="d3mIndex", inplace=True)

            self.dataframes[filename] = dataframe

    def tearDown(self):
        del self.dataframes
        # print(json.dumps(sort_by_compute_time(metafeatures), indent=4))
        # print(len(metafeatures), "metafeatures")

    def test_run_without_fail(self):
        for filename, dataframe in self.dataframes.items():
            metafeatures_df = Metafeatures().compute(dataframe)
            metafeatures_dict = metafeatures_df.to_dict('records')[0]
            # print(json.dumps(metafeatures_dict, sort_keys=True, indent=4))

    def test_correctness(self):
        """ For each dataset that has a corresponding mf (metafeature) file present,
            check differences in columns we do not expect to change.
        """
        # Known value file was made with seeds set to 0 (some mfs use randomness)
        np.random.seed(0)
        random.seed(0)

        fails = {}
        for filename, dataframe in self.dataframes.items():
            last_results_file = filename.replace('data','mf').replace('.csv','.json')
            if os.path.exists(self.data_folder + last_results_file):
                with open(self.data_folder + last_results_file) as fh:
                    known_mfs = json.load(fh)

                # Explicitly create empty dict because this provides information about successful tests.
                fails[last_results_file] = {}

                metafeatures_df = Metafeatures().compute(dataframe)
                computed_mfs = metafeatures_df.to_dict('records')[0]

                for mf, computed_value in computed_mfs.items():
                    if '_Time' in mf:
                        # Timing metafeatures will always differ anyway.
                        # For now we pay no mind, no matter how big a difference may be.
                        continue

                    known_value = known_mfs[mf]
                    if not math.isclose(known_value, computed_value):
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



class MetaFeaturesTestCase(unittest.TestCase):
    """ Contains tests for MetaFeatures that can be executed without loading data. """

    def setUp(self):
        self.dummy_df = pd.DataFrame(np.random.rand(50,50))
        self.dummy_df.columns = [*self.dummy_df.columns[:-1], "target"]

        self.invalid_metafeature_message_start = "One or more requested metafeatures are not valid:"
        self.invalid_metafeature_message_start_fail_message = "Error message indicating invalid metafeatures did not start with expected string."
        self.invalid_metafeature_message_contains_fail_message = "Error message indicating invalid metafeatures should include names of invalid features."

    def test_dataframe_input_error(self):
        """ Tests if `compute` gives a user-friendly error when a TypeError occurs. """

        expected_error_message = "DataFrame has to be Pandas DataFrame."
        fail_message = "We expect a user friendly message the object passed to compute is not a Pandas.DataFrame."
        # We don't check for the Type of TypeError explicitly as any other error would fail the unit test.

        with self.assertRaises(TypeError) as cm:
            Metafeatures().compute(dataframe = None)
        self.assertEqual(str(cm.exception), expected_error_message, fail_message)

        with self.assertRaises(TypeError) as cm:
            Metafeatures().compute(dataframe = np.zeros((500,50)))
        self.assertEqual(str(cm.exception), expected_error_message, fail_message)

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
            Metafeatures().compute(dataframe = self.dummy_df, metafeatures = invalid_metafeatures)

        self._check_invalid_metafeature_exception_string(str(cm.exception), invalid_metafeatures)

    def test_metafeatures_input_partial_invalid(self):
        """ Test case where only some requested metafeatures are invalid. """

        invalid_metafeatures = ["ThisIsNotValid", "ThisIsAlsoNotValid"]
        valid_metafeatures = ["NumberOfInstances", "NumberOfFeatures"]

        with self.assertRaises(ValueError) as cm:
            Metafeatures().compute(dataframe = self.dummy_df, metafeatures = invalid_metafeatures+valid_metafeatures)

        self._check_invalid_metafeature_exception_string(str(cm.exception), invalid_metafeatures)

        # Order should not matter
        with self.assertRaises(ValueError) as cm:
            Metafeatures().compute(dataframe = self.dummy_df, metafeatures = valid_metafeatures+invalid_metafeatures)
        self._check_invalid_metafeature_exception_string(str(cm.exception), invalid_metafeatures)


def metafeatures_suite():
    test_cases = [MetaFeaturesTestCase, MetaFeaturesWithDataTestCase]
    return unittest.TestSuite(map(unittest.TestLoader().loadTestsFromTestCase, test_cases))

""" === Anything under is line is currently not in use. === """
def import_openml_dataset(id=4):
    # get a dataset from openml using a dataset id
    dataset = openml.datasets.get_dataset(id)
    # get the metafeatures from the dataset
    omlMetafeatures = dataset.qualities
    # get X, Y, and attributes from the dataset
    X, Y, attributes = dataset.get_data(target=dataset.default_target_attribute, return_attribute_names=True)

    # create datafrom object from X,Y, and attributes
    dataframe = pd.DataFrame(X, columns=attributes)
    dataframe = dataframe.assign(target=pd.Series(Y))

    # format attributes
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
                similarityString = "no"

            sharedMf.append((omlMetafeatureName, str(omlMetafeatureValue), ourMetafeatureName, str(ourMetafeatureValue), similarityString))
            del ourExclusiveMf[ourMetafeatureName]
        # Grab metafeatures computed by OpenML and not us
        else :
            omlExclusiveMf[omlMetafeature] = omlMetafeatures[omlMetafeature]


    # print shared metafeature comparison
    print("Shared metafeature comparison")
    print("{0:40} {1:30} {2:40} {3:30} {4:5}".format(sharedMf[0][0],sharedMf[0][1],sharedMf[0][2],sharedMf[0][3],sharedMf[0][4]).rjust(3))
    for x in sharedMf :
        if (x[4] == "yes") :
            print("{0:40} {1:30} {2:40} {3:30} {4:5}".format(x[0],x[1],x[2],x[3],x[4]).rjust(3))

    for x in sharedMf :
        if (x[4] == "no") :
            print("{0:40} {1:30} {2:40} {3:30} {4:5}".format(x[0],x[1],x[2],x[3],x[4]).rjust(3))

    # print metafeatures calculated only by OpenML
    print("\nMetafeatures calculated by OpenML exclusively:")
    print(json.dumps(omlExclusiveMf, sort_keys=True, indent=4))

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
