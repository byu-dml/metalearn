""" Contains unit tests for the MetaFeatures class.
# TODO
# compare computed metafeatures against a static file of pre-computed metafeatures
# this would allow us to see if we have fundamentally changed how we are computing metafeatures
# during any development process
# we then manually decide which metafeatures are correct and update the static file as needed
"""
import json
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
        raise ValueError("load file type '{}' not implemented")
    return dataframe

class MetaFeaturesWithDataTestCase(unittest.TestCase):
    """ Contains tests for MetaFeatures that require loading data first. """
    
    def setUp(self):
        self.dataframes = {}
        data_folder = './data/'
        with open(data_folder + "test_datasets.json", "r") as fh:
            datasets = json.load(fh)
            
        for obj in datasets:
            filename = data_folder + obj["path"]
            target_name = obj["target_name"]
            
            dataframe = load_data(filename)
            dataframe.rename(columns={target_name: "target"}, inplace=True)
            
            if "d3mIndex" in dataframe.columns:
                dataframe.drop(columns="d3mIndex", inplace=True)
            
            self.dataframes[filename] = dataframe
        
    def tearDown(self):
        del self.dataframes
        # print(json.dumps(sort_by_compute_time(metafeatures), indent=4))
        # print(len(metafeatures), "metafeatures")
    
    def test_correctness(self):
        for filename, dataframe in self.dataframes.items():
            print(filename)
            metafeatures_df = Metafeatures().compute(dataframe)
            metafeatures_dict = metafeatures_df.to_dict('records')[0]
            print(json.dumps(metafeatures_dict, sort_keys=True, indent=4))
        
class MetaFeaturesTestCase(unittest.TestCase):
    """ Contains tests for MetaFeatures that can be executed without loading data. """
    
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
            similarityString = ""
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
