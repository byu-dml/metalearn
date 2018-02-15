import json

import numpy as np
import pandas as pd
from arff2pandas import a2p

import openml

from metalearn.metafeatures.simple_metafeatures import SimpleMetafeatures
from metalearn.metafeatures.statistical_metafeatures import StatisticalMetafeatures
from metalearn.metafeatures.information_theoretic_metafeatures import InformationTheoreticMetafeatures
from metalearn.metafeatures.landmarking_metafeatures import LandmarkingMetafeatures


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

def sort_by_compute_time(metafeatures):
    metafeature_times = {}
    for key in metafeatures:
        if "_Time" in key:
            metafeature_times[key] = metafeatures[key]
    return dict(sorted(metafeature_times.items(), key=lambda x: x[1], reverse=True))

def main():
    # todo compare computed metafeatures against a static file of pre-computed metafeatures
    # this would allow us to see if we have fundamentally changed how we are computing metafeatures
    # during any development process
    # we then manually decide which metafeatures are correct and update the static file as needed

    for filename, target_name in [
        ("./data/LL0_1_anneal.csv", "class"),
        ("./data/LL0_12_mfeat_factors.csv", "class"),
        ("./data/LL0_337_spectf.csv", "OVERALL_DIAGNOSIS"),
        ("./data/iris.arff", "class"),
        ("./data/38_sick_train_data.csv","Class"),
        ("./data/LL0_40509_Australian.csv", "Y"),
        ("./data/LL0_4153_Smartphone_Based_Recognition_of_Human_Activities.csv", "Activity"),
        ("./data/LL0_1008_analcatdata_reviewer.csv", "binaryClass"),
        ("./data/LL0_42_soybean.csv", "class"),
        ("./data/LL0_155_pokerhand.csv", "class"),
        ("./data/LL0_475_analcatdata_germangss.csv", "Political_system"),
    ]:

        print(filename)
        ext = filename.split(".")[-1]
        if ext == "arff":
            dataframe = load_arff(filename)
        elif ext == "csv":
            dataframe = pd.read_csv(filename)
            dataframe.rename(columns={target_name: "target"}, inplace=True)
        else:
            raise ValueError("load file type '{}' not implemented")

        if "d3mIndex" in dataframe.columns:
            dataframe.drop(columns="d3mIndex", inplace=True)

        if dataframe.shape[0] > 100000 or dataframe.shape[1] > 50:
            print("skipped")
            continue
        metafeatures = extract_metafeatures(dataframe)
        # print(json.dumps(sort_by_compute_time(metafeatures), indent=4))
        # print(json.dumps(metafeatures, sort_keys=True, indent=4))
        # print(len(metafeatures), "metafeatures")
    # print("tests finished")

if __name__ == "__main__":
    # dataframe, omlMetafeatures = import_openml_dataset()
    # compare_with_openml(dataframe,omlMetafeatures)
    main()
