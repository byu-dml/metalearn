import json

import openml
import pandas as pd
from arff2pandas import a2p

from metalearn.metafeatures.metafeatures import Metafeatures


def import_openml_dataset(id=1):
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
    metafeatureDictPath = "oml_metafeature_map.json"
    mfNameMap = json.load(open(metafeatureDictPath, "r"))

    omlExclusiveMf = {x: v for x, v in omlMetafeatures.items()}
    ourExclusiveMf = {}
    sharedMfList = []
    sharedMf = pd.DataFrame(
        columns=("OML Metafeature Name", "OML Metafeature Value", "Our Metafeature Name", "Our Metafeature Value", "Similar?"))

    # iterate over outMetafeatures.items()
    for metafeatureName, metafeatureValue in ourMetafeatures.items():
        # if metafeatureName is not found in the dictionary, the metafeature is calculated
        # exclusively by us, and therefore it is added to ourExclusiveMf
        if mfNameMap.get(metafeatureName) is None:
            ourExclusiveMf[metafeatureName] = metafeatureValue
        else:
            # if oml DOES compute a metafeature that is equivalent to the metafeature represented by "metafeatureName"
            # BUT they did not compute said metafeature for this specific dataset, the metafeature is added to ourExclusiveMf
            # (ie, we're the only ones who calculated this metafeature)
            openmlName = mfNameMap[metafeatureName]["openmlName"]
            if omlMetafeatures.get(openmlName) is None:
                ourExclusiveMf[metafeatureName] = metafeatureValue

            # else both oml and us calculated the metafeature
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

    # print metafeatures calculated only by OpenML
    print("\nMetafeatures calculated by OpenML exclusively:")
    print(json.dumps(omlExclusiveMf, sort_keys=True, indent=4))




def load_arff(infile_path):
    f = open(infile_path)
    dataframe = a2p.load(f)
    column_name = [name for name in list(dataframe.columns) if 'class@' in name][0]
    dataframe = dataframe.rename(index=str, columns={column_name: 'target'})
    return dataframe


def extract_metafeatures(dataframe):
    metafeatures = {}
    features_df = Metafeatures().compute(dataframe)
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
    datasets = json.load(open("./data/test_datasets.json", "r"))
    for obj in datasets:
        filename = "./data/" + obj["path"]
        target_name = obj["target_name"]
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

        metafeatures = extract_metafeatures(dataframe)
        # print(json.dumps(sort_by_compute_time(metafeatures), indent=4))
        print(json.dumps(metafeatures, sort_keys=True, indent=4))
        # print(len(metafeatures), "metafeatures")
    print("tests finished")


if __name__ == "__main__":
    # TODO: pass in oml dataset id from command line
    dataframe, omlMetafeatures = import_openml_dataset()
    compare_with_openml(dataframe, omlMetafeatures)
    # main()
