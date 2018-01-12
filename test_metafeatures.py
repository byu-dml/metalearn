import codecs
import arff
import numpy as np
import openml
from metalearn.metafeatures.simple_metafeatures import SimpleMetafeatures
from metalearn.metafeatures.statistical_metafeatures import StatisticalMetafeatures
from metalearn.metafeatures.information_theoretic_metafeatures import InformationTheoreticMetafeatures


def load_arff(infile_path, data_format="dict"):
    file_data = codecs.open(infile_path, "rb", "utf-8")
    arff_data = arff.load(file_data)
    data = np.array(arff_data["data"], dtype=object)
    Y = data[: ,-1]
    X = data[:,:-1]
    attributes = []
    for i in range(len(X[0])):
        attributes.append((arff_data["attributes"][i][0],str(type(data[0,i]))))
    attributes.append(('class', list(set(Y))))
    return X, Y, attributes

def import_openml_dataset(id=1):
    # get a dataset from openml using a dataset id
    dataset = openml.datasets.get_dataset(id)
    # get the metafeatures from the dataset
    omlMetafeatures = dataset.qualities
    # get X, Y, and attributes from the dataset
    X, Y, attributes = dataset.get_data(target=dataset.default_target_attribute, return_attribute_names=True)

    # format attributes
    for i in range(len(X[0])):
        attributes[i] = (attributes[i], str(type(X[0][i])))

    return X, Y, attributes, omlMetafeatures

def compare_with_openml(X,Y,attributes,omlMetafeatures):
    # get metafeatures from dataset using our metafeatures

    print("\nopenml Metafeatures: \n",omlMetafeatures,"\n\n")
    metafeatures = extract_metafeatures(X,Y,attributes)
    print("\nOur Metafeatures: \n",metafeatures,"\n\n")



def extract_metafeatures(X,Y,attributes):
    metafeatures = {}
    features, time = SimpleMetafeatures().timed_compute(X,Y,attributes)
    print("simple metafeatures compute time: {}".format(time))
    total_time = time
    metafeatures.update(features)

    features, time = StatisticalMetafeatures().timed_compute(X,Y,attributes)
    print("statistical metafeatures compute time: {}".format(time))
    total_time += time
    metafeatures.update(features)

    features, time = InformationTheoreticMetafeatures().timed_compute(X,Y,attributes)
    print("information theoretic metafeatures compute time: {}".format(time))
    total_time += time
    metafeatures.update(features)

    return metafeatures

def compute_metafeatures(dataset_path):
    X, Y, attributes = load_arff(dataset_path, "dict")
    metadata = extract_metafeatures(X, Y, attributes)
    return metadata

if __name__ == "__main__":
    # print(compute_metafeatures("./iris.arff"))
    print("\n\n\n")
    X,Y,attributes,omlMetafeatures = import_openml_dataset()
    compare_with_openml(X,Y,attributes,omlMetafeatures)
    

