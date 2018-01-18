import codecs
import arff
import numpy as np
from metalearn.metafeatures.simple_metafeatures import SimpleMetafeatures
from metalearn.metafeatures.statistical_metafeatures import StatisticalMetafeatures
from metalearn.metafeatures.information_theoretic_metafeatures import InformationTheoreticMetafeatures
from metalearn.metafeatures.statistical_metafeatures import get_pca_values


def load_arff(infile_path, data_format="dict"):
    file_data = codecs.open(infile_path, "rb", "utf-8")
    arff_data = arff.load(file_data)
    data = np.array(arff_data["data"], dtype=object)
    Y = data[: ,-1]
    X = data[:,:-1]
    attributes = []
    for i in range(0,len(X[0])):
        attributes.append((arff_data["attributes"][i][0],str(type(data[0][i]))))
    attributes.append(('class', list(set(Y))))
    return X, Y, attributes

def extract_metafeatures(X,Y,attributes):
    metafeatures = {}
    features, time = SimpleMetafeatures().timed_compute(X,Y,attributes)
    print("simple metafeatures compute time: {}".format(time))
    total_time = time
    for key, value in features.items():
        metafeatures[key] = value

    features, time = StatisticalMetafeatures().timed_compute(X,Y,attributes)
    print("statistical metafeatures compute time: {}".format(time))
    total_time = total_time + time
    for key, value in features.items():
        metafeatures[key] = value

    features, time = InformationTheoreticMetafeatures().timed_compute(X,Y,attributes)
    print("information theoretic metafeatures compute time: {}".format(time))
    total_time = total_time + time
    for key, value in features.items():
        metafeatures[key] = value

    return metafeatures

def compute_metafeatures(dataset_path):
    X, Y, attributes = load_arff(dataset_path, "dict")
    metadata = get_pca_values(X, Y, attributes)
    return metadata

if __name__ == "__main__":
    print(compute_metafeatures("./iris.arff"))
