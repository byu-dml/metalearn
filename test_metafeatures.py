import os
import json
import codecs
import arff
import numpy as np
from metalearn.features.metafeatures import MetaFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


def file_find(base_dir, file_type=".arff"):
    paths = []
    for base, directories, files in os.walk(base_dir):
        for file_name in files:
            if(file_type in file_name):
                paths.append(base+"/"+file_name)
    return paths

def get_dataset_paths():
    return ['./test/iris.arff']
    '''
    base_dir = "/home/bschoenfeld/Research/datasets/"
    dataset_names = open("test/small_dataset_names.csv", "r").read().strip().split(",")
    raw_paths = file_find(base_dir)
    paths = []
    for rp in raw_paths:
        for name in dataset_names:
            if name in rp:
                paths.append(rp)
                break
    return paths
    '''

def get_dataset_name_from_path(dataset_path):
    return dataset_path.split("/")[-1].split(".")[0]

def load_arff(infile_path, data_format="dict"):
    file_data = codecs.open(infile_path, "rb", "utf-8")
    arff_data = arff.load(file_data)
    raw_attribute_names = arff_data["attributes"]
    attribute_names = [attr[0] for attr in raw_attribute_names]
    raw_data = np.array(arff_data["data"], dtype=object)
    if data_format == "dict":
        features = {attr: raw_data[:,i] for i, attr in enumerate(attribute_names[:-1])}
    else:
        features = raw_data[:,:-1]
    labels = raw_data[:,-1]
    return features, labels

def extract_metafeatures(features, labels):    
    MF = MetaFeatures()
    outputs = MF.produce(inputs = [2, features, labels])
    metadata_dict = outputs[1]
    return metadata_dict

def compute_metafeatures(dataset_paths, outfile_path):    
    with open(outfile_path, "w") as f:        
        for path in dataset_paths:
            #try:                
            dataset_name = get_dataset_name_from_path(path)
            print("\ndataset: '{}'".format(dataset_name))
            features, labels = load_arff(path, "dict")
            metadata = extract_metafeatures(features, labels)
            #
            return metadata
            #
            output = {"dataset": dataset_name, "metadata": metadata}
            f.write(json.dumps(output)+"\n")
            print("success")
            #except Exception as e:
                #print("fail:\n{}".format(e))

def get_score(clf, features, labels):
    scores = cross_val_score(clf, features, labels, cv=10, scoring='accuracy')
    return scores.mean()

def compute_metalabels(dataset_paths, outfile_path, errfile_path):
    with open(outfile_path, "w") as outfile, open(errfile_path, "w") as errfile:
        outfile.write("Dataset, Best, BestAcc, Worst, WorstAcc\n")
        for dataset_path in dataset_paths:
            try:
                dataset_name = get_dataset_name_from_path(dataset_path)
                print("Dataset found: ", dataset_name)
                features, labels = load_arff(dataset_path, "array")
                tree_score = get_score(DecisionTreeClassifier(), features, labels)
                knn_score = get_score(KNeighborsClassifier(n_neighbors = 3), features, labels)
                if(tree_score >= knn_score):
                    outfile.write(dataset_name + ",DecisionTree," + str(tree_score) + ",kNN," + str(knn_score) + "\n")
                    print("Result: Decision Tree")
                else:
                    outfile.write(dataset_name + ",kNN," + str(knn_score) + ",DecisionTree," + str(tree_score) + "\n")
                    print("Result: kNN")
            except Exception as e:
                print("Error:", e)
                errfile.write(dataset_name+"\n")

def combine_metafeatures_and_metalabels(metafeature_path, metalabel_path, outfile_path):
    metadataset = {"dataset_name": [], "label": []}
    # get metafeatures
    with open(metafeature_path, "r") as f:
        for line in f:
            metafeatures_instance = json.loads(line)
            dataset_name = metafeatures_instance["dataset"]
            metadataset["dataset_name"].append(dataset_name)
            for key, value in metafeatures_instance["metadata"].items():
                if key not in metadataset:
                    metadataset[key] = []
                metadataset[key].append(value)
            metadataset["label"].append(None)
    # get metalabels
    with open(metalabel_path, "r") as f:
        attributes = f.readline().strip().split(",")
        for line in f:
            parsed_line = line.strip().split(",")
            dataset_name = parsed_line[0]
            label = parsed_line[1]
            dataset_index = metadataset["dataset_name"].index(dataset_name)
            metadataset["label"][dataset_index] = label
    # remove datasets without labels
    metadataset_attributes = ["dataset_name"]
    for key in metadataset.keys():
        if key not in ["dataset_name", "label"]:
            metadataset_attributes.append(key)
    metadataset_attributes.append("label")
    metadataset_values = np.array([metadataset[key] for key in metadataset_attributes]).T
    metadataset_values = metadataset_values[metadataset_values[:,-1] != None]
    # write to outfile
    join = lambda arr: ",".join([str(item) for item in arr])
    with open(outfile_path, "w") as f:
        f.write(join(metadataset_attributes)+"\n")
        for row in metadataset_values:
            f.write(join(row)+"\n")

def get_trained_metamodel(metadata_path):
    # load training data
    metadataset = np.loadtxt(metadata_path, dtype=str, delimiter=",")
    attributes = metadataset[0]
    values = metadataset[1:]
    features = values[:,1:-1].astype(float)
    labels = values[:,-1]

    tree = DecisionTreeClassifier()
    cv_score = get_score(tree, features, labels)
    tree.fit(features, labels)
    return tree, attributes[1:-1], set(labels), cv_score

def load_d3m_dataset(feature_path, label_path):
    feature_dataset = np.loadtxt(feature_path, dtype=str, delimiter=",")
    feature_attributes = feature_dataset[0, 1:]
    features = feature_dataset[1:, 1:]

    label_dataset = np.loadtxt(label_path, dtype=str, delimiter=",")
    label_attributes = label_dataset[0, 1:]
    labels = label_dataset[1:, 1:]

    return features, labels

def pipeline(feature_path, label_path):
    pass

if __name__ == "__main__":
    metafeature_path = "test/metafeatures.jsonl"
    metalabel_path = "test/tree_vs_knn_output.csv"
    metadata_path = "test/metadata.csv"
    dataset_paths = get_dataset_paths()
    print(compute_metafeatures(dataset_paths, metafeature_path))
    #compute_metalabels(dataset_paths, metalabel_path, "test/tree_vs_knn_error.log")
    # combine_metafeatures_and_metalabels(metafeature_path, metalabel_path, metadata_path)
    # metamodel, attributes, label_set, cv_score = get_trained_metamodel(metadata_path)
    #d3m_test_feature_path = "/home/bschoenfeld/Research/datasets/d3m/o_38/data/trainData.csv"
    #d3m_test_label_path = "/home/bschoenfeld/Research/datasets/d3m/o_38/data/trainTargets.csv"
    #features, labels = load_d3m_dataset(d3m_test_feature_path, d3m_test_label_path)
    #print(features, labels)

