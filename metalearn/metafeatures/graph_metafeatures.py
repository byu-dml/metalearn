import numpy as np
import networkx as nx
from networkx.algorithms.bipartite import clustering
from sklearn.neighbors import radius_neighbors_graph, RadiusNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from .common_operations import profile_distribution



class HVDM:

    def __init__(self, x, y, column_types):
        self.attributes = []
        self.classes = np.unique(y)
        for i, attr in enumerate(x):
            self.attributes.append({'type': column_types[attr]})
            col = x[attr]
            if column_types[attr] == 'NUMERIC':
                self.attributes[i]['sd'] = np.std(col)
            elif column_types[attr] == 'CATEGORICAL':
                values, counts = np.unique(col, return_counts=True)
                for attr_value, count in zip(values, counts):
                    self.attributes[i][attr_value] = {'count': count}
                    for class_value in self.classes:
                        value_class_count = x[attr][(y == class_value) & (x[attr] == attr_value)].count()
                        self.attributes[i][attr_value][class_value] = {'count': value_class_count}

    def normalized_diff(self, a, b, attr):
        diff = np.abs(a - b) / (4 * self.attributes[attr]['sd'])
        return diff

    def normalized_vdm(self, a, b, attr):
        total = 0
        for value in self.classes:
            v1 = self.attributes[attr][a][value]['count'] / self.attributes[attr][a]['count']
            v2 = self.attributes[attr][b][value]['count'] / self.attributes[attr][b]['count']
            total += np.abs(v1 - v2) ** 2
        return np.sqrt(total)

        # return np.sqrt(sum(
        #     [
        #         np.abs((self.attributes[attr][a][value]['count']/self.attributes[attr][a]['count']) -
        #                (self.attributes[attr][b][value]['count']/self.attributes[attr][b]['count'])) ** 2
        #         for value in self.classes
        #     ]
        # ))

    def get_hvdm(self, x, y):
        total = 0
        for i, (v1, v2) in enumerate(zip(x, y)):
            if self.attributes[i]['type'] == 'NUMERIC':
                total += self.normalized_diff(v1, v2, i) ** 2
            elif self.attributes[i]['type'] == 'CATEGORICAL':
                total += self.normalized_vdm(v1, v2, i)

        return np.sqrt(total)

        # return np.sqrt(sum(
        #     [
        #         (self.normalized_diff(v1, v2, i)
        #          if self.attributes[i]['type'] == 'NUMERIC'
        #          else self.normalized_vdm(v1, v2, i))**2
        #         for i, (v1, v2) in enumerate(zip(x, y))
        #      ]
        # ))


def get_radius_neighbors_graph(X, Y, column_types, mode):
    data = X
    le = LabelEncoder()
    for col, col_type in column_types.items():
        if col in data.columns and col_type != 'NUMERIC':
            data[col][data[col] != None] = le.fit_transform(data[col][data[col] != None])
    hvdm = HVDM(data, Y, column_types)
    graph = radius_neighbors_graph(data, radius=0.5, mode=mode, metric=hvdm.get_hvdm)
    return nx.from_scipy_sparse_matrix(graph),


def get_number_of_edges(graph):
    return graph.number_of_edges(),


def get_path_lengths(graph):
    length_dict = nx.all_pairs_dijkstra_path_length(graph)
    lengths = []
    for source, paths in length_dict.items():
        for target, length in paths.items():
            lengths.append(length)
    return profile_distribution(lengths)


def get_clusterings(graph):
    clusterings = [cc for cc in clustering(graph).values()]
    return profile_distribution(clusterings)


def get_degrees(graph):
    degrees = [degree for degree in graph.degree().values()]
    return profile_distribution(degrees)


def get_class_changes(graph):
    tree = nx.minimum_spanning_edges
    pass


# metric='pyfunc',metric_params={"func":myDistance}
