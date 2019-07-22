import numpy as np
import networkx as nx
from networkx.algorithms.bipartite import clustering
from sklearn.neighbors import radius_neighbors_graph

from .common_operations import profile_distribution


def get_radius_neighbors_graph(X, mode):
    graph = radius_neighbors_graph(X, radius=0.5, mode=mode)
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


def get_class_changes(graph, Y):
    tree = nx.minimum_spanning_edges
    pass
