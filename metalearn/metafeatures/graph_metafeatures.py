import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
from networkx.algorithms.bipartite import clustering
from sklearn.neighbors import radius_neighbors_graph

from .common_operations import profile_distribution


def get_radius_neighbors_graph(X, mode):
    adj = radius_neighbors_graph(X, radius=np.inf, mode='distance').toarray()
    radius = np.percentile(adj[np.triu_indices(len(adj), 1)], 15)
    if mode == 'weighted':
        adj = np.where(adj < radius, adj, 0)
    elif mode == 'unweighted':
        adj = np.where(adj < radius, 1, 0)
    else:
        raise NotImplementedError
    graph = nx.Graph(adj)
    return graph,


def get_number_of_edges(graph):
    edges = graph.number_of_edges()
    return edges,


def get_path_lengths(graph):
    lengths = []
    for source, paths in nx.all_pairs_dijkstra_path_length(graph):
        for target, length in paths.items():
            if length > 0:
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
