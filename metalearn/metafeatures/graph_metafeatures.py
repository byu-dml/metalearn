import numpy as np
from heapq import heapify, heappop, heappush
import networkx as nx
from networkx.algorithms.bipartite import clustering
from sklearn.neighbors import radius_neighbors_graph

from .common_operations import profile_distribution


class Graph:

    def __init__(self, adj):
        self.adj = adj
        self.nodes = [i for i in range(len(self.adj))]

    def get_all_shortest_path_lengths(self):
        lengths = self._floyd_warshall()
        return lengths

    def _floyd_warshall(self):
        dists = np.full((len(self.adj), len(self.adj)), fill_value=np.inf)
        np.fill_diagonal(dists, val=0)
        dists[np.nonzero(self.adj)] = self.adj[np.nonzero(self.adj)]
        V = self.nodes

        for k in V:
            for i in V:
                for j in V:
                    new_dist = dists[i, k] + dists[k, j]
                    if dists[i, j] > new_dist:
                        dists[i, j] = new_dist

        return dists


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
