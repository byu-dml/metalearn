import numpy as np
import cProfile
import line_profiler
from itertools import product
import timeit


class Graph:

    def __init__(self, adj):
        self.adj = adj
        self.nodes = [i for i in range(len(self.adj))]

    def get_all_shortest_path_lengths(self):
        lengths = self._floyd_warshall()
        return lengths

    # @profile
    # def floyd_warshall_1(self):
    #     dists = np.full((len(self.adj), len(self.adj)), fill_value=np.inf)
    #     np.fill_diagonal(dists, val=0)
    #     dists[np.nonzero(self.adj)] = self.adj[np.nonzero(self.adj)]
    #     V = self.nodes
    #
    #     for k in V:
    #         for i in V:
    #             for j in V:
    #                 new_dist = dists.item((i, k)) + dists.item((k, j))
    #                 if dists.item((i, j)) > new_dist:
    #                     dists.itemset((i, j), new_dist)
    #
    #     return dists

    # @profile
    # def floyd_warshall_2(self):
    #     dists = np.full((len(self.adj), len(self.adj)), fill_value=np.inf)
    #     np.fill_diagonal(dists, val=0)
    #     dists[np.nonzero(self.adj)] = self.adj[np.nonzero(self.adj)]
    #     V = self.nodes
    #
    #     for k, i, j in product(V, V, V):
    #         new_dist = dists[i, k] + dists[k, j]
    #         if dists[i, j] > new_dist:
    #             dists[i, j] = new_dist
    #
    #     return dists

    # @profile
    # def floyd_warshall_3(self):
    #     dists = np.full((len(self.adj), len(self.adj)), fill_value=np.inf)
    #     np.fill_diagonal(dists, val=0)
    #     dists[np.nonzero(self.adj)] = self.adj[np.nonzero(self.adj)]
    #     V = self.nodes
    #
    #     for k, i, j in product(V, V, V):
    #         new_dist = dists.item((i, k)) + dists.item((k, j))
    #         if dists.item((i, j)) > new_dist:
    #             dists.itemset((i, j), new_dist)
    #
    #     return dists

    # @profile
    # def floyd_warshall_4(self):
    #     dists = np.full((len(self.adj), len(self.adj)), fill_value=np.inf)
    #     np.fill_diagonal(dists, val=0)
    #     dists[np.nonzero(self.adj)] = self.adj[np.nonzero(self.adj)]
    #     V = self.nodes
    #
    #     for k in V:
    #         for i in V:
    #             for j in V[::2]:
    #
    #                 new_dist1 = dists.item((i, k)) + dists.item((k, j))
    #                 if dists.item((i, j)) > new_dist1:
    #                     dists.itemset((i, j), new_dist1)
    #
    #                 new_dist2 = dists.item((i, k)) + dists.item((k, j+1))
    #                 if dists.item((i, j+1)) > new_dist2:
    #                     dists.itemset((i, j+1), new_dist2)
    #
    #     return dists

    # @profile
    # def floyd_warshall_5(self):
    #     dists = np.full((len(self.adj), len(self.adj)), fill_value=np.inf)
    #     np.fill_diagonal(dists, val=0)
    #     dists[np.nonzero(self.adj)] = self.adj[np.nonzero(self.adj)]
    #     V = self.nodes
    #
    #     for k in V:
    #         for i in V:
    #             for j in V:
    #                 old_dist = dists.item((i, j))
    #                 new_dist = dists.item((i, k)) + dists.item((k, j))
    #                 dists.itemset((i, j), new_dist if old_dist > new_dist else old_dist)
    #
    #     return dists

    # @profile
    def floyd_warshall_old(self):
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

    # @profile
    def floyd_warshall(self):
        dists = np.full((len(self.adj), len(self.adj)), fill_value=np.inf)
        np.fill_diagonal(dists, val=0)
        dists[np.nonzero(self.adj)] = self.adj[np.nonzero(self.adj)]
        V = self.nodes

        for k in V:
            new_dists = np.add.outer(dists[:, k], dists[k])
            np.copyto(dists, new_dists, where=(dists > new_dists))

        return dists

    def bellman_ford(self):
        dists = np.array([self._bellman_ford(n) for n in self.nodes])
        return dists

    def _bellman_ford(self, source):
        dist = [np.inf for v in self.nodes]
        dist[source] = 0

        changed = False
        for v in self.nodes:
            for u in self.nodes:
                if self.adj[v, u] > 0:
                    if dist[v] > dist[u] + self.adj[v, u]:
                        dist[v] = dist[u] + self.adj[v, u]
                        changed = True
            if not changed:
                break

        return dist


def make_graph(n=100):
    np.random.seed(0)
    x = np.random.choice([0, 1], size=(n, n), p=[.85, .15])
    x[np.tril_indices(n)] = x[np.triu_indices(n)]
    x[np.diag_indices(n)] = 0
    return Graph(x)


if __name__ == "__main__":
    sets = 1
    repetitions = 1
    size = 15
    g = make_graph(size)
    print("FW run time:", end=' ')
    fw_timer = timeit.Timer('g.floyd_warshall()', globals=globals())
    fw_time = min(fw_timer.repeat(sets, repetitions)) / repetitions
    print(f"{fw_time}")

    print(f"BF run time:", end=' ')
    bf_timer = timeit.Timer('g.bellman_ford()', globals=globals())
    bf_time = min(bf_timer.repeat(sets, repetitions)) / repetitions
    print(f"{bf_time}")

    print(f'FW is {(bf_time/fw_time):.3f} times faster than BF')

    fw = g.floyd_warshall_old()
    bf = g.bellman_ford()

    if np.all(fw == bf):
        print('outputs are THE SAME')
    else:
        print('outputs are NOT the same:')
        print('FW:')
        print(fw)
        print('BF')
        print(bf)