from __future__ import absolute_import
from graph_tool import spectral as gt_spectral
from graph_tool import collection as gt_collection
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs, spsolve
import math
import matplotlib.pyplot as plt
import timeit
import comdet.community_detection.network_utils as nu
import comdet.data.marvel as marvel


class PersonalizedPageRank:

    def __init__(self, graph, k, weight=None):

        self.lap = gt_spectral.laplacian(graph, normalized=True, weight=weight)
        self.adj = gt_spectral.adjacency(graph, weight=weight)

        self.weight = weight
        self.graph = graph
        self.n = graph.num_vertices()

        deg_vec = np.squeeze(np.asarray(self.adj.sum(axis=1)))
        self.deg = diags([deg_vec], [0], shape=(self.n, self.n))
        if k < graph.num_vertices():
            self.fast_inversion = True
            self.deg_inv_sqrt = diags([np.power(deg_vec, -0.5)], [0],
                                      shape=(self.n, self.n))
            self.eig_val, self.eig_vec = eigs(self.lap, k, which='SM')
            self.eig_val = np.real(self.eig_val)
            self.eig_vec = np.real(self.eig_vec)
            plt.figure()
            plt.plot(self.eig_val)
        else:
            self.fast_inversion = False

    def vector(self, seeds, gamma):
        vol_graph = nu.return_set_sum_edge_weights(self.graph, self.weight)
        vol_seeds = nu.volume(seeds, self.weight)
        vol_complement_seeds = vol_graph - vol_seeds

        temp = math.sqrt(vol_seeds * vol_complement_seeds / vol_graph)
        seed_vector = np.ones((self.graph.num_vertices(), 1))
        seed_vector *= temp / vol_complement_seeds
        for s in seeds:
            idx = self.graph.vertex_index[s]
            seed_vector[idx] = temp / vol_seeds

        ds = np.squeeze(self.deg.dot(seed_vector))

        if self.fast_inversion:
            dv = self.deg_inv_sqrt.dot(self.eig_vec)
            weights = dv.T.dot(ds) / (self.eig_val - gamma)
            x = dv.dot(weights)
        else:
            x = spsolve(self.lap - gamma * self.deg, ds)
        x[x < 0] = 0
        x /= np.sum(x)
        return x


def test():
    graph = marvel.characters_network()
    # graph = gt_collection.data['football']

    t = timeit.default_timer()
    ppr = PersonalizedPageRank(graph, 6445)
    print timeit.default_timer() - t
    t = timeit.default_timer()
    x1 = ppr.vector([graph.vertex(1)], -1e-1)
    print timeit.default_timer() - t

    t = timeit.default_timer()
    ppr = PersonalizedPageRank(graph, 200)
    print timeit.default_timer() - t
    t = timeit.default_timer()
    x2 = ppr.vector([graph.vertex(1)], -1e-1)
    print timeit.default_timer() - t

    plt.figure()
    plt.hold(True)
    plt.plot(x1, 'r')
    plt.plot(x2, 'b')

if __name__ == '__main__':
    # import cProfile
    # cProfile.run('test()', sort='cumtime')
    test()
    plt.show()

