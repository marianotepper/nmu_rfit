from __future__ import absolute_import
from graph_tool import spectral as gt_spectral
from collections import defaultdict, deque
from operator import itemgetter
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs, spsolve
import math
import comdet.network.utils as nu


def approximate_page_rank(graph, alpha, epsilon, seeds, weight):
    pagerank = defaultdict(int)
    residual = defaultdict(int)
    candidates = deque(seeds)
    for s in seeds:
        residual[s] = 1.0 / len(seeds)

    def _push(u):
        deg_u = u.out_degree()
        push_val = residual[u] - 0.5 * epsilon * deg_u
        put_val = (1.0 - alpha) * push_val / deg_u
        pagerank[u] += alpha * push_val
        residual[u] = 0.5 * epsilon * deg_u
        for e in u.out_edges():
            v = e.target()
            deg_v = v.out_degree()
            put_val_uv = put_val * weight[e]
            old_residual = residual[v]
            residual[v] = old_residual + put_val_uv
            thresh = deg_v * epsilon
            if thresh - put_val_uv <= old_residual < thresh:
                candidates.append(v)

    n_1000 = 1000 * graph.num_vertices()
    k = 0
    while candidates:
        _push(candidates.pop())
        k += 1
        if k > n_1000:
            break
    return pagerank, residual


def support_sweep(graph, sorted_nodes, weight):
    edges2 = nu.return_set_sum_edge_weights(graph, weight=weight)
    volume = 0.
    cut = 0.
    conductance = []
    sorted_nodes_examined = set([])
    for i, u in enumerate(sorted_nodes):
        deg_u = u.out_degree(weight=weight)
        cut_size = deg_u
        for e in u.out_edges():
            v = e.target()
            if v in sorted_nodes_examined:
                cut_size -= 2 * weight[e]
        volume += deg_u
        cut += cut_size
        sorted_nodes_examined.add(u)
        if volume >= edges2:
            cond = 1.
        elif 2 * volume > edges2:
            cond = cut / (edges2 - volume)
        else:
            cond = cut / volume
        conductance.append(cond)
    return conductance


def pagerank_nibble(graph, alpha, cluster_size, seeds, weight):
    try:
        iter(seeds)
    except TypeError:
        seeds = [seeds]
    pagerank, residual = approximate_page_rank(graph, alpha, 1.0 / cluster_size,
                                               seeds, weight=weight)
    if not pagerank:
        return [], 0
    for u in pagerank:
        pagerank[u] /= u.out_degree(weight=weight)
    sorted_nodes = sorted(pagerank.keys(), key=lambda x: pagerank[x],
                          reverse=True)
    conductance = support_sweep(graph, sorted_nodes, weight=weight)
    best_cut_idx, min_conductance = min(enumerate(conductance),
                                        key=itemgetter(1))
    # plt.figure()
    # plt.plot(conductance)
    # plt.scatter([best_cut_idx], [min_conductance])

    # print(best_cut_idx, len(sorted_nodes))
    return sorted_nodes[:best_cut_idx + 1], min_conductance


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
