from __future__ import absolute_import
import graph_tool as gt
import string
import numpy as np
from scipy.sparse import coo_matrix as sparse
from scipy.sparse import find, eye, triu


def create():
    filename = '/Users/mariano/Documents/datasets/marvel social network/' \
               'labeled_edges.tsv'

    characters = set()
    comics = set()
    edges = []
    with open(filename, 'r') as f:
        for line in f:
            edge = string.split(line, '\t')
            edge = (edge[0][1:-1], edge[1][1:-2])
            characters.add(edge[0])
            comics.add(edge[1])
            edges.append(edge)

    characters = list(characters)
    comics = list(comics)
    m = len(characters)
    n = len(comics)

    order_characters = range(m)
    dict_characters = dict(zip(characters, order_characters))

    comic_titles = [c.split()[0] for c in comics]
    order_comics = [x[0] + m for x in sorted(enumerate(comic_titles),
                                             key=lambda e: e[1])]
    comics = [comics[i - m] for i in order_comics]
    dict_comics = dict(zip(comics, order_comics))

    edges = map(lambda e: (dict_characters[e[0]], dict_comics[e[1]]), edges)

    graph = gt.Graph(directed=False)
    graph.add_edge_list(edges)

    n_edges = len(edges)
    i = map(lambda e: e[0], edges)
    j = map(lambda e: e[1] - m, edges)
    adjacency = sparse((np.ones((n_edges,)), (i, j)), (m, n))

    return sparse(adjacency), characters, comics


def characters_network():
    bipartite_adj, characters, comics = create()

    adj = bipartite_adj.dot(bipartite_adj.T)
    adj = triu(adj)
    row_idx, col_idx, _ = find(adj)
    edges = zip(row_idx, col_idx)

    graph = gt.Graph(directed=False)
    graph.add_edge_list(edges)
    return graph, characters

