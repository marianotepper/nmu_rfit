from networkx import Graph, complement
from networkx.algorithms.clique import find_cliques


def maximal_independent_sets(mat):
    g = Graph(mat)
    for u in g.nodes():
        if g.has_edge(u, u):
            g.remove_edge(u, u)
    cg = complement(g)
    isets = list(find_cliques(cg))
    return isets
