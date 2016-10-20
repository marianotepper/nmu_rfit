from networkx import Graph, connected_components
from networkx.algorithms.approximation.clique import clique_removal


def max_independent_set(mat):
    iset = []
    g = Graph(mat)
    for u in g.nodes():
        if g.has_edge(u, u):
            g.remove_edge(u, u)
    for comp in connected_components(g):
        sg = g.subgraph(comp)
        max_iset, isets = clique_removal(sg)
        iset.extend(max_iset)
    return iset
