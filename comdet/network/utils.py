from __future__ import absolute_import, print_function

_sew_name = 'sum_edge_weights'


def return_set_sum_edge_weights(graph, weight=None):
    if _sew_name not in graph.graph_properties:
        if weight is None:
            sew = 2 * graph.num_edges()
        else:
            sew = 2 * sum([weight[e] for e in graph.edges()])
        graph.graph_properties[_sew_name] = graph.new_graph_property('float',
                                                                     val=sew)
    return graph.graph_properties[_sew_name]


def inner_volume(graph, bunch, weight):
    """
    Returns the sum of the costs (weights) of the edges between two
    sets of nodes.

    :param graph: the graph of interest
    :type graph: graph_tool graph
    :param bunch: A container of nodes.
    :type bunch: iterable container
    :param weight: Specifies the edge data key to use as weight. If the
    graph is unweighted, all edges have weight 1.
    :return: an nonnegative integer for unweighted graphs. For weighted
    graphs, its type depends on the type of the weighting key.
    """
    restricted = graph.new_vertex_property('bool', vals=False)
    for v in bunch:
        restricted[v] = True
    graph.set_vertex_filter(restricted)
    inner_vol = sum([weight[e] for e in graph.edges()])
    graph.set_vertex_filter(None)
    return inner_vol


def volume(bunch, weight):
    """
    Returns the sum of the costs (weights) of the outgoing edges from a
    set of nodes.

    :param bunch: A container of nodes.
    :type bunch: iterable container
    :return: an nonnegative integer for unweighted graphs. For weighted
    graphs, its type depends on the type of the weighting key.
    """
    sew = 0
    for v in bunch:
        sew += v.out_degree(weight=weight)
    return sew


def conductance(graph, bunch, weight):
    """
    Returns the cut size between two sets of nodes, divided by the
    minimum of the volumes of both sets.

    If c1 forms a singleton connected component or covers the full
    graph, 0 is returned.

    :param graph: the graph of interest
    :type graph: graph_tool graph
    :param bunch: A container of nodes.
    :type bunch: iterable container
    :return: a nonnegative integer for unweighted graphs. For weighted
    graphs, its type depends on the type of the weighting key.
    """
    sum_edge_weights = return_set_sum_edge_weights(graph, weight)
    vol_bunch = volume(bunch, weight)
    if vol_bunch == 0 or vol_bunch == sum_edge_weights:
        return 0
    cs = vol_bunch - inner_volume(graph, bunch, weight)
    vol_complement = sum_edge_weights - vol_bunch + cs
    return cs / float(min(vol_bunch, vol_complement))
