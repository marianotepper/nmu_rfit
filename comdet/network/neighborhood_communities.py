from __future__ import absolute_import
import comdet.network.utils as nu

_ncc = 'neighborhood_community_conductance'


def neighborhood_community(v):
    neigh_com = list(v.out_neighbours())
    neigh_com.append(v)
    return neigh_com


def neighborhood_community_conductance(graph, v, weight):
    vprop_ncc = graph.vertex_properties[_ncc]
    if vprop_ncc[v] == -1:
        vprop_ncc[v] = nu.conductance(graph, neighborhood_community(v), weight)
    return vprop_ncc[v]


def locally_minimal_neighborhood_communities(graph, min_degree=2, weight=None):
    if weight is None:
        weight = graph.new_edge_property('float', vals=1)
    vprop_ncc = graph.new_vertex_property('float', vals=-1)
    graph.vertex_properties[_ncc] = vprop_ncc

    loc_min_list = []
    for i, v in enumerate(graph.vertices()):
        d = v.out_degree()
        if d < min_degree:
            continue
        v_cond = neighborhood_community_conductance(graph, v, weight)
        if all(v_cond <= neighborhood_community_conductance(graph, w, weight)
               for w in v.out_neighbours()):
            loc_min_list.append(v)
    return loc_min_list
