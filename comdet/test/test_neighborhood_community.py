from __future__ import absolute_import, print_function
from scipy.sparse import coo_matrix as sparse
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import timeit
import numpy as np
from graph_tool import stats as gt_stats, collection as gt_collection
import comdet.network.neighborhood_communities as nc
import comdet.network.ppr as lsc
import comdet.network.plot as comdetplot
import comdet.data.marvel as marvel



def add_col(preference_matrix, col_idx, value=1):
    col_idx = [int(v) for v in col_idx]
    col_shape = (preference_matrix.shape[0], 1)
    data_shape = (len(col_idx),)
    column = sparse((value * np.ones(data_shape),
                     (col_idx, np.zeros(data_shape))),
                    col_shape)
    if preference_matrix.shape[1] > 0:
        preference_matrix = hstack([preference_matrix, column])
    else:
        preference_matrix = column
    return preference_matrix


def test_loc_min_neighborhood_communities(name, graph, min_degree, alpha,
                                          cluster_size, weight=None):
    print(graph.num_vertices(), graph.num_edges())

    t = timeit.default_timer()
    seeds = nc.locally_minimal_neighborhood_communities(graph,
                                                        min_degree=min_degree,
                                                        weight=weight)
    t = timeit.default_timer() - t
    print(t)

    t = timeit.default_timer()
    neigh_communities = []
    for i, s in enumerate(seeds):
        l = nc.neighborhood_community(s)
        neigh_communities.append(l)
    t = timeit.default_timer() - t
    print(t)

    if not seeds:
        return

    print(1.0 * len(seeds) / graph.num_vertices())

    preference_matrix = sparse((graph.num_vertices(), 0))
    seed_communities = []
    t = timeit.default_timer()
    k = 0
    for i, neigh_comm in enumerate(neigh_communities):

        for u in neigh_comm:
            comm, _ = lsc.pagerank_nibble(graph, alpha, cluster_size, u, weight)
            preference_matrix = add_col(preference_matrix, comm, value=1)
            k += 1

        comm, _ = lsc.pagerank_nibble(graph, alpha, cluster_size, seeds[i], weight)
        preference_matrix = add_col(preference_matrix, comm, value=2)
        seed_communities.append(comm)
        print(i, len(neigh_communities), len(comm))
        k += 1

        comm, _ = lsc.pagerank_nibble(graph, alpha, cluster_size, neigh_comm, weight)
        preference_matrix = add_col(preference_matrix, comm, value=3)
        k += 1

    t = timeit.default_timer() - t
    print(t, len(seeds))

    if name == 'football':
        vprop_communities = graph.new_vertex_property('int', vals=-1)
        for i, u in enumerate(seeds):
            vprop_communities[u] = i
        pos = comdetplot.create_layout_football(graph)
        comdetplot.draw(graph, vprop_communities, pos=pos,
                        output=name + '_seeds.pdf')
        out_file_name = name + '_seed_communities_alpha({0})_cs({1}).pdf'
        comdetplot.draw(graph, seed_communities, pos=pos,
                        output=out_file_name.format(alpha, cluster_size))

    labels = ['Vertex of a\nlocally minimal\nneighborhood\ncommuntity',
              'Seed vertex of a\nlocally minimal\nneighborhood\ncommuntity',
              'Locally minimal\nneighborhood\ncommuntity']
    title = 'Initialized using:'
    plt.figure(figsize=(8, 4))
    comdetplot.plot_preference_matrix(preference_matrix.toarray(),
                                      neigh_communities,
                                      offset=2, labels=labels, title=title)
    out_file_name = name + '_initialization_alpha({0})_cs({1}).pdf'
    plt.savefig(out_file_name.format(alpha, cluster_size), transparent=True)


def process_graph(name, min_degree=2, alpha=1e-3, cluster_size=1e2):
    if name == 'marvel':
        graph = marvel.characters_network()
    else:
        graph = gt_collection.data[name]

    if name == 'football':
        graph = gt_collection.data[name]

        communities = graph.vertex_properties['value_tsevans']
        pos = comdetplot.create_layout_football(graph)
        comdetplot.draw(graph, communities, pos=pos, output=name + '_gt.pdf')

    gt_stats.remove_self_loops(graph)
    weight = graph.new_edge_property('float', vals=1)
    test_loc_min_neighborhood_communities(name, graph, min_degree, alpha,
                                          cluster_size, weight=weight)

def main():
    data_dir = '/Users/mariano/PycharmProjects/comdet/comdet/data/'

    # name = 'karate'
    # process_graph(name, min_degree=2, alpha=1e-3, cluster_size=5e1)
    # process_graph(name, min_degree=2, alpha=1e-3, cluster_size=7e1)

    # name = 'amazon'
    # graph = nx.read_adjlist('../data/com-amazon.ungraph.txt')
    # test_loc_min_neighborhood_communities(graph, 10)

    name = 'football'
    process_graph(name, min_degree=2, alpha=1e-3, cluster_size=1.5e2)
    # process_graph(name, min_degree=2, alpha=1e-3, cluster_size=3e2)
    # process_graph(name, min_degree=2, alpha=1e-3, cluster_size=4e2)
    # process_graph(name, min_degree=2, alpha=1e-3, cluster_size=5e2)

    # name = 'dolphins'
    # graph = nx.read_gml('../data/dolphins.gml')
    # print(graph.number_of_nodes(), graph.number_of_edges())
    # alpha = 1e-3
    # # cluster_size = 2e2
    # cluster_size = 1e2
    # test_loc_min_neighborhood_communities(name, graph, 2, alpha, cluster_size)

    # name = 'celegans'
    # graph = nx.read_gml('../data/celegansneural.gml')
    # print(graph.number_of_nodes(), graph.number_of_edges())
    # alpha = 1e-3
    # cluster_size = 1e3
    # test_loc_min_neighborhood_communities(name, graph, 2, alpha, cluster_size)

    # name = 'marvel'
    # process_graph(name, 2, alpha=1e-3, cluster_size=1e2)
    # process_graph(name, 6, alpha=1e-3, cluster_size=5e2)


if __name__ == '__main__':
    # plt.switch_backend('TkAgg')
    main()
    plt.show()