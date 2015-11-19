from __future__ import absolute_import, print_function
from graph_tool import collection as gt_collection
from scipy.sparse import coo_matrix as sparse
from scipy.sparse import hstack
from scipy.io import savemat
import matplotlib.pyplot as plt
import timeit
import numpy as np
import comdet.network.neighborhood_communities as nc
import comdet.network.ppr as lsc
import comdet.data.marvel as marvel


def add_col(preference_matrix, comm, value=1):
    col_shape = (preference_matrix.shape[0], 1)
    data_shape = (len(comm),)
    col_idx = [int(v) for v in comm]
    column = sparse((np.ones(data_shape), (col_idx, np.zeros(data_shape))),
                    col_shape)
    if preference_matrix.shape[1] > 0:
        preference_matrix = hstack([preference_matrix, column])
    else:
        preference_matrix = column
    return preference_matrix


def test_loc_min_neighborhood_communities(name, graph, min_degree, alpha,
                                          cluster_size, weight=None):
    t = timeit.default_timer()
    seeds = nc.locally_minimal_neighborhood_communities(graph,
                                                        min_degree=min_degree)
    neigh_communities = []
    for i, s in enumerate(seeds):
        l = nc.neighborhood_community(s)
        neigh_communities.append(l)
    time_neigh_communities = timeit.default_timer() - t
    print(time_neigh_communities)

    if not seeds:
        return

    preference_matrix = sparse((graph.num_vertices(), 0))
    t = timeit.default_timer()
    k = 0
    for i, neigh_comm in enumerate(neigh_communities):
        print(i, len(neigh_communities))
        for u in neigh_comm:
            comm, _ = lsc.pagerank_nibble(graph, alpha, cluster_size, u, weight)
            preference_matrix = add_col(preference_matrix, comm, value=1)
            k += 1

        comm, _ = lsc.pagerank_nibble(graph, alpha, cluster_size, seeds[i],
                                      weight)
        preference_matrix = add_col(preference_matrix, comm, value=2)
        k += 1

        comm, _ = lsc.pagerank_nibble(graph, alpha, cluster_size, neigh_comm,
                                      weight)
        preference_matrix = add_col(preference_matrix, comm, value=3)
        k += 1

    time_loc_clustering = timeit.default_timer() - t
    print(time_loc_clustering, len(seeds))

    mat_dict = {'name': name,
                'min_degree': min_degree,
                'alpha': alpha,
                'cluster_size': cluster_size,
                'seeds': seeds,
                'neigh_communities': neigh_communities,
                'preference_matrix': preference_matrix,
                'time_neigh_communities': time_neigh_communities,
                'time_loc_clustering': time_loc_clustering}
    savemat(name + '_stage1_alpha({0})_cs({1}).mat'.format(alpha, cluster_size),
            mat_dict)


def main():
    # name = 'karate_club'
    # graph = nx.karate_club_graph()
    # print(graph.number_of_nodes(), graph.number_of_edges())
    # alpha = 1e-3
    # cluster_size = 7e1
    # test_loc_min_neighborhood_communities(name, graph, 2, alpha, cluster_size)
    # cluster_size = 2e2
    # test_loc_min_neighborhood_communities(name, graph, 2, alpha, cluster_size)

    # name = 'dolphins'
    # graph = nx.read_gml('../data/dolphins.gml')
    # print(graph.number_of_nodes(), graph.number_of_edges())
    # alpha = 1e-3
    # cluster_size = 2e2
    # test_loc_min_neighborhood_communities(name, graph, 2, alpha, cluster_size)
    # cluster_size = 1e1
    # test_loc_min_neighborhood_communities(name, graph, 2, alpha, cluster_size)

    name = 'football'
    graph = gt_collection.data['football']
    print(graph.num_vertices(), graph.num_edges())
    alpha = 1e-3
    cluster_size = 2e2
    test_loc_min_neighborhood_communities(name, graph, 2, alpha, cluster_size)
    cluster_size = 3e2
    test_loc_min_neighborhood_communities(name, graph, 2, alpha, cluster_size)

    # name = 'roget_thesaurus'
    # graph = nx.read_pajek('../data/roget.net')
    # test_loc_min_neighborhood_communities(graph, 2, alpha, cluster_size)

    # name = 'amazon'
    # graph = nx.read_adjlist('../data/com-amazon.ungraph.txt')
    # test_loc_min_neighborhood_communities(graph, 10)

    # name = 'marvel'
    # graph = marvel.characters_network()
    # test_loc_min_neighborhood_communities(graph, 2)


if __name__ == '__main__':
    plt.switch_backend('TkAgg')
    main()

    plt.show()

    # import profile
    # profile.run('main()', sort='cumtime')