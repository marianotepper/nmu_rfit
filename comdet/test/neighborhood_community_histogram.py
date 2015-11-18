from __future__ import absolute_import, print_function
import matplotlib.pyplot as plt
import numpy as np
from graph_tool import stats as gt_stats
import pickle
import comdet.data.marvel as marvel
import comdet.community_detection.neighborhood_communities as nc


def test_loc_min_neighborhood_communities(name, graph, weight=None):
    print(graph.num_vertices(), graph.num_edges())
    out_file_name = name + '_histogram'

    seeds = nc.locally_minimal_neighborhood_communities(graph,
                                                        min_degree=1,
                                                        weight=weight)

    degrees = map(lambda x: x.out_degree(), seeds)
    with open(out_file_name + '.pickle', 'w') as out_file:
        pickle.dump(degrees, out_file)
    with open(out_file_name + '.pickle', 'r') as in_file:
        degrees = pickle.load(in_file)

    print(filter(lambda x: x >= 200, degrees))
    degrees = map(lambda x: (x >= 200) * 200 + (x < 200) * x, degrees)

    bins = range(201)
    hist = np.histogram(degrees, bins=bins)[0]
    hist = hist.astype(float)
    hist /= graph.num_vertices()
    hist = np.flipud(np.cumsum(np.flipud(hist)))
    print(hist)

    plt.figure()
    plt.plot([6, 6], [0, hist[6]], color='k', linestyle=':')
    plt.plot([0, 6], [hist[6], hist[6]], color='k', linestyle=':')
    plt.text(6 + 1, hist[6], '({0:d}, {1:1.3f})'.format(6, hist[6]),
             {'fontsize': 16})
    plt.plot([10, 10], [0, hist[10]], color='k', linestyle=':')
    plt.plot([0, 10], [hist[10], hist[10]], color='k', linestyle=':')
    plt.text(10 + 1, hist[10], '({0:d}, {1:1.3f})'.format(10, hist[10]),
             {'fontsize': 16})
    plt.plot([20, 20], [0, hist[20]], color='k', linestyle=':')
    plt.plot([0, 20], [hist[20], hist[20]], color='k', linestyle=':')
    plt.text(20 + 1, hist[20], '({0:d}, {1:1.3f})'.format(20, hist[20]),
             {'fontsize': 16})
    plt.plot(hist, linewidth=2)

    plt.xlabel('Local neighborhood size', fontsize=16)

    ax = plt.axes()
    # locs, _ = plt.xticks()
    # locs = np.sort(np.append(locs, [6, 10, 20]))
    # plt.xticks(locs)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    # locs = np.arange(0, 0.041, 0.01)
    # locs = np.sort(np.append(locs, [hist[6], hist[10], hist[20]]))
    # plt.yticks(locs)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)

    plt.savefig(out_file_name + '.pdf', transparent=True)


def main():
    name = 'marvel'
    graph = marvel.characters_network()
    gt_stats.remove_self_loops(graph)
    weight = graph.new_edge_property('float', vals=1)
    test_loc_min_neighborhood_communities(name, graph, weight=weight)


if __name__ == '__main__':
    main()
    plt.show()