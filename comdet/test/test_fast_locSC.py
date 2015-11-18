from __future__ import absolute_import
from graph_tool import collection as gt_collection
import matplotlib.pyplot as plt
import timeit
import comdet.network.ppr as lsc
import comdet.data.marvel as marvel


def test():
    graph = marvel.characters_network()
    # graph = gt_collection.data['football']

    t = timeit.default_timer()
    ppr = lsc.PersonalizedPageRank(graph, 6445)
    print timeit.default_timer() - t
    t = timeit.default_timer()
    x1 = ppr.vector([graph.vertex(1)], -1e-1)
    print timeit.default_timer() - t

    t = timeit.default_timer()
    ppr = lsc.PersonalizedPageRank(graph, 200)
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

