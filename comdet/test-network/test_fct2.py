import numpy as np
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import timeit
from scipy.sparse import coo_matrix as sparse

import comdet.data.marvel as marvel
import comdet.biclustering.compression as comp
# import comdet.snmf.snpa as snpa


def binary_matrix_generator(m, seed=0):
    np.random.seed(0)
    r = 10
    step = 50
    x = (np.abs(np.random.rand(m, r)) > 0.6).astype(np.float)
    mat = np.hstack((np.atleast_2d(x[:, i]).T for i in range(x.shape[1])
                     for _ in range(step)))

    np.random.seed(seed)
    noise = np.random.uniform(size=mat.shape)
    mask = np.abs(noise) > 0.97
    for i in range(0, r*step, step):
        mask[:, i:i+1] = False
    mat[mask] = 1 - mat[mask]
    return sparse(mat)#, noiseless_mat, r


def test_artificial():
    plt.switch_backend('TkAgg')  # otherwise, monospace fonts do not work in mac
    m = 250
    mat = binary_matrix_generator(m)

    samples = 4
    mat_comp1, mask1 = comp.compression(mat, samples)
    # mat_comp2, mask2 = comp.compression(mat.T, samples)
    mat_comp2, mask2 = comp.compression(mat.T, samples)

    print mask2

    cmap = sns.cubehelix_palette(256, start=2.5, rot=0, dark=0.15, light=1,
                                 reverse=False, as_cmap=True)

    plt.figure()
    plt.subplot(131)
    plt.imshow(mat.toarray(), interpolation='none', cmap=cmap)
    plt.subplot(132)
    plt.imshow(mat_comp1.toarray(), interpolation='none', cmap=cmap)
    plt.subplot(133)
    plt.imshow(mat_comp2.T.toarray(), interpolation='none', cmap=cmap)

    plt.figure()
    plt.imshow(mat_comp1.toarray(), interpolation='none', cmap=cmap)

    plt.figure()
    plt.imshow(mat_comp1.T.dot(mat_comp1).toarray(), interpolation='none', cmap=cmap)


def test_marvel0():
    mat, characters, comics = marvel.create()
    samples = 128
    comp.compression(mat, samples)


def test_marvel():
    plt.switch_backend('TkAgg')  # otherwise, monospace fonts do not work in mac

    mat, characters, comics = marvel.create()

    samples = 64
    t = timeit.default_timer()
    mat_comp1, selection1 = comp.compression(mat, samples)
    time1 = timeit.default_timer() - t
    t = timeit.default_timer()
    mat_comp1, selection12 = comp.compression(mat, samples)
    time2 = timeit.default_timer() - t
    print 'times', time1, time2

    selected_characters = [(i, characters[s]) for i, s in enumerate(selection1)]
    selected_characters2 = [characters[i] for i in selection12]
    print selected_characters
    print 'intersection', len(set(selected_characters).intersection(set(selected_characters2)))
    print 'nnz', selection1.size
    print 1.0 * len(set(selected_characters).intersection(set(selected_characters2))) / selection1.size

    t = timeit.default_timer()
    mat_comp2, selection2 = comp.compression(mat.T, samples)
    time1 = timeit.default_timer() - t
    print 'time:', time1
    # t = timeit.default_timer()
    # mat_comp2, selection22 = comp.compression(mat.T, samples)
    # time2 = timeit.default_timer() - t
    # print 'time:', time2

    selected_comics = [comics[i] for i in selection2]
    selected_comics2 = [comics[i].split()[0] for i in selection2]
    print selected_comics
    print selected_comics2
    # print 'intersection', len(set(selected_comics).intersection(set(selected_comics2)))
    # print len(set(selected_comics)), len(set(selected_comics2))
    # print 'nnz', selection2.size
    # print 1.0 * len(set(selected_comics).intersection(set(selected_comics2))) / max(len(set(selected_comics)), len(set(selected_comics2)))


    cmap = sns.cubehelix_palette(256, start=2.5, rot=0, dark=0.15, light=1,
                                 reverse=False, as_cmap=True)

    # mat_comp1 = mat_comp1[:, np.sum(mat_comp1.toarray(), axis=0) != 0]
    # mat_comp2 = mat_comp2[:, np.sum(mat_comp2.toarray(), axis=0) != 0]

    # plt.figure()
    # plt.subplot(121)
    # plt.plot(np.sum(mat_comp1.toarray(), axis=0))
    # plt.subplot(122)
    # plt.plot(np.sum(mat_comp2.toarray(), axis=0))

    plt.figure()
    # plt.subplot(121)
    # plt.imshow(mat.toarray(), interpolation='none', cmap=cmap)
    plt.subplot(121)
    plt.imshow(mat_comp1.toarray(), interpolation='none', cmap=cmap)
    plt.axis('tight')
    plt.subplot(122)
    plt.imshow(mat_comp2.toarray(), interpolation='none', cmap=cmap)
    plt.axis('tight')

    # cols, weights, cl = snpa.snmf(mat_full, ncols=-100, normalize=False,
    #                                         selection='mt', method='snpa',
    #                                        func=snpa.robust_loss, pool_size=0)
    #
    # plt.figure()
    # plt.plot(cl)
    #
    # plt.figure()
    # plt.imshow(weights, interpolation='none', cmap=cmap)
    #
    # print [(i, selected_comics[c], np.sum(weights[i, :])) for i, c in enumerate(cols)]

if __name__ == '__main__':
    # test_artificial()
    # test_marvel()
    import cProfile
    cProfile.run('test_marvel0()', sort='cumtime')

    plt.show()


