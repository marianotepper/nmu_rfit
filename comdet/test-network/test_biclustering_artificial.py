from scipy.sparse import hstack, rand
import numpy as np
import timeit
import matplotlib.pyplot as plt
import seaborn as sns
from comdet.biclustering.utils import sparse
import comdet.biclustering.nmf as bc
import comdet.biclustering.deflation as deflation


def binary_matrix_generator2(m, r, step=50):
    x = (rand(m, r, density=0.15) > 0).astype(np.float)
    y = sparse(np.ones((1, step)))
    sections = []
    for i in range(r):
        temp = x[:, i].dot(y)
        temp_lil = temp.tolil()
        noise_mask = rand(m, step, density=0.01) > 0
        temp_lil[noise_mask] = 1 - temp[noise_mask]
        sections.append(temp_lil.tocsc())
    mat = hstack(sections)
    return sparse(mat)


def test():
    nrows = [500, 1000, 2000, 3000, 4000, 5000, 10000, 20000]
    r = 3
    trials = 20

    time_normal = np.zeros((len(nrows), trials))
    time_compressed_init = np.zeros((len(nrows), trials))
    time_compressed_run = np.zeros((len(nrows), trials))

    # np.random.seed(0)

    for i, m in enumerate(nrows):
        for j in range(trials):
            array = binary_matrix_generator2(m, r, step=50)

            t = timeit.default_timer()
            online_deflator = deflation.Deflator(array)
            bc.bicluster(online_deflator, n=r)
            t1 = timeit.default_timer() - t
            time_normal[i, j] = t1

            t = timeit.default_timer()
            compression_level = 128
            online_deflator = deflation.L1CompressedDeflator(array,
                                                             compression_level)
            t1 = timeit.default_timer() - t
            time_compressed_init[i, j] = t1

            t = timeit.default_timer()
            bc.bicluster(online_deflator, n=r)
            t2 = timeit.default_timer() - t
            time_compressed_run[i, j] = t2

    ind = np.arange(len(nrows))
    width = 0.35

    palette = sns.color_palette("Paired", n_colors=4)

    plt.figure()
    plt.bar(ind, time_normal.mean(axis=1), width, color=palette[1],
            yerr=np.std(time_normal, axis=1), ecolor='k',
            bottom=t1, label='Uncompressed')
    plt.bar(ind + width, time_compressed_init.mean(axis=1), width,
            color=palette[2], yerr=time_compressed_init.std(axis=1), ecolor='k',
            bottom=t1, label='Compressed - initialization')
    plt.bar(ind + width, time_compressed_run.mean(axis=1), width,
            color=palette[3], yerr=time_compressed_run.std(axis=1), ecolor='k',
            bottom=t1, label='Compressed - runtime')

    plt.xlabel('Number of rows')
    plt.ylabel('Speed (s)')
    plt.xticks(ind + width, nrows)
    plt.legend(loc='upper left')


if __name__ == '__main__':
    # import cProfile
    # cProfile.run('test-network()', sort='cumtime')
    test()
    plt.show()