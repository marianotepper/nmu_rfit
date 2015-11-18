from scipy.sparse import hstack, rand
import numpy as np
import timeit
import matplotlib.pyplot as plt
from comdet.biclustering.utils import sparse
import comdet.biclustering.nmf as bc
import comdet.biclustering.deflation as deflation


def binary_matrix_generator2(m, r, step=50, seed=0):
    np.random.seed(seed)
    x = (rand(m, r, density=0.1) > 0).astype(np.float)
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
    m = 400
    r = 4
    array = binary_matrix_generator2(m, r, step=50)

    t = timeit.default_timer()
    compression_level = 128
    online_deflator = deflation.L1CompressedDeflator(array, compression_level)
    bc.bicluster(online_deflator, n=r)
    print timeit.default_timer() - t

    # plt.figure()
    # plt.subplot(121)
    plt.matshow(array.toarray())
    # plt.subplot(122)
    # plt.spy(rows[0].dot(cols[1]))

if __name__ == '__main__':
    # import cProfile
    # cProfile.run('test()', sort='cumtime')
    test()
    plt.show()