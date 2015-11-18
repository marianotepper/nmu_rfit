import numpy as np
import comdet.biclustering.fct as fct
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')  # otherwise, monospace fonts do not work in mac
import seaborn.apionly as sns


def imagesc(mat, cmap_name, center_value=0):
    array = mat.toarray()
    values = np.unique(array)
    mid1 = np.nonzero(values == center_value)[0][0]
    mid2 = values.shape[0] - mid1 - 1
    palette1 = sns.color_palette(cmap_name, 2 * mid1 + 1)
    palette2 = sns.color_palette(cmap_name, 2 * mid2 + 1)
    palette = palette1[:mid1+1] + palette2[-mid2:]

    idx_img = np.zeros((array.shape[0], array.shape[1], 3))
    for k, v in enumerate(values):
        i, j = np.nonzero(array == v)
        idx_img[i, j, :] = palette[k]

    plt.imshow(idx_img, interpolation='none')

size = (4096, 4096)
s = 128
r = 128

h = fct.spread_matrix(size[0], s)
print 'Done H'
d = fct.cauchy(2*size[0], 2*s)
print 'Done D'
b = fct.basis((r, 2*size[0]))
print 'Done B'

# plt.figure()
# plt.subplot(131)
# imagesc(b, 'RdBu_r')
# plt.subplot(132)
# imagesc(d, 'RdBu_r')
# plt.subplot(133)
# imagesc(h, 'RdBu_r')

t = fct.fast_cauchy_transform(size[0], s, r)
print 1.0 * t.nnz / t.size
t = t.toarray()
print 1.0 * np.count_nonzero(t) / t.size

# plt.figure()
# imagesc(fast_cauchy_transform(size[0], s, r), 'RdBu_r')

plt.show()
