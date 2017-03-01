from __future__ import absolute_import, print_function
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import seaborn.apionly as sns
import timeit
import rnmu.nmu as nmu

dir_name = '../results/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
dir_name += 'digits/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

imgs = [Image.open('./digits/digit2.png'),
        Image.open('./digits/digit3.png'),
        Image.open('./digits/digit5.png'),
        Image.open('./digits/digit6.png'),
        Image.open('./digits/digit8.png')
        ]
imgs = [np.array(im.convert('L'), dtype=np.float) / 255. for im in imgs]
img_size = imgs[0].shape

mat = 1 - np.stack([im.flatten() for im in imgs], axis=1)

t = timeit.default_timer()
factors = nmu.recursive_nmu(mat, r=10, init='svd', refine_v=True)
t = timeit.default_timer() - t
print('time {:.2f}'.format(t))

recs = sum(u.dot(v) for u, v in factors)

with sns.axes_style("whitegrid"):
    plt.figure()
    for i, im in enumerate(imgs):
        plt.subplot(1, len(imgs), i+1)
        plt.imshow(im, interpolation='nearest', cmap='gray')
        plt.tick_params(
            axis='both',
            which='both',
            bottom='off',
            top='off',
            labelbottom='off',
            left='off',
            right='off',
            labelleft='off')
        plt.grid(b=False)
    plt.tight_layout()
    plt.savefig(dir_name + 'digits_original.pdf',
                dpi=150, bbox_inches='tight')

    plt.figure()
    for i, (u, v) in enumerate(factors):
        plt.subplot(1, len(factors), i + 1)
        plt.imshow(1 - u.reshape(img_size), interpolation='nearest',
                   cmap='gray')
        plt.tick_params(
            axis='both',
            which='both',
            bottom='off',
            top='off',
            labelbottom='off',
            left='off',
            right='off',
            labelleft='off')
        plt.grid(b=False)
        plt.title('{}'.format(i + 1))
    plt.tight_layout()
    plt.savefig(dir_name + 'digits_left_factors.pdf',
                dpi=150, bbox_inches='tight')

    plt.figure()
    for i in range(len(imgs)):
        plt.subplot2grid((1, len(imgs)), (0, i))
        plt.imshow(1 - recs[:, i].reshape(img_size), vmin=0, vmax=1,
                   interpolation='nearest', cmap='gray')
        plt.tick_params(
            axis='both',
            which='both',
            bottom='off',
            top='off',
            labelbottom='off',
            left='off',
            right='off',
            labelleft='off')
        plt.grid(b=False)

    plt.tight_layout()
    plt.savefig(dir_name + 'digits_reconstruction.pdf',
                dpi=300, bbox_inches='tight')

    plt.figure(figsize=(8, 2.5))
    for i in range(len(imgs)):
        plt.subplot2grid((1, len(imgs)), (0, i))
        x_vals = [v[0, i] for _, v in factors]
        y_vals = np.arange(1, len(factors) + 1)
        plt.hlines(y_vals, 0, x_vals, color='#e41a1c', linewidth=4)
        plt.scatter(x_vals, y_vals, color='#e41a1c', marker='o', linewidth=4)
        plt.xlim(-0.1, 1.1)
        plt.ylim(0.5, len(factors) + 0.5)
        plt.xticks([0, 0.5, 1])

    plt.tight_layout()
    plt.savefig(dir_name + 'digits_right_factors.pdf',
                dpi=300, bbox_inches='tight')

plt.show()
