import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as distance


class Point(object):
    def __init__(self, data=None, weights=None):
        self.pt = None
        if data is not None:
            self.fit(data, weights=weights)

    @property
    def min_sample_size(self):
        return 1

    def fit(self, data, weights=None):
        if data.shape[0] < self.min_sample_size:
            raise ValueError('At least one point is needed to fit a point')
        if (weights is not None and
                    np.count_nonzero(weights) < self.min_sample_size):
            raise ValueError('At least one point is needed to fit a point')
        if data.shape[1] != 2:
            raise ValueError('Points must be 2D')

        self.pt = np.average(data, weights=weights, axis=0)

    def distances(self, data):
        return np.squeeze(distance.cdist(data, self.pt[np.newaxis, :]))

    def plot(self, **kwargs):
        if 'edgecolor' not in kwargs:
            kwargs['edgecolor'] = 'none'
        plt.scatter(self.pt[0], self.pt[1], **kwargs)
