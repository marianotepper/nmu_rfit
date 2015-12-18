from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt


class Circle(object):

    def __init__(self, data=None):
        self.center = None
        self.radius = None
        if data is not None:
            self.fit(data)

    @property
    def min_sample_size(self):
        return 3

    def fit(self, data):
        if data.shape[0] < self.min_sample_size:
            raise ValueError('At least three points are needed to fit a circle')

        a = np.hstack((data, np.ones((data.shape[0], 1))))
        b = -np.sum(data ** 2, axis=1)
        y = np.linalg.lstsq(a, b)[0]
        self.center = -0.5 * y[:2]
        self.radius = np.sqrt(np.sum(self.center ** 2) - y[2])

    def distances(self, data):
        return np.abs(np.linalg.norm(data - self.center, axis=1) - self.radius)

    def project(self, data):
        vecs = data - self.center
        angles = np.arctan2(vecs[:, 1], vecs[:, 0])
        proj = np.vstack((self.center[0] + self.radius * np.cos(angles),
                          self.center[1] + self.radius * np.sin(angles))).T
        return proj, angles

    def plot(self, **kwargs):
        t = np.arange(0, 2*np.pi + 0.1, 0.1)
        x = self.center[0] + self.radius * np.cos(t)
        y = self.center[1] + self.radius * np.sin(t)
        plt.plot(x, y, **kwargs)


def _test():
    x = np.array([[-0.08045062, 1.42318932],
                  [ 0.33886852, 1.16171235],
                  [-0.21174648, 1.08196667]])
    # x = np.array([[1, 1], [1, 2], [0, 0]])
    c1 = Circle(x)
    print c1.distances(x)
    # print c1.distances(np.array([[0.5, 0], [0, 1], [0, 2]]))
    # x = np.array([[0, 0], [2, 2]])
    # l2 = Line(x)
    # print l2.distances(x)
    #
    plt.figure()
    plt.plot(x[:, 0], x[:, 1], 'bo')
    c1.plot(color='g', linewidth=2)
    # l2.plot(color='g', linewidth=2)
    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
    _test()
