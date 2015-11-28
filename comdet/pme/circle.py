from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt


class Circle:

    def __init__(self, data=None):
        self.center = None
        self.radius = None
        if data is not None:
            self.fit(data)

    @property
    def min_sample_size(self):
        return 2

    def fit(self, data):
        if data.shape[0] < 3:
            raise ValueError('At least three points are needed to fit a circle')

        M = np.ones((data.shape[0], 3))
        b = np.zeros((data.shape[0],))

        for j in range(data.shape[0]):
            M[j, :2] = data[j, :]
            b[j] = - np.sum(data[j, :] ** 2)

        y = np.linalg.solve(M, b)
        self.center = -y[:2] / 2
        self.radius = np.linalg.norm(self.center) - y[2]
        print self.center, self.radius

    def distances(self, data):
        return np.abs(np.linalg.norm(data - self.center, axis=1) - self.radius)

    def plot(self, **kwargs):
        t = np.arange(0, 2*np.pi + 0.1, 0.1)
        x = self.center[0] + self.radius * np.cos(t)
        y = self.center[1] + self.radius * np.sin(t)
        plt.plot(x, y, **kwargs)


def test():
    x = np.array([[1, 1], [1, 2], [0, 0]])
    c1 = Circle(x)
    print c1.distances(x)
    print c1.distances(np.array([[0.5, 0], [0, 1], [0, 2]]))
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
    test()
