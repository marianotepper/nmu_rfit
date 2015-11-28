import numpy as np
import matplotlib.pyplot as plt
import utils


class Line:

    def __init__(self, data=None):
        self.eq = None
        if data is not None:
            self.fit(data)

    @property
    def min_sample_size(self):
        return 2

    def fit(self, data):
        if data.shape[0] < 2:
            raise ValueError('At least two points are needed to fit a line')
        if data.shape[1] != 2:
            raise ValueError('Points must be 2D')

        data = np.hstack((data, np.ones((data.shape[0], 1))))
        data_norm, trans = utils.normalize_2d(data)
        if data_norm.shape[0] == 2:
            data_norm = np.vstack((data_norm, np.zeros((1, 3))))
        _, _, v = np.linalg.svd(data_norm, full_matrices=False)
        self.eq = v[2, :].dot(trans.T)

    def distances(self, data):
        if data.shape[1] == 2:
            data = np.hstack((data, np.ones((data.shape[0], 1))))
        return np.abs(np.dot(data, self.eq)) / np.linalg.norm(self.eq[:2])

    def plot(self, limits=None, **kwargs):
        if limits is None:
            xlim = plt.xlim()
            ylim = plt.ylim()
        else:
            xlim = limits[0]
            ylim = limits[1]

        if abs(self.eq[0]) > abs(self.eq[1]):  # line is more vertical
            p1 = np.array([0., 1., -ylim[0]])
            p2 = np.array([0., 1., -ylim[1]])
        else:  # line more horizontal
            p1 = np.array([1., 0., -xlim[0]])
            p2 = np.array([1., 0., -xlim[1]])

        p1 = np.cross(self.eq, p1)
        p2 = np.cross(self.eq, p2)
        p1 /= p1[2]
        p2 /= p2[2]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], **kwargs)


def test():
    x = np.array([[1, 1], [1, 2], [1, 3]])
    l1 = Line(x)
    print l1.distances(x)
    print l1.distances(np.array([[0.5, 0], [0, 1], [0, 2]]))
    x = np.array([[0, 0], [2, 2]])
    l2 = Line(x)
    print l2.distances(x)

    plt.figure()
    l1.plot(color='g', linewidth=2)
    l2.plot(color='g', linewidth=2)
    plt.show()

if __name__ == '__main__':
    test()
