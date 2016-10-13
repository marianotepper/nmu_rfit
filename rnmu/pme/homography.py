import numpy as np
from rnmu.pme.proj_geom_utils import keep_finite, normalize_2d


class Homography(object):
    def __init__(self, data=None, weights=None):
        self.H = None
        if data is not None:
            self.fit(data, weights=weights)

    @property
    def min_sample_size(self):
        return 4

    def fit(self, data, weights=None):
        mask = keep_finite(data)
        data = data[mask, :]
        if weights is not None:
            weights = weights[mask]

        if len(data) < self.min_sample_size:
            raise ValueError('At least four points are needed to fit an '
                             'homography')
        if (weights is not None and
                    np.count_nonzero(weights) < self.min_sample_size):
            raise ValueError('At least four points are needed to fit an '
                             'homography')
        if data.shape[1] != 6:
            raise ValueError('Points must be 6D (2 x 3D)')

        pts1, trans1 = normalize_2d(data[:, :3], weights=weights)
        pts2, trans2, trans2_inv = normalize_2d(data[:, 3:], weights=weights,
                                                ret_inv=True)

        mat = np.zeros((9, 9))
        zero = np.zeros((3,))
        for i in range(np.count_nonzero(mask)):
            r1 = np.hstack([-pts1[i, :], zero, pts1[i, :] * pts2[i, 0]])
            r2 = np.hstack([zero, -pts1[i, :], pts1[i, :] * pts2[i, 1]])
            if weights is not None:
                r1 *= weights[i]
                r2 *= weights[i]
            mat += np.outer(r1, r1)
            mat += np.outer(r2, r2)
        try:
            _, v = np.linalg.eigh(mat)
            self.H = v[:, 0].reshape((3, 3))
            self.H = trans1.dot(self.H.T).dot(trans2_inv)
            self.H /= self.H[2, 2]
        except np.linalg.LinAlgError:
            self.H = None

    def distances(self, data):
        if self.H is None:
            return np.ones((len(data),)) * np.inf
        pts1 = data[:, :3]
        pts2 = data[:, 3:]
        trans = np.dot(pts1, self.H)

        mask = trans[:, 2] == 0
        trans[mask] = 1e-16

        trans /= np.atleast_2d(trans[:, 2]).T
        return np.sum(np.power(pts2 - trans, 2), axis=1)


if __name__ == '__main__':
    data1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    data2 = data1 * 2 + 3
    data = np.hstack((data1, np.ones((4, 1)), data2, np.ones((4, 1))))
    print data.shape
    Homography(data=data)
