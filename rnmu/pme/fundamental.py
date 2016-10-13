import numpy as np
from rnmu.pme.proj_geom_utils import keep_finite, normalize_2d


class Fundamental(object):
    def __init__(self, data=None, weights=None):
        self.F = None
        if data is not None:
            self.fit(data, weights=weights)

    @property
    def min_sample_size(self):
        return 8

    def fit(self, data, weights=None):
        mask = keep_finite(data)
        data = data[mask, :]
        if weights is not None:
            weights = weights[mask]

        if len(data) < self.min_sample_size:
            raise ValueError('At least eight points are needed to fit a '
                             'fundamental matrix')
        if (weights is not None and
                    np.count_nonzero(weights) < self.min_sample_size):
            raise ValueError('At least eight points are needed to fit a '
                             'fundamental matrix')
        if data.shape[1] != 6:
            raise ValueError('Points must be 6D (2 x 3D)')

        pts1, trans1 = normalize_2d(data[:, :3], weights=weights)
        pts2, trans2 = normalize_2d(data[:, 3:], weights=weights)

        mat = np.zeros((9, 9))
        for i in range(np.count_nonzero(mask)):
            row = np.hstack([pts1[i, :] * pts2[i, 0],
                             pts1[i, :] * pts2[i, 1],
                             pts1[i, :]])
            if weights is not None:
                row *= weights[i]
            mat += np.outer(row, row)
        try:
            _, v = np.linalg.eigh(mat)
            u, s, vt = np.linalg.svd(v[:, 0].reshape((3, 3)))
            s[2] = 0
            self.F = u.dot(np.diag(s)).dot(vt)
            self.F = trans1.dot(self.F.T.dot(trans2.T))
            self.F /= self.F[2, 2]
        except np.linalg.LinAlgError:
            self.F = None

    def distances(self, data):
        if self.F is None:
            return np.ones((len(data),)) * np.inf

        pts1 = data[:, :3]
        pts2 = data[:, 3:]

        epi_lines1 = np.dot(pts1, self.F)
        epi_lines1 /= np.linalg.norm(epi_lines1[:, :2], axis=1)[:, np.newaxis]
        return np.abs(np.sum(pts2 * epi_lines1, axis=1))
