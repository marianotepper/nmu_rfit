import numpy as np
import cv2
import warnings



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
        pts2, trans2 = normalize_2d(data[:, 3:], weights=weights)

        mat = []
        zero = np.zeros((3,))
        for i in range(np.count_nonzero(mask)):
            r1 = np.hstack([-pts1[i, :], zero, pts1[i, :] * pts2[i, 0]])
            r2 = np.hstack([zero, -pts1[i, :], pts1[i, :] * pts2[i, 1]])
            if weights is None:
                mat.append(r1)
                mat.append(r2)
            elif weights[i] > 0:
                mat.append(r1 * weights[i])
                mat.append(r2 * weights[i])

        mat = np.array(mat)
        try:
            _, _, vt = np.linalg.svd(mat)
            self.H = vt[8, :].reshape((3, 3))

            self.H = np.linalg.solve(trans2.T, self.H.dot(trans1.T)).T
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


def keep_finite(data):
    if data.shape[1] != 6:
        raise ValueError('The points must be Nx6');

    # Find the indices of the points that are not at infinity
    finite_idx = np.logical_and(np.abs(data[:, 2]) > np.finfo(float).eps,
                                np.abs(data[:, 5]) > np.finfo(float).eps)

    if np.count_nonzero(finite_idx) != len(data):
        warnings.warn('Found points at infinity', RuntimeWarning)

    return finite_idx


def normalize_2d(points, weights=None):
    if points.shape[1] != 3:
        raise ValueError('The points must be Nx3');

    # Ensure that homogeneous coord is 1
    points[:, 0] /= points[:, 2]
    points[:, 1] /= points[:, 2]
    points[:, 2] = 1

    centroid = np.average(points[:, :2], weights=weights, axis=0)
    dist = np.linalg.norm(points[:, :2] - centroid, ord=2, axis=1)
    scale = np.sqrt(2) / np.average(dist, weights=weights)

    trans = np.array([[scale, 0, 0],
                      [0, scale, 0],
                      [-scale * centroid[0], -scale * centroid[1], 1]])

    return points.dot(trans), trans


if __name__ == '__main__':
    data1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    data2 = data1 * 2 + 3
    data = np.hstack((data1, np.ones((4, 1)), data2, np.ones((4, 1))))
    print data.shape
    Homography(data=data)
