import numpy as np
import warnings

def keep_finite(data):
    if data.shape[1] != 6:
        raise ValueError('The points must be Nx6');

    # Find the indices of the points that are not at infinity
    finite_idx = np.logical_and(np.abs(data[:, 2]) > np.finfo(float).eps,
                                np.abs(data[:, 5]) > np.finfo(float).eps)

    if np.count_nonzero(finite_idx) != len(data):
        warnings.warn('Found points at infinity', RuntimeWarning)

    return finite_idx


def normalize_2d(points, weights=None, ret_inv=False):
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

    if not ret_inv:
        return points.dot(trans), trans
    else:
        scale = 1 / scale
        trans_inv = np.array([[scale, 0, 0],
                              [0, scale, 0],
                              [centroid[0], centroid[1], 1]])
        return points.dot(trans), trans, trans_inv