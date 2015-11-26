import warnings
import numpy as np


def normalize_2d(points):
    if points.shape[1] != 3:
        raise ValueError('The points must be Nx3');


    # Find the indices of the points that are not at infinity
    finite_idx = np.where(np.abs(points[:, 2]) > np.finfo(float).eps)[0]
    print finite_idx, finite_idx.size, points.shape[0]

    if finite_idx.size != points.shape[0]:
        warnings.warn('Found points at infinity', RuntimeWarning)

    new_points = np.copy(points)
    # For the finite points ensure homogeneous coords have scale of 1
    new_points[finite_idx, 0] /= points[finite_idx, 2]
    new_points[finite_idx, 1] /= points[finite_idx, 2]
    new_points[finite_idx, 2] = 1

    centroid = np.mean(new_points[finite_idx, :2], axis=0)
    dist = np.linalg.norm(new_points[finite_idx, :2] - centroid, ord=0, axis=1)
    scale = np.sqrt(2) / np.mean(dist)

    trans = np.array([[scale, 0, 0],
                      [0, scale, 0],
                      [-scale * centroid[0], -scale * centroid[1], 1]])

    return new_points.dot(trans), trans

