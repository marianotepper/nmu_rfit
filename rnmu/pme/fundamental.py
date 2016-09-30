import numpy as np
import cv2


class Fundamental(object):
    def __init__(self, data=None):
        self.F = None
        self.status = None
        if data is not None:
            self.fit(data)

    @property
    def min_sample_size(self):
        return 8

    def fit(self, data):
        if data.shape[0] < self.min_sample_size:
            raise ValueError('At least eight points are needed to fit a '
                             'fundamental matrix')
        if data.shape[1] != 6:
            raise ValueError('Points must be 6D (2 x 3D)')

        pts1_norm = data[:, :3]
        pts2_norm = data[:, 3:]
        self.F, _ = cv2.findFundamentalMat(pts1_norm[:, 0:2].astype(np.float32),
                                           pts2_norm[:, 0:2].astype(np.float32),
                                           method=cv2.FM_8POINT)

    def distances(self, data):
        pts1_norm = data[:, :3]
        pts2_norm = data[:, 3:]

        epi_lines1 = np.dot(pts1_norm, self.F.T)
        epi_lines1 /= np.linalg.norm(epi_lines1[:, :2], axis=1)[:, np.newaxis]
        return np.abs(np.sum(pts2_norm * epi_lines1, axis=1))
