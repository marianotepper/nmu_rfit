import numpy as np
import cv2
import utils


class Homography(object):
    def __init__(self, data=None):
        self.H = None
        self._trans1 = None
        self._trans2 = None
        if data is not None:
            self.fit(data)

    @property
    def min_sample_size(self):
        return 4

    def fit(self, data):
        if data.shape[0] < self.min_sample_size:
            raise ValueError('At least four points are needed to fit an '
                             'homography')
        if data.shape[1] != 6:
            raise ValueError('Points must be 6D (2 x 3D)')

        pts1_norm, self._trans1 = utils.normalize_2d(data[:, :3])
        pts2_norm, self._trans2 = utils.normalize_2d(data[:, 3:])
        self.H, _ = cv2.findHomography(pts1_norm[:, 0:2].astype(np.float32),
                                       pts2_norm[:, 0:2].astype(np.float32))

    def distances(self, data):
        pts1_norm = data[:, :3].dot(self._trans1)
        pts2_norm = data[:, 3:].dot(self._trans2)
        trans = np.dot(pts1_norm, self.H.T)
        trans /= np.atleast_2d(trans[:, 2]).T
        return np.sum(np.power(pts2_norm - trans, 2), axis=1)
