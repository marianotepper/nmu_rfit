import numpy as np
import matplotlib.pyplot as plt


class VanishingPoint():
    def __init__(self, data=None):
        self.point = None
        if data is not None:
            self.fit(data)

    @property
    def min_sample_size(self):
        return 2

    def fit(self, data):
        if len(data) < self.min_sample_size:
            raise ValueError('At least two segments are needed to fit a VP')
        lines = np.array([seg.line for seg in data])
        sol = np.linalg.lstsq(lines[:, :2], -lines[:, 2])
        if sol[2] == 2:  # matrix rank
            self.point = np.append(sol[0], [1])
        else:
            angles = _normalize(np.atan2(lines[:, 0], lines[:, 1]))
            alpha = np.mean(angles)
            self.point = np.array([np.cos(alpha), np.sin(alpha), 0])

    # noinspection PyMethodMayBeStatic
    def project(self, data):
        lines = np.array([seg.line for seg in data])
        if self.point[2] != 0:
            return np.atan2(lines[:, 1], lines[:, 0])
        else:
            return _normalize(np.atan2(lines[:, 0], lines[:, 1]))

    def distances(self, data):
        lines = np.array([seg.line for seg in data])
        if self.point[2] != 0:
            return np.abs(lines.dot(self.point))
        else:
            angleVP = _normalize(np.atan2(self.point[1], self.point[0]))
            angleL = _normalize(np.atan2(lines[:, 0], lines[:, 1]))
            angle_diff = _normalize(np.abs(angleVP - angleL))
            if np.abs(angle_diff - np.pi) < np.abs(angle_diff):
                angle_diff = np.abs(angle_diff - np.pi)
            return angle_diff

    def plot(self, **kwargs):
        plt.scatter(self.point[0], self.point[1], **kwargs)


def _normalize(a):
    return np.fmod(a + np.pi, np.pi)
