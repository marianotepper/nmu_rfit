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
            angles = _normalize(np.arctan2(lines[:, 0], lines[:, 1]))
            alpha = np.mean(angles)
            self.point = np.array([np.cos(alpha), np.sin(alpha), 0])

    # noinspection PyMethodMayBeStatic
    def project(self, data):
        lines = np.array([seg.line for seg in data])
        if self.point[2] != 0:
            return np.arctan2(lines[:, 1], lines[:, 0])
        else:
            return _normalize(np.arctan2(lines[:, 0], lines[:, 1]))

    def distances(self, data):
        lines = np.array([seg.line for seg in data])
        if self.point[2] != 0:
            points_a, points_b = zip(*[(s.p_a, s.p_b) for s in data])
            points_a = np.array(points_a)
            points_b = np.array(points_b)
            d_a = np.linalg.norm(points_a - self.point, axis=1)
            d_b = np.linalg.norm(points_b - self.point, axis=1)
            closer_points = np.copy(points_a)
            closer_points[d_a > d_b] = points_b[d_a > d_b]
            midpoints = (points_a + points_b) / 2
            v1 = self.point - midpoints
            v2 = closer_points - midpoints
            diff = np.sum(v1 * v2, axis=1)
            diff /= (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
            ang_diff = np.arccos(diff) * np.sign(np.cross(v1, v2)[:, 2])
            return ang_diff
        else:
            angleVP = _normalize(np.arctan2(self.point[1], self.point[0]))
            angleL = _normalize(np.arctan2(lines[:, 0], lines[:, 1]))
            angle_diff = angleVP - angleL
            return angle_diff

    def plot(self, **kwargs):
        plt.scatter(self.point[0], self.point[1], **kwargs)


def _normalize(a):
    return np.fmod(a + np.pi, np.pi)
