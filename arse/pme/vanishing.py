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
        if len(data) == self.min_sample_size:
            self.point = np.cross(data[0].line, data[1].line)
            self.point /= self.point[2]
        else:
            lines = np.array([seg.line for seg in data])
            sol = np.linalg.lstsq(lines[:, :2], -lines[:, 2])
            point_plane = np.append(sol[0], [1])
            dists_plane = distances(point_plane, data)

            angles = _normalize(np.arctan2(lines[:, 0], lines[:, 1]))
            alpha = np.mean(angles)
            point_inf = np.array([np.cos(alpha), np.sin(alpha), 0])
            dists_inf = distances(point_inf, data)

            if np.max(np.abs(dists_plane)) < np.max(np.abs(dists_inf)):
                self.point = point_plane
            else:
                self.point = point_inf

    def distances(self, data):
        return distances(self.point, data)

    def plot(self, **kwargs):
        plt.scatter(self.point[0], self.point[1], **kwargs)


def _normalize(a):
    return np.fmod(a + np.pi, np.pi)


def basis_vector(segment):
    u = segment.p_a - segment.p_b
    u /= np.linalg.norm(u)
    return np.atleast_2d(u[:2])


def distances(point, data):
    lines = np.array([seg.line for seg in data])
    if point[2] != 0:
        points_a, points_b = zip(*[(s.p_a, s.p_b) for s in data])
        points_a = np.array(points_a)
        points_b = np.array(points_b)
        d_a = np.linalg.norm(points_a - point, axis=1)
        d_b = np.linalg.norm(points_b - point, axis=1)
        closer_points = np.copy(points_a)
        closer_points[d_a > d_b] = points_b[d_a > d_b]
        midpoints = (points_a + points_b) / 2
        v1 = point - midpoints
        v2 = closer_points - midpoints
        diff = np.sum(v1 * v2, axis=1)  # dot product
        diff /= (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
        angle_diff = np.arccos(diff) * np.sign(np.cross(v1, v2)[:, 2])
    else:
        angle_vp = _normalize(np.arctan2(point[1], point[0]))
        angle_lines = _normalize(np.arctan2(lines[:, 0], lines[:, 1]))
        angle_diff = angle_vp - angle_lines
        mask = angle_diff > np.pi / 2
        angle_diff[mask] = np.pi - angle_diff[mask]
        mask = angle_diff < -np.pi / 2
        angle_diff[mask] = np.pi + angle_diff[mask]
    return np.abs(angle_diff)