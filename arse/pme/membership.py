import numpy as np


class GlobalHardThresholder(object):
    def __init__(self, inliers_threshold):
        self.inliers_threshold = inliers_threshold

    def membership(self, model, data):
        return model.distances(data) <= self.inliers_threshold


class LocalHardThresholder(object):
    def __init__(self, inliers_threshold, ratio=2.):
        self.inliers_threshold = inliers_threshold
        self.ratio = ratio

    def membership(self, model, data):
        dist = model.distances(data)
        membership = dist <= self.inliers_threshold

        if np.all(membership):
            return membership

        outliers = np.logical_not(membership)
        upper_threshold = self.inliers_threshold * self.ratio
        upper_threshold = np.maximum(upper_threshold, np.min(dist[outliers]))
        upper_threshold = np.minimum(upper_threshold, np.max(dist[outliers]))

        membership = membership.astype(np.float)
        membership[dist > upper_threshold] = np.nan
        return membership


class ExponentialThresholder(object):
    def __init__(self, inliers_threshold, sigma, cutoff=5):
        self.inliers_threshold = inliers_threshold
        self.sigma = sigma
        self.cutoff = cutoff * self.inliers_threshold

    def membership(self, model, data, inliers_threshold=None):
        if inliers_threshold is None:
            inliers_threshold = self.inliers_threshold
        dist = model.distances(data)
        v = np.exp(-dist / inliers_threshold)
        v[dist > self.cutoff] = 0
        return v
