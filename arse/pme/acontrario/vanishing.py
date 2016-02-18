import numpy as np
from . import nfa


class VanishingThresholder(object):
    def __init__(self, threshold_in_image, img_radius, img_center):
        self.threshold_in_image = threshold_in_image
        self.img_radius = img_radius
        self.img_center = img_center

    def threshold(self, model):
        if model.point[2] != 0:
            dist = np.linalg.norm(model.point[:2] - self.img_center)
            if dist > self.img_radius:
                length_in_out = 2 * np.pi * self.threshold_in_image
                midpoint = (dist - self.img_radius) / 2
                r = (midpoint + self.img_radius) / dist
                q = (midpoint - self.img_radius) / dist
                radius = length_in_out
                radius -= 2 * dist * (np.sqrt(1 - r**2) - np.sqrt(1 - q**2))
                radius -= 2 * self.img_radius *  (np.arcsin(r) + np.arcsin(q))
                radius /= 2 * (np.arcsin(r) - np.arcsin(q))
                return radius
            else:
                return self.threshold_in_image
        else:
            return (self.threshold_in_image / self.img_radius) * 2 * np.pi


class GlobalNFA(nfa.BinomialNFA, VanishingThresholder):
    def __init__(self, data, epsilon, threshold_in_image, img_radius,
                 img_center):
        utils.BinomialNFA.__init__(self, data, epsilon)
        VanishingThresholder.__init__(self, threshold_in_image, img_radius,
                                      img_center)

    def _binomial_params(self, model, data, inliers_threshold):
        inliers_mask = self.inliers(model, data, inliers_threshold)
        if model.point[2] != 0 and inliers_threshold == self.threshold_in_image:
            p = self.threshold_in_image / self.img_radius
        elif model.point[2] != 0:
            dist = np.linalg.norm(model.point[:2] - self.img_center)
            if dist > self.img_radius:
                phi = np.arccos((self.img_radius + inliers_threshold) / dist)
                theta = np.arccos((self.img_radius - inliers_threshold) / dist)
                length_in = 2 * (self.img_radius + inliers_threshold)
                length_in *= np.tan(phi) + np.pi - phi
                length_out = 2 * dist * np.sin(theta)
                length_out += self.img_radius * (2 * (np.pi - theta))
                length_out += inliers_threshold * 2 * theta
                length_img = 2 * np.pi * self.img_radius
                p = (length_in - length_out) / length_img
            else:
                p = inliers_threshold / self.img_radius
        else:
            p = inliers_threshold / (2 * np.pi)
        return len(data), inliers_mask.sum(), p

    def threshold(self, model):
        return VanishingThresholder.threshold(self, model)
