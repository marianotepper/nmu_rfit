import numpy as np
import comdet.pme.acontrario.utils as utils


class GlobalNFA(utils.BinomialNFA):
    def __init__(self, data, epsilon, threshold_in_image, img_radius, img_center):
        super(GlobalNFA, self).__init__(data, epsilon)
        self.threshold_in_image = threshold_in_image
        self.img_radius = img_radius
        self.img_center = img_center

    def _random_probability(self, model, data=None, inliers_threshold=None):
        if not inliers_threshold:
            return self.threshold_in_image / self.img_radius
        else:
            if model.point[2] != 0:
                dist = np.linalg.norm(model.point[:2] - self.img_center)
                if dist > self.img_radius:
                    ph = np.arccos((self.img_radius + inliers_threshold) / dist)
                    ro = np.arccos((self.img_radius - inliers_threshold) / dist)
                    length_in = 2 * (self.img_radius + inliers_threshold)
                    length_in *= np.tan(ph) + np.pi - ph
                    length_out = 2 * dist * np.sin(ro)
                    length_out += self.img_radius * (2 * (np.pi - ro))
                    length_out += inliers_threshold * 2 * ro
                    length_img = 2 * np.pi * self.img_radius
                    return (length_in - length_out) / length_img
                else:
                    return inliers_threshold / self.img_radius

            else:
                return inliers_threshold / (2 * np.pi)

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
