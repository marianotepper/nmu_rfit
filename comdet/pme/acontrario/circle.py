import numpy as np
import comdet.pme.acontrario.utils as utils
import matplotlib.pyplot as plt


class GlobalNFA(utils.BinomialNFA):
    def __init__(self, data, epsilon, inliers_threshold):
        super(GlobalNFA, self).__init__(data, epsilon, inliers_threshold)
        self.area = np.prod(np.max(data, axis=0) - np.min(data, axis=0))

    def _random_probability(self, model, inliers_threshold=None):
        if inliers_threshold is None:
            inliers_threshold = self.inliers_threshold
        # (a + b)**2 - (a - b)**2 == 4ab
        ring_area = np.pi * 4 * model.radius * inliers_threshold
        return min(ring_area / self.area, 1)


class LocalNFA(object):
    def __init__(self, data, epsilon, inliers_threshold):
        self.data = data
        self.epsilon = epsilon
        self.inliers_threshold = inliers_threshold

    def nfa(self, model, n_inliers, data=None, inliers_threshold=None,
            plot=False):
        if data is None:
            data = self.data
        if inliers_threshold is None:
            inliers_threshold = self.inliers_threshold

        if model.radius < inliers_threshold:
            return np.inf

        dist = model.distances(data)
        proj, s = model.project(data)
        mask_out = dist > inliers_threshold

        step = 60
        bins = np.linspace(-np.pi, np.pi, step)

        if plot:
            mask_in = dist <= inliers_threshold
            plt.figure()
            plt.axis('equal')
            plt.scatter(data[:, 0], data[:, 1], c='w')
            plt.scatter(data[mask_in, 0], data[mask_in, 1], c='r')
            model.plot()
            x = np.vstack((model.center[0] + model.radius * np.cos(bins),
                           model.center[1] + model.radius * np.sin(bins))).T
            plt.scatter(x[:, 0], x[:, 1], marker='x')

        idx = np.searchsorted(bins, s)
        dist_selected = np.zeros((bins.size,)) + inliers_threshold
        for k in range(dist_selected.size):
            sel = np.logical_and(mask_out, idx == k)
            if not np.any(sel):
                dist_selected[k] = np.nan
            else:
                dist_selected[k] = dist[sel].min()
                if plot:
                    data_sel = data[sel, :]
                    proj_sel = proj[sel, :]
                    i_m = np.argmin(dist[sel])
                    plt.scatter(data_sel[i_m, 0], data_sel[i_m, 1], marker='+')
                    plt.plot([proj_sel[i_m, 0], data_sel[i_m, 0]],
                             [proj_sel[i_m, 1], data_sel[i_m, 1]], color='k')

        upper_threshold = np.minimum(np.nanmedian(dist_selected), model.radius)
        region_mask = dist <= upper_threshold

        p = inliers_threshold / upper_threshold
        k = n_inliers - model.min_sample_size
        pfa = utils.log_binomial(region_mask.sum(), k, p)
        n_tests = utils.log_nchoosek(data.shape[0], model.min_sample_size)
        return (pfa + n_tests) / np.log(10)

    def meaningful(self, model, n_inliers):
        return self.nfa(model, n_inliers) < self.epsilon