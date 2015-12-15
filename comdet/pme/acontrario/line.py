import numpy as np
import comdet.pme.acontrario.utils as utils
import matplotlib.pyplot as plt


class GlobalNFA(utils.BinomialNFA):
    def __init__(self, data, epsilon, inliers_threshold):
        super(GlobalNFA, self).__init__(data, epsilon, inliers_threshold)

    def _random_probability(self, model, inliers_threshold=None):
        if inliers_threshold is None:
            inliers_threshold = self.inliers_threshold
        area = np.prod(np.max(self.data, axis=0) - np.min(self.data, axis=0))
        _, s = model.project(self.data)
        length = s.max() - s.min()
        return length * 2 * inliers_threshold / area


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

        dist = model.distances(data)
        proj, s = model.project(data)
        mask_in = dist <= inliers_threshold
        mask_out = np.logical_not(mask_in)

        step = inliers_threshold * 2
        bins = np.arange(s.min(), s.max() + step, step)

        if plot:
            u, x0 = model.point_and_basis()
            plt.figure()
            plt.axis('equal')
            plt.scatter(data[:, 0], data[:, 1], c='w')
            plt.scatter(data[mask_in, 0], data[mask_in, 1], c='r')
            model.plot()
            x = x0 + np.atleast_2d(bins).T * u
            plt.scatter(x[:, 0], x[:, 1], marker='x')

        idx = np.searchsorted(bins, s)
        dist_selected = np.zeros((bins.size - 1,)) + inliers_threshold
        for k in range(dist_selected.size):
            sel = np.logical_and(mask_out, idx == (k+1))
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

        upper_threshold = np.nanmedian(dist_selected)
        region_mask = dist <= upper_threshold

        p = inliers_threshold / upper_threshold
        k = n_inliers - model.min_sample_size
        pfa = utils.log_binomial(region_mask.sum(), k, p)
        n_tests = utils.log_nchoosek(data.shape[0], model.min_sample_size)
        return (pfa + n_tests) / np.log(10)

    def meaningful(self, model, n_inliers):
        return self.nfa(model, n_inliers) < self.epsilon
