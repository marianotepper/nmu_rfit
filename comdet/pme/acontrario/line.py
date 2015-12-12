import numpy as np
import scipy.stats as stats
import comdet.pme.acontrario.utils as utils
# import matplotlib.pyplot as plt


class LineBinomialNFA(utils.BinomialNFA):
    def __init__(self, data, epsilon, inliers_threshold):
        super(LineBinomialNFA, self).__init__(data, epsilon, inliers_threshold)

    def _random_probability(self, model, inliers_threshold=None):
        if inliers_threshold is None:
            inliers_threshold = self.inliers_threshold
        area = np.prod(np.max(self.data, axis=0) - np.min(self.data, axis=0))
        _, s = model.project(self.data)
        length = s.max() - s.min()
        return length * 2 * inliers_threshold / area


class BoxedNFA(object):
    def __init__(self, data, epsilon, inliers_threshold):
        self.data = data
        self.epsilon = epsilon
        self.inliers_threshold = inliers_threshold

    def nfa(self, model, n_inliers, inliers_threshold=None):
        if inliers_threshold is None:
            inliers_threshold = self.inliers_threshold

        dist = model.distances(self.data)
        proj, s, u, x0 = model.project(self.data)
        mask_in = dist <= inliers_threshold
        mask_out = np.logical_not(mask_in)

        step = inliers_threshold * 2
        bins = np.arange(s[mask_in].min() - inliers_threshold,
                         s[mask_in].max() + step, step)

        # plt.figure()
        # plt.axis('equal')
        # plt.scatter(self.data[:, 0], self.data[:, 1], c='w')
        # plt.scatter(self.data[mask_in, 0], self.data[mask_in, 1], c='r')
        # model.plot()
        # x = x0 + np.atleast_2d(bins).T * u
        # plt.scatter(x[:, 0], x[:, 1], marker='x')

        idx = np.searchsorted(bins, s)
        dist_selected = np.zeros((bins.size - 1,)) + inliers_threshold
        for k in range(dist_selected.size):
            sel = np.logical_and(mask_out, idx == (k+1))
            if not np.any(sel):
                dist_selected[k] = np.nan
            else:
                # data_sel = self.data[sel, :]
                # proj_sel = proj[sel]
                # i_m = np.argmin(dist[sel])
                # plt.scatter(data_sel[i_m, 0], data_sel[i_m, 1], marker='+')
                # plt.plot([proj_sel[i_m, 0], data_sel[i_m, 0]], [proj_sel[i_m, 1], data_sel[i_m, 1]], color='k')
                dist_selected[k] = dist[sel].min()

        upper_threshold = np.nanmedian(dist_selected)
        region_mask = dist <= upper_threshold

        line_length = s.max() - s.min()
        region_area = line_length * 2 * upper_threshold
        line_area = (2 * inliers_threshold) * line_length
        p = line_area / region_area
        pfa = utils.log_binomial(region_mask.sum(), mask_in.sum(), p)
        n_tests = utils.log_nchoosek(self.data.shape[0], model.min_sample_size)
        # TODO: think why the correction by bins.size - 1 is needed
        return (pfa + n_tests) / np.log(10) + np.log10(bins.size - 1)

    def meaningful(self, model, n_inliers):
        return self.nfa(model, n_inliers) < self.epsilon
