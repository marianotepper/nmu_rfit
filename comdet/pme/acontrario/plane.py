import numpy as np
import comdet.pme.acontrario.utils as utils
import matplotlib.pyplot as plt


class GlobalNFA(utils.BinomialNFA):
    def __init__(self, data, epsilon, inliers_threshold):
        super(GlobalNFA, self).__init__(data, epsilon, inliers_threshold)

    def _random_probability(self, model, inliers_threshold=None):
        if inliers_threshold is None:
            inliers_threshold = self.inliers_threshold
        volume = np.prod(np.max(self.data, axis=0) - np.min(self.data, axis=0))
        _, s = model.project(self.data)
        area = np.prod(s.max(axis=0) - s.min(axis=0))
        return area * 2 * inliers_threshold / volume


class LocalNFA(object):
    def __init__(self, data, epsilon, inliers_threshold):
        self.data = data
        self.epsilon = epsilon
        self.inliers_threshold = inliers_threshold

    def nfa(self, model, n_inliers, data=None, inliers_threshold=None, plot=False):
        if data is None:
            data = self.data
        if inliers_threshold is None:
            inliers_threshold = self.inliers_threshold

        dist = model.distances(data)
        proj, s = model.project(data)
        mask_in = dist <= inliers_threshold
        mask_out = np.logical_not(mask_in)

        s_x = s[:, 0]
        s_y = s[:, 1]

        step = 100
        bins_x = np.linspace(s_x.min(), s_x.max(), step)
        bins_y = np.linspace(s_y.min(), s_y.max(), step)

        if plot:
            # u, x0 = model.point_and_basis()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.axis('equal')
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='w')
            ax.scatter(data[mask_in, 0], data[mask_in, 1], data[mask_in, 2], c='r')
            model.plot(ax)
            # x = x0 + bins.dot(u)
            # plt.scatter(x[:, 0], x[:, 1], marker='x')

        # idx_x = np.searchsorted(bins_x, s_x)
        # idx_y = np.searchsorted(bins_y, s_y)
        # dist_selected = np.zeros((step, step,)) + inliers_threshold
        # for k_x in range(step):
        #     for k_y in range(step):
        #         sel = np.logical_and(idx_x == k_x, idx_y == k_y)
        #         sel = np.logical_and(mask_out, sel)
        #         if not np.any(sel):
        #             dist_selected[k_x, k_y] = np.nan
        #         else:
        #             dist_selected[k_x, k_y] = dist[sel].min()
        #             if plot:
        #                 data_sel = data[sel, :]
        #                 proj_sel = proj[sel, :]
        #                 i_m = np.argmin(dist[sel])
        #                 ax.scatter(data_sel[i_m, 0], data_sel[i_m, 1], data_sel[i_m, 2], marker='+')
        #                 ax.plot([proj_sel[i_m, 0], data_sel[i_m, 0]],
        #                         [proj_sel[i_m, 1], data_sel[i_m, 1]],
        #                         [proj_sel[i_m, 2], data_sel[i_m, 2]], color='k')

        # upper_threshold = np.nanmedian(dist_selected)
        upper_threshold = inliers_threshold * 3
        region_mask = dist <= upper_threshold

        p = inliers_threshold / upper_threshold
        k = n_inliers - model.min_sample_size
        pfa = utils.log_binomial(region_mask.sum(), k, p)
        n_tests = utils.log_nchoosek(data.shape[0], model.min_sample_size)
        return (pfa + n_tests) / np.log(10)

    def meaningful(self, model, n_inliers):
        return self.nfa(model, n_inliers) < self.epsilon
