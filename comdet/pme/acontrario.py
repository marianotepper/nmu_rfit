import numpy as np
import scipy.sparse as sp
import scipy.special as special
import itertools
import multipledispatch
import heapq
import utils
import line
import circle

import matplotlib.pyplot as plt


def meaningful(data, model, inliers, inliers_threshold, epsilon):
    return compute_nfa(data, model, inliers, inliers_threshold) < epsilon


def filter_meaningful(meaningful_fun, mod_inliers_iter):
    return itertools.ifilter(meaningful_fun, mod_inliers_iter)


def compute_nfa(data, model, inliers, inliers_threshold):
    probability = random_probability(data, inliers_threshold, model)
    return inner_compute_nfa(inliers, model.min_sample_size, probability)


def inner_compute_nfa(inliers, n_samples, instance_proba):
    n = inliers.shape[0]
    k = inliers.sum() - n_samples
    if k <= 0:
        pfa = np.inf
    else:
        pfa = log_betainc(k, n - k + 1, instance_proba)
    n_tests = log_nchoosek(n, n_samples)
    return (pfa + n_tests) / np.log(10)


@multipledispatch.dispatch(np.ndarray, float, line.Line)
def random_probability(data, inliers_threshold, model):
    vec = np.max(data, axis=0) - np.min(data, axis=0)
    area = np.prod(vec)
    length = np.linalg.norm(vec)
    return length * 2 * inliers_threshold / area


@multipledispatch.dispatch(np.ndarray, float, circle.Circle)
def random_probability(data, inliers_threshold, model):
    upper = np.maximum(np.max(data, axis=0), model.center + model.radius)
    lower = np.minimum(np.min(data, axis=0), model.center - model.radius)
    area = np.prod(np.max(data, axis=0) - np.min(data, axis=0))
    ring_area = np.pi * ((model.radius + inliers_threshold) ** 2 -
                         (model.radius - inliers_threshold) ** 2)
    return min(ring_area / area, 1)


def log_nchoosek(n, k):
    return special.gammaln(n + 1) - special.gammaln(n - k + 1)\
           - special.gammaln(k + 1)


def log_betainc(a, b, x):
    if a <= 0.0 or b <= 0.0:
        raise ValueError('Bad a or b in function log_betainc')
    if x <= 0.0 or x > 1.0:
        raise ValueError('Bad x in function log_betainc')
    if x == 1.0:
        return 0

    logbt = special.gammaln(a + b) - special.gammaln(a) - special.gammaln(b) +\
            a * np.log(x) + b * np.log(1.0 - x)

    if x < (a + 1.0) / (a + b + 2.0):
        # Use continued fraction directly
        return logbt + np.log(special.betainc(a, b, x))
    else:
        # Factors in front of the continued fraction.
        bt = np.exp(logbt)
        # Use continued fraction after making the symmetry transformation.
        return np.log(1.0 - bt * special.betainc(b, a, 1.0 - x) / b)


class _DummyArray:
    def __init__(self, shape, nnz):
        self.shape = shape
        self.nnz = nnz

    def sum(self):
        return self.nnz


def multiscale_meaningful(data, model, epsilon, min_count=None, max_count=None,
                          max_thresh=None):
    dist_arr = model.distances(data)
    dist = model.distances(data).tolist()
    heapq.heapify(dist)
    nnz = 0
    while dist:
        s = heapq.heappop(dist)
        nnz += 1
        if min_count is not None and nnz < min_count:
            continue
        if max_count is not None and nnz > max_count:
            break
        if max_thresh is not None and s > max_thresh:
            break
        inliers = _DummyArray((data.shape[0],), nnz)
        if meaningful(data, model, inliers, s, epsilon):
            print 'meaningful', s, nnz, max_count
            if nnz == 31 and max_count == 36:
                print '----',
                mask = dist_arr <= s
                print mask.sum(),
                print '----'

                x_lim = (data[:, 0].min() - 0.1, data[:, 0].max() + 0.1)
                y_lim = (data[:, 1].min() - 0.1, data[:, 1].max() + 0.1)
                delta_x = x_lim[1] - x_lim[0]
                delta_y = y_lim[1] - y_lim[0]
                min_delta = min([delta_x, delta_y])
                delta_x /= min_delta
                delta_y /= min_delta
                fig_size = (4 * delta_x, 4 * delta_y)

                plt.figure(figsize=fig_size)
                plt.xlim(x_lim)
                plt.ylim(y_lim)
                plt.hold(True)
                plt.scatter(data[:, 0], data[:, 1], c='w', marker='o', s=10)
                plt.scatter(data[mask, 0], data[mask, 1], c='r', marker='o', s=10)
                model.plot(threshold=s)
            return True
    return False


def optimal_nfa(data, model, max_thresh, considered=None):
    if considered is None:
        dist = model.distances(data)
    else:
        dist = model.distances(data[sp.find(considered)[0], :])
    min_nfa = np.inf
    sorted_dist = np.sort(dist)
    for k, d in enumerate(sorted_dist):
        if d >= max_thresh:
            break
        inliers = _DummyArray((data.shape[0],), k+1)
        nfa = compute_nfa(data, model, inliers, d)
        if nfa < min_nfa:
            min_nfa = nfa
            min_d = (compute_nfa(data, model, inliers, d), k+1, d)
    print min_d
    return min_nfa


def best_nfa(data, model, inliers):
    included = sp.find(inliers)[0]
    if included.shape[0] == 0:
        return np.inf
    data_included = data[included, :]
    c = circle.Circle(data_included)
    max_dist = c.distances(data_included).max()
    # max_dist = model.distances(data[included, :]).max()
    nfa = compute_nfa(data, model, inliers, max_dist)
    print included.shape, max_dist, nfa, inliers.shape

    if included.shape[0] == 86 or included.shape[0] == 76 or included.shape[0] == 71 or included.shape[0] == 81:
        x_lim = (data[:, 0].min() - 0.1, data[:, 0].max() + 0.1)
        y_lim = (data[:, 1].min() - 0.1, data[:, 1].max() + 0.1)
        delta_x = x_lim[1] - x_lim[0]
        delta_y = y_lim[1] - y_lim[0]
        min_delta = min([delta_x, delta_y])
        delta_x /= min_delta
        delta_y /= min_delta
        fig_size = (4 * delta_x, 4 * delta_y)

        plt.figure(figsize=fig_size)
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.hold(True)
        plt.scatter(data[:, 0], data[:, 1], c='w', marker='o', s=10)
        plt.scatter(data_included[:, 0], data_included[:, 1], c='r', marker='o', s=10)
        model.plot(threshold=max_dist)
    return nfa


def exclusion_principle(data, mod_inliers_list, inliers_threshold, epsilon):
    def inner_meaningful((model, inliers)):
        return meaningful(data, model, inliers, inliers_threshold, epsilon)

    filtered = list(filter_meaningful(inner_meaningful, mod_inliers_list))

    nfa_list = []
    # threshold_list = []
    for i, (mod, in_a) in enumerate(filtered):
        print i
        in_list = [in_b for j, (_, in_b) in enumerate(filtered) if i != j]
        in_list = map(lambda x: in_a.multiply(x).astype(bool), in_list)
        inliers = in_a - reduce(lambda x, y: (x + y).astype(bool), in_list)

        # nfa = compute_nfa(data, mod, in_a, inliers_threshold)
        # nfa = optimal_nfa(data, mod, inliers_threshold, considered=inliers)
        # nfa = best_nfa(data, mod, inliers)
        nfa = best_nfa(data, mod, in_a)
        print inliers.nnz, in_a.nnz, nfa
        # if nfa < epsilon:
        nfa_list.append(nfa)
        # threshold_list.append(new_thresh)
    idx = utils.argsort(nfa_list)

    print [(pick, nfa_list[pick]) for pick in idx]

    keep_list = list(idx)
    out_list = []
    for i, pick in enumerate(idx):
        mod, in_a = mod_inliers_list[pick]
        if i == 0:
            out_list.append(pick)
            continue

        # in_list = [mod_inliers_list[k][1] for k in out_list]
        # in_list = map(lambda x: in_a - in_a.multiply(x).astype(bool), in_list)

        # new_thresh = threshold_list[pick]

        in_list = [mod_inliers_list[k][1] for k in keep_list if k != pick]
        in_list = map(lambda x: in_a.multiply(x).astype(bool), in_list)
        inliers = in_a - reduce(lambda x, y: (x + y).astype(bool), in_list)
        # print inliers.nnz

        # proba = random_probability(data, new_thresh, mod)
        # print pick, in_a.nnz, [in_b.nnz for in_b in in_list],\
        #     [compute_nfa(in_b, mod.min_sample_size, proba)
        #              for in_b in in_list],\
        #     [multiscale_meaningful(data, mod, epsilon, max_count=in_b.nnz) for in_b in in_list],\
        print i, in_a.nnz

        # if all([compute_nfa(data, mod, in_b, inliers_threshold) < epsilon for in_b in in_list]):
        # if all([best_nfa(data, mod, in_b) < epsilon for in_b in in_list]):
        # if all([multiscale_meaningful(data, mod, epsilon, min_count=in_b.nnz*0.9, max_count=in_b.nnz) for in_b in in_list]):
        # if best_nfa(data, mod, inliers) < epsilon:
        if multiscale_meaningful(data, mod, epsilon, max_count=inliers.nnz):
            print pick, 'meaningful', best_nfa(data, mod, in_b)
            # if pick != 8:
            out_list.append(pick)
        else:
            keep_list.remove(pick)
            print pick, 'discard', best_nfa(data, mod, in_b)

    return out_list
