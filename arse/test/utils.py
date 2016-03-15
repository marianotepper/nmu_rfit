from __future__ import absolute_import
import sys
import numpy as np
import arse.biclustering.utils as bic_utils
import arse.pme.acontrario as ac
import arse.test.measures as mes


def compute_measures(gt_groups, left_factors, verbose=True):
    gnmi = mes.gnmi(gt_groups, left_factors)
    prec, rec = mes.mean_precision_recall(gt_groups, left_factors)
    measures_str = 'GNMI: {0:1.3f}; Precision: {1:1.3f}; Recall: {2:1.3f}'
    if verbose:
        print measures_str.format(gnmi, prec, rec)
    return gnmi, prec, rec


def compute_stats(stats, verbose=True):
    def inner_print(attr):
        try:
            vals = [s[attr.lower()] for s in stats]
            val_str = attr.capitalize() + ' -> '
            val_str += 'mean: {mean:1.3f}, '
            val_str += 'std: {std:1.3f}, '
            val_str += 'median: {median:1.3f}'
            summary = {'mean': np.mean(vals), 'std': np.std(vals),
                       'median': np.median(vals)}
            if verbose:
                print(val_str.format(**summary))
            return summary
        except KeyError:
            return {}

    measures = ['Time', 'GNMI', 'Precision', 'Recall']
    global_summary = {}
    for m in measures:
        global_summary[m] = inner_print(m)
    return global_summary


def clean(model_class, x, thresholder, ac_tester, bic_list, share_elements=True):
    min_sample_size = model_class().min_sample_size
    bic_list = [bic for bic in bic_list
                if bic[1].nnz > 1 and bic[0].nnz >= min_sample_size]

    inliers_list = []
    model_list = []
    for lf, _ in bic_list:
        inliers = np.squeeze(lf.toarray())
        mod = model_class(x[inliers])
        model_list.append(mod)
        inliers = thresholder.membership(mod, x)
        inliers_list.append(inliers)

    if not model_list:
        return [], []

    # filter out non-meaningful groups
    inliers_list, model_list, bic_list = meaningful(ac_tester, inliers_list,
                                                    model_list, bic_list)

    keep = ac.exclusion_principle(x, thresholder, ac_tester, inliers_list,
                                  model_list)

    inliers_list = [inliers_list[s] for s in keep]
    model_list = [model_list[s] for s in keep]
    bic_list = [bic_list[s] for s in keep]
    if not share_elements:
        solve_intersections(x, inliers_list, model_list)
        inliers_list, model_list, bic_list = meaningful(ac_tester, inliers_list,
                                                        model_list, bic_list)

    bic_list = inliers_to_left_factors(inliers_list, bic_list)

    return model_list, bic_list


def meaningful(ac_tester, inliers_list, model_list, bic_list):
    z_list = zip(inliers_list, model_list, bic_list)
    z_list = filter(lambda e: ac_tester.meaningful(e[0]), z_list)
    if z_list:
        return zip(*z_list)
    else:
        return [], [], []


def solve_intersections(x, inliers_list, model_list):
    intersection = np.sum(np.vstack(inliers_list) > 0, axis=0) > 1
    dists = [mod.distances(x[intersection, :]) for mod in model_list]
    idx = np.argmin(np.vstack(dists), axis=0)
    for i, inliers in enumerate(inliers_list):
        inliers[intersection] = idx == i


def inliers_to_left_factors(inliers_list, bic_list):
    left_factors = [bic_utils.sparse(inliers[:, np.newaxis] > 0,
                                     dtype=np.bool)
                    for inliers in inliers_list]
    return [(lf, rf) for lf, (_, rf) in zip(left_factors, bic_list)]


class Logger(object):
    def __init__(self, filename="Console.log"):
        self.stdout = sys.stdout
        self.log = open(filename, "w")

    def __del__(self):
        self.log.close()

    def close(self):
        self.log.close()

    def write(self, message):
        self.stdout.write(message)
        self.log.write(message)
        self.log.flush()

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)
