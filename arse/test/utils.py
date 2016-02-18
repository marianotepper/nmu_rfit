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


def clean(model_class, x, ac_tester, bic_list):
    bic_list = [bic for bic in bic_list
                if bic[1].nnz > 1 and
                bic[0].nnz > model_class().min_sample_size]

    bic_list_new = []
    models = []
    for lf, rf in bic_list:
        inliers = np.squeeze(lf.toarray())
        mod = model_class(x[inliers])
        models.append(mod)
        inliers = np.atleast_2d(ac_tester.inliers(mod)).T
        bic_list_new.append((bic_utils.sparse(inliers), rf))

    if models:
        survivors = ac.exclusion_principle(ac_tester, models)
        models = [models[s] for s in survivors]
        bic_list_new = [bic_list_new[s] for s in survivors]

    return models, bic_list_new


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
