from __future__ import absolute_import
import sys
import numpy as np
import comdet.pme.acontrario as ac
import comdet.test.measures as mes


def compute_measures(gt_groups, left_factors, verbose=True):
    gnmi = mes.gnmi(gt_groups, left_factors)
    prec, rec = mes.mean_precision_recall(gt_groups, left_factors)
    measures_str = 'GNMI: {0:1.3f}; Precision: {1:1.3f}; Recall: {2:1.3f}'
    if verbose:
        print measures_str.format(gnmi, prec, rec)
    return gnmi, prec, rec


def print_stats(stats):
    def inner_print(attr):
        try:
            vals = [s[attr.lower()] for s in stats]
            val_str = attr.capitalize() + ' -> '
            val_str += 'mean: {0:1.3f}, '
            val_str += 'std: {1:1.3f}, '
            val_str += 'median: {2:1.3f}'
            print(val_str.format(np.mean(vals), np.std(vals), np.median(vals)))
        except KeyError:
            pass

    inner_print('time')
    inner_print('GNMI')
    inner_print('Precision')
    inner_print('Recall')


def clean(model_class, x, ac_tester, bic_list):
    bic_list = [bic for bic in bic_list
                if bic[1].nnz > 1 and
                bic[0].nnz > model_class().min_sample_size]

    models = []
    for lf, _ in bic_list:
        inliers = np.squeeze(lf.toarray())
        mod = model_class(x[inliers])
        models.append(mod)

    survivors = ac.exclusion_principle(ac_tester, models)

    models = [models[s] for s in survivors]
    bic_list = [bic_list[s] for s in survivors]
    return models, bic_list


class Logger(object):
    def __init__(self, filename="Console.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)
