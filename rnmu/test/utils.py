from __future__ import absolute_import
import sys
import numpy as np
import itertools as itt
import rnmu.test.measures as mes


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
