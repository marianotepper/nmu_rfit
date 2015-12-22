from __future__ import absolute_import
import sys
import numpy as np
import comdet.pme.acontrario as ac
import comdet.pme.measures as mes


def compute_measures(gt_groups, left_factors):
    gnmi = mes.gnmi(gt_groups, left_factors)
    prec, rec = mes.mean_precision_recall(gt_groups, left_factors)
    measures_str = 'GNMI: {0:1.3f}; Precision: {1:1.3f}; Recall: {2:1.3f}'
    print measures_str.format(gnmi, prec, rec)


def clean(model_class, x, ac_tester, bic_list, restimate=True):
    bic_list = [bic for bic in bic_list
                if bic[1].nnz > 1 and
                bic[0].nnz > model_class().min_sample_size]

    mod_inliers_list = []
    for rf, _ in bic_list:
        inliers = np.squeeze(rf.toarray())
        mod = model_class(x[inliers])
        if restimate:
            inliers = mod.distances(x) <= ac_tester.threshold(mod)
        mod_inliers_list.append((mod, inliers))

    survivors = ac.exclusion_principle(ac_tester, mod_inliers_list)

    mod_inliers_list = [mod_inliers_list[s] for s in survivors]
    bic_list = [bic_list[s] for s in survivors]
    return mod_inliers_list, bic_list


class Logger(object):
    def __init__(self, filename="Console.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)
