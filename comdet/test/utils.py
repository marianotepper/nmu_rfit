from __future__ import absolute_import
import numpy as np
import comdet.biclustering.preference as pref
import comdet.pme.acontrario as ac
import comdet.pme.measures as mes


def build_preference_matrix(n_elements, ransac_gen, ac_tester):
    filtered = ac.ifilter(ac_tester, ransac_gen)

    pref_matrix = pref.create_preference_matrix(n_elements)
    original_models = []
    for i, (model, inliers) in enumerate(filtered):
        pref_matrix = pref.add_col(pref_matrix, inliers)
        original_models.append(model)

    return pref_matrix, original_models


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
