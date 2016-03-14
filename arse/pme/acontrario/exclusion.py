import numpy as np


def exclusion_principle(x, thresholder, ac_tester, inliers_list, model_list):
    keep_list = []
    excluded = None
    for k, (inliers, mod) in enumerate(zip(inliers_list, model_list)):
        if excluded is None:
            keep_list.append(k)
            excluded = inliers > 0
            continue

        considered = np.logical_not(excluded)
        membership = thresholder.membership(mod, x[considered])
        if ac_tester.meaningful(membership):
            keep_list.append(k)
            excluded = np.logical_or(excluded, inliers > 0)

    return keep_list
