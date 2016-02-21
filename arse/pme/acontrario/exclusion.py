import numpy as np
import operator


def exclusion_principle(x, thresholder, ac_tester, inliers_list, model_list):
    nfa_list = [ac_tester.nfa(inliers, mod.min_sample_size)
                for inliers, mod in zip(inliers_list, model_list)]
    nfa_list = list(enumerate(nfa_list))
    nfa_list = filter(lambda e: e[1] < ac_tester.epsilon, nfa_list)
    nfa_list = sorted(nfa_list, key=operator.itemgetter(1))
    candidates = zip(*nfa_list)[0]

    keep_list = []
    for cand in candidates:
        cand_mod = model_list[cand]
        in_list = [np.nan_to_num(inliers_list[other]) for other in keep_list]
        if not in_list:
            keep_list.append(cand)
            continue
        excluded = reduce(lambda x, y: np.logical_or(x, y), in_list)
        considered = np.logical_not(excluded)

        membership = thresholder.membership(cand_mod, x[considered])
        if ac_tester.meaningful(membership, cand_mod.min_sample_size):
            keep_list.append(cand)
    return keep_list
