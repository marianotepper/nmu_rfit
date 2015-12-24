import numpy as np
import scipy.sparse as sp
import operator
from . import utils


def optimal_nfa(ac_tester, model, inliers, considered=None):
    if considered is None:
        data_considered = None
    else:
        if considered.sum() == inliers.sum():
            return -np.inf
        data_considered = ac_tester.data[considered]
    if sp.issparse(inliers):
        inliers = np.squeeze(inliers.toarray())
    if model.min_sample_size >= inliers.sum():
        return np.inf
    dist = np.abs(model.distances(ac_tester.data[inliers]))
    dist.sort()
    min_nfa = np.inf
    for k, s in enumerate(dist):
        if k + 1 < model.min_sample_size:
            continue
        if s < np.finfo(np.float32).resolution:
            continue
        nfa = ac_tester.nfa(model, inliers_threshold=s, data=data_considered)
        min_nfa = np.minimum(nfa, min_nfa)
    return min_nfa + np.log10(len(dist) - model.min_sample_size)


def exclusion_principle(ac_tester, models):
    mod_inliers_list = [(mod, ac_tester.inliers(mod)) for mod in models]
    # nfa_list = [(i, optimal_nfa(ac_tester, mod, in_a))
    #             for i, (mod, in_a) in enumerate(mod_inliers_list)]
    nfa_list = [(i, ac_tester.nfa(mod))
                for i, (mod, in_a) in enumerate(mod_inliers_list)]

    nfa_list = filter(lambda e: utils.meaningful(e[1], ac_tester.epsilon), nfa_list)
    nfa_list = sorted(nfa_list, key=operator.itemgetter(1))
    idx = zip(*nfa_list)[0]

    keep_list = list(idx)
    for pick in idx:
        mod, in_a = mod_inliers_list[pick]
        in_list = [mod_inliers_list[k][1] for k in keep_list if k < pick]
        if not in_list:
            continue
        excluded = reduce(lambda x, y: np.logical_or(x, y), in_list)
        considered = np.logical_not(excluded)
        inliers = in_a - np.logical_and(in_a, excluded)
        nfa = ac_tester.nfa(mod, data=ac_tester.data[considered])
        # nfa = optimal_nfa(ac_tester, mod, inliers, considered=considered)
        if not utils.meaningful(nfa, ac_tester.epsilon):
            keep_list.remove(pick)

    return keep_list
