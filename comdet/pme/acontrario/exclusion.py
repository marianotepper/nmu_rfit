import numpy as np
import operator
from . import utils


def exclusion_principle(ac_tester, models):
    mod_inliers_list = [(mod, ac_tester.inliers(mod)) for mod in models]
    nfa_list = [(i, ac_tester.nfa(mod)) for i, mod in enumerate(models)]
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
        if not ac_tester.meaningful(mod, data=ac_tester.data[considered]):
            keep_list.remove(pick)

    return keep_list
