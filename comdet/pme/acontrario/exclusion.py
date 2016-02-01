import numpy as np
import operator
from . import utils


def exclusion_principle(ac_tester, models):
    nfa_list = [(i, ac_tester.nfa(mod)) for i, mod in enumerate(models)]
    nfa_list = filter(lambda e: e[1] < ac_tester.epsilon, nfa_list)
    nfa_list = sorted(nfa_list, key=operator.itemgetter(1))
    candidates = zip(*nfa_list)[0]

    inliers_list = [ac_tester.inliers(mod) for mod in models]
    keep_list = []
    for cand in candidates:
        cand_mod = models[cand]
        in_list = [inliers_list[other] for other in keep_list]
        if not in_list:
            keep_list.append(cand)
            continue
        excluded = reduce(lambda x, y: np.logical_or(x, y), in_list)
        considered = np.logical_not(excluded)
        if ac_tester.meaningful(cand_mod, considered=considered):
            keep_list.append(cand)
    return keep_list