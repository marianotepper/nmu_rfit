# coding: utf-8
import numpy as np
import scipy.sparse as sp
import sklearn.utils.linear_assignment_ as hungarian


def nmi(groups1, groups2):
    """
    Compute the normalised mutual information between two lists
    Depending on the sizes of the lists the implementation will use a full
    or a sparse representation to optimise speed and memory allocation.

    :param a1: first list of groups
    :param a2: second list of groups
    :return: normalised mutual information between the two lists
    """
    if not groups1 or not groups2:
        return 0

    c = confusion_matrix(groups1, groups2)
    n = groups1[0].shape[0]

    sumc0 = c.sum(axis=0)
    sumc1 = c.sum(axis=1)

    if sp.issparse(c):
        sumc0 = sumc0.A.squeeze()
        sumc1 = sumc1.A.squeeze()
        i, j, v = sp.find(c)
        temp = v.copy()
        temp = np.log(temp)
        temp += np.log(n)
        temp -= np.log(sumc0[j])
        temp -= np.log(sumc1[i])
        temp = c * v
    else:
        mask = [c > 0]
        temp = c.copy()
        temp[mask] = np.log(c[mask])
        temp += np.log(n)
        temp -= np.log(np.atleast_2d(sumc0))
        temp -= np.log(np.atleast_2d(sumc1).T)
        temp = c * temp

    term1 = temp.sum()
    term2 = np.sum(sumc1 * (np.log(sumc1) - np.log(n)))
    term3 = np.sum(sumc0 * (np.log(sumc0) - np.log(n)))

    return (-2 * term1) / (term2 + term3)


def confusion_matrix(groups1, groups2):
    # Number of values in each list
    nc1 = len(groups1)
    nc2 = len(groups2)

    sparse_conf = nc1 * nc2 > 1e6
    # Build the confusion matrix
    if sparse_conf:
        conf = sp.lil_matrix((nc1, nc2))
    else:
        conf = np.zeros((nc1, nc2))
    for i, ci in enumerate(groups1):
        for j, cj in enumerate(groups2):
            conf[i, j] = intersect_size(ci, cj)
    return conf


def plogp(p):
    if p == 0:
        return 0
    else:
        return p * np.log(p)


def entropy(c, n):
    p = size(c) / n
    if p == 0 or p == 1:
        return 0
    else:
        return -plogp(p) - plogp(1-p)


def entropy_per_group(group_list):
    h = np.zeros((len(group_list),))
    for i, c in enumerate(group_list):
        h[i] = entropy(c, c.shape[0])
    return h


def gnmi(groups1, groups2):
    """
    Compute the generalised normalised mutual information between two
    sets of overlapping communities as defined in (Lancichinetti, Fortunato,
    Kertész; Detecting the overlapping and hierarchical community structure
    in complex networks, New Journal of Physics, 2009)

    :param groups1: first list of groups
    :param groups2: second list of groups
    :return: normalised mutual information
    """
    if not groups1 or not groups2:
        return 0

    n = float(groups1[0].shape[0])
    h1 = entropy_per_group(groups1)
    h2 = entropy_per_group(groups2)

    # H(Xk|Yl)
    tested1 = np.zeros((len(groups1),), dtype=bool)
    tested2 = np.zeros((len(groups2),), dtype=bool)
    min_h12 = np.full((len(groups1),), np.inf)
    min_h21 = np.full((len(groups2),), np.inf)
    for i1, c1 in enumerate(groups1):
        lc1 = size(c1)
        for i2, c2 in enumerate(groups2):
            lc2 = size(c2)
            l12 = intersect_size(c1, c2)

            ent = np.array([[-plogp((n - lc1 - lc2 + l12) / n),
                             -plogp((lc1 - l12) / n)],
                            [-plogp((lc2 - l12) / n), -plogp(l12 / n)]])

            if np.trace(ent) > np.trace(np.fliplr(ent)):
                h12 = np.sum(ent) - h2[i2]
                h21 = np.sum(ent) - h1[i1]

                tested1[i1] = True
                min_h12[i1] = min([min_h12[i1], h12])
                tested2[i2] = True
                min_h21[i2] = min([min_h21[i2], h21])

    # H(Xk|Y) norm summed
    mask = np.logical_and(tested1, h1 > 0)
    sum_h12 = np.sum(min_h12[mask] / h1[mask]) + np.sum(np.logical_not(mask))
    sum_h12 /= len(groups1)

    # H(Yk|X) norm summed
    mask = np.logical_and(tested2, h2 > 0)
    sum_h21 = np.sum(min_h21[mask] / h2[mask]) + np.sum(np.logical_not(mask))
    sum_h21 /= len(groups2)

    # N(X|Y)
    return 1 - (sum_h12 + sum_h21)/2


def intersect_size(c1, c2):
    if sp.issparse(c1):
        c1 = c1.toarray()
    if sp.issparse(c2):
        c2 = c2.toarray()
    return float(np.logical_and(c1, c2).sum())


def size(c):
    return float(c.sum())


def mean_precision_recall(gt_groups, groups):
    """
    Compute the precision and recall.

    :param gt_groups: set of ground truth groups
    :param groups: set of tested groups
    :return: precision and recall
    """
    if not groups:
        return 0, 0
    conf = confusion_matrix(gt_groups, groups)
    idx = hungarian.linear_assignment(conf.max() - conf)
    conf = conf.take(idx[:, 0], axis=0).take(idx[:, 1], axis=1)
    precision = np.trace(conf) / sum([size(c) for c in groups])
    recall = np.trace(conf) / sum([size(c) for c in gt_groups])
    return precision, recall
