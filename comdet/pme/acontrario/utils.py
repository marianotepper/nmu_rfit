import itertools


def ifilter(ac_tester, model_generator):
    def inner_meaningful(model):
        return ac_tester.meaningful(model)
    return itertools.ifilter(inner_meaningful, model_generator)
