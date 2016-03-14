import itertools


def ifilter(model_generator, thresholder, ac_tester):
    def inner_meaningful(model):
        membership = thresholder.membership(model, model_generator.elements)
        return ac_tester.meaningful(membership)
    return itertools.ifilter(inner_meaningful, model_generator)
