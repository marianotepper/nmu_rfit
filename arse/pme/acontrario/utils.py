import itertools


def ifilter(model_generator, thresholder, ac_tester):
    def inner_meaningful(model):
        membership = thresholder.membership(model, model_generator.elements)
        return ac_tester.meaningful(membership, model.min_sample_size)
    return itertools.ifilter(inner_meaningful, model_generator)
