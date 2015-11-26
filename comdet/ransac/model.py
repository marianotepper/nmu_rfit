from abc import ABCMeta, abstractmethod, abstractproperty


class Model(object):
    __metaclass__ = ABCMeta

    @abstractproperty
    def min_sample_size(self):
        pass

    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def distances(self, data):
        pass

    @abstractmethod
    def plot(self, **kwargs):
        pass
