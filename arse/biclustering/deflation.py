from __future__ import absolute_import
from . import compression
from . import utils


class DeflationError(RuntimeError):
    def __init__(self, *args, **kwargs):
        super(DeflationError, self).__init__(*args, **kwargs)


class Deflator(utils.Downdater):
    def __init__(self, array):
        super(Deflator, self).__init__(array)

    @property
    def compressed_array(self):
        raise DeflationError('Could not compress the array.')


class L1CompressedDeflator(Deflator):
    def __init__(self, array, n_samples):
        super(L1CompressedDeflator, self).__init__(array)
        if n_samples >= array.shape[1]:
            self._compressor = DummyCompressor(array, n_samples)
        else:
            self._compressor = compression.OnlineColumnCompressor(array,
                                                                  n_samples)
            self._inner_compress()

    def _inner_compress(self):
        selection = self._compressor.compress()
        if selection is None or self.n_samples > selection.size:
            try:
                del self._selection
                del self._compressed_array
            except AttributeError:
                pass
        else:
            self._selection = selection
            self._compressed_array = self.array[:, self.selection]

    @property
    def compressed_array(self):
        try:
            return self._compressed_array
        except AttributeError:
            raise DeflationError('Could not compress the array.')

    @property
    def selection(self):
        try:
            return self._selection
        except AttributeError:
            raise DeflationError('Could not compress the array.')

    @property
    def n_samples(self):
        return self._compressor.n_samples

    def additive_downdate(self, u, v):
        super(L1CompressedDeflator, self).additive_downdate(u, v)
        self._compressor.additive_downdate(u, v)
        self._inner_compress()

    def remove_columns(self, idx_cols):
        super(L1CompressedDeflator, self).remove_columns(idx_cols)
        self._compressor.remove_columns(idx_cols)
        self._inner_compress()

    def remove_rows(self, idx_rows):
        super(L1CompressedDeflator, self).remove_rows(idx_rows)
        self._compressor.remove_rows(idx_rows)
        self._inner_compress()


class DummyCompressor(object):
    def __init__(self, array, n_samples):
        self.n_samples = n_samples

    def compress(self):
        return None

    def additive_downdate(self, u, v):
        pass

    def remove_columns(self, idx):
        pass

    def remove_rows(self, idx):
        pass
