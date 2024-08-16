import numpy as np


class PathContext:
    def __init__(self, source, short_path, target, attention, vector):
        self._source = source
        self._short_path = short_path
        self._target = target
        self._attention: np.float32 = attention
        self._vector = vector

    def get_key(self):
        return (self._source, self._short_path, self._target)

    @property
    def source(self):
        return self._source

    @property
    def short_path(self):
        return self._short_path

    @property
    def target(self):
        return self._target

    @property
    def attention(self) -> np.float32:
        return self._attention

    @property
    def vector(self):
        return self._vector
