from typing import List

from data.path_context import PathContext


class PredictName:
    def __init__(self):
        self._predicted_name = None
        self._path_context_list: List[PathContext] = []

    def append(self, path_context):
        self._path_context_list.append(path_context)

    def set_predicted_name(self, name):
        self._predicted_name = name

    @property
    def predicted_name(self):
        return self._predicted_name

    @property
    def path_context_list(self):
        return self._path_context_list
