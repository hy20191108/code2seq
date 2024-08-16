from typing import List

from code2seq.data.predict_name import PredictName


class Method:
    def __init__(self) -> None:
        self._name = ""
        self._predict_name_list: List[PredictName] = []
        self._top_scores = []

    def append(self, predict_name: PredictName):
        self._predict_name_list.append(predict_name)

    def set_name(self, name: str):
        self._name = name

    def set_top_scores(self, top_scores):
        self._top_scores = top_scores

    @property
    def name(self):
        return self._name

    @property
    def predict_name_list(self):
        return self._predict_name_list
