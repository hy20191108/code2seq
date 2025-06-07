import re
import subprocess
import sys
from typing import Dict, List

import numpy as np

from data.code import Code


class ContextInfo:
    def __init__(
        self,
        token1: str,
        path: str,
        token2: str,
        attention_score: float,
        vector: np.ndarray,
    ) -> None:
        self._token1 = token1
        self._astpath = path
        self._token2 = token2
        self._attention_score = attention_score
        self._vector = vector

    @property
    def attention_score(self) -> float:
        return self._attention_score

    @property
    def vector(self) -> np.ndarray:
        return self._vector


class PredictionResults:
    def __init__(self, original_name):
        self.original_name = original_name
        self.predictions: List[SingleTimeStepPrediction] = list()

    def append_prediction(self, name, current_timestep_paths):
        self.predictions.append(SingleTimeStepPrediction(name, current_timestep_paths))


class SingleTimeStepPrediction:
    def __init__(self, prediction, attention_paths):
        self.prediction = prediction
        if attention_paths is not None:
            paths_with_scores = []
            for attention_score, vector, pc_info in attention_paths:
                path_context_dict = {
                    "score": attention_score,
                    "vector": vector,
                    "path": pc_info.longPath,
                    "token1": pc_info.token1,
                    "token2": pc_info.token2,
                }
                paths_with_scores.append(path_context_dict)
            self.attention_paths = paths_with_scores


class PathContextInformation:
    def __init__(self, context):
        self.token1 = context["name1"]
        self.longPath = context["path"]
        self.shortPath = context["shortPath"]
        self.token2 = context["name2"]
        self.token1_begin = context["name1Begin"]
        self.token1_end = context["name1End"]
        self.token2_begin = context["name2Begin"]
        self.token2_end = context["name2End"]

    @property
    def lineColumns(self):
        return (
            self.token1_begin["line"],
            self.token1_begin["column"],
            self.token1_end["line"],
            self.token1_end["column"],
            self.token2_begin["line"],
            self.token2_begin["column"],
            self.token2_end["line"],
            self.token2_end["column"],
        )

    def __str__(self):
        return f"{self.token1},{self.shortPath},{self.token2}"


class Common:
    internal_delimiter = "|"
    SOS = "<S>"
    EOS = "</S>"
    PAD = "<PAD>"
    UNK = "<UNK>"

    @staticmethod
    def normalize_word(word):
        stripped = re.sub(r"[^a-zA-Z]", "", word)
        if len(stripped) == 0:
            return word.lower()
        else:
            return stripped.lower()

    @staticmethod
    def load_histogram(path, max_size=None):
        histogram = {}
        with open(path) as file:
            for line in file.readlines():
                parts = line.split(" ")
                if not len(parts) == 2:
                    continue
                histogram[parts[0]] = int(parts[1])
        sorted_histogram = [
            (k, histogram[k])
            for k in sorted(histogram, key=histogram.get, reverse=True)
        ]
        return dict(sorted_histogram[:max_size])

    @staticmethod
    def load_vocab_from_dict(word_to_count, add_values=[], max_size=None):
        word_to_index, index_to_word = {}, {}
        current_index = 0
        for value in add_values:
            word_to_index[value] = current_index
            index_to_word[current_index] = value
            current_index += 1
        sorted_counts = [
            (k, word_to_count[k])
            for k in sorted(word_to_count, key=word_to_count.get, reverse=True)
        ]
        limited_sorted = dict(sorted_counts[:max_size])
        for word, count in limited_sorted.items():
            word_to_index[word] = current_index
            index_to_word[current_index] = word
            current_index += 1
        return word_to_index, index_to_word, current_index

    @staticmethod
    def binary_to_string(binary_string: bytes):
        return binary_string.decode("utf-8")

    @staticmethod
    def binary_to_string_list(binary_string_list):
        return [Common.binary_to_string(w) for w in binary_string_list]

    @staticmethod
    def binary_to_string_matrix(binary_string_matrix):
        return [Common.binary_to_string_list(l) for l in binary_string_matrix]

    @staticmethod
    def binary_to_string_3d(binary_string_tensor):
        return [Common.binary_to_string_matrix(l) for l in binary_string_tensor]

    @staticmethod
    def legal_method_names_checker(name):
        return name not in [Common.UNK, Common.PAD, Common.EOS]

    @staticmethod
    def filter_impossible_names(top_words):
        result = list(filter(Common.legal_method_names_checker, top_words))
        return result

    @staticmethod
    def unique(sequence):
        return list(set(sequence))

    @staticmethod
    def parse_results(code: Code, pc_info_dict) -> Dict[int, PredictionResults]:
        prediction_results = []

        # method
        for single_method in code.method_list:
            current_method_prediction_results = PredictionResults(single_method.name)

            if len(single_method.predict_name_list) > 0:
                predict_name_list = single_method.predict_name_list

                predict_name_list = [
                    predict_name
                    for predict_name in single_method.predict_name_list
                    if Common.legal_method_names_checker(predict_name.predicted_name)
                ]

                # word
                for predict_name in predict_name_list[:1]:
                    # path_context
                    current_timestep_paths = []
                    for path_context in predict_name.path_context_list:
                        if path_context.get_key() in pc_info_dict:
                            pc_info = pc_info_dict[path_context.get_key()]
                            current_timestep_paths.append(
                                (
                                    path_context.attention.item(),
                                    path_context.vector,
                                    pc_info,
                                )
                            )

                    current_method_prediction_results.append_prediction(
                        predict_name.predicted_name, current_timestep_paths
                    )
            else:
                for predicted_seq in single_method.predict_name_list:
                    filtered_seq = [
                        word
                        for word in predicted_seq
                        if Common.legal_method_names_checker(word)
                    ]
                    current_method_prediction_results.append_prediction(
                        filtered_seq, None
                    )

                raise ValueError("Error in extracting paths")

            prediction_results.append(current_method_prediction_results)
        return prediction_results

    @staticmethod
    def compute_bleu(ref_file_name, predicted_file_name):
        with open(predicted_file_name) as predicted_file:
            pipe = subprocess.Popen(
                ["perl", "scripts/multi-bleu.perl", ref_file_name],
                stdin=predicted_file,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
