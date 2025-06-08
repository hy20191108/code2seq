import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

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
    def __init__(self, original_name: str) -> None:
        self.original_name = original_name
        self.predictions: List[SingleTimeStepPrediction] = list()

    def append_prediction(
        self, name: str, current_timestep_paths: Optional[List[Any]]
    ) -> None:
        self.predictions.append(SingleTimeStepPrediction(name, current_timestep_paths))


class SingleTimeStepPrediction:
    def __init__(self, prediction: str, attention_paths: Optional[List[Any]]) -> None:
        self.prediction = prediction
        if attention_paths is not None:
            paths_with_scores = []
            for (
                attention_score,
                vector,
                path_context_info,
                source_vector,
                target_vector,
                astpath_vector,
            ) in attention_paths:
                path_context_dict = {
                    "score": attention_score,
                    "vector": vector,
                    "path": path_context_info.longPath,
                    "source": path_context_info.source,
                    "target": path_context_info.target,
                    "source_vector": source_vector,
                    "target_vector": target_vector,
                    "astpath_vector": astpath_vector,
                }
                paths_with_scores.append(path_context_dict)
            self.attention_paths = paths_with_scores


class PathContextInformation:
    def __init__(self, context: Dict[str, Any]) -> None:
        self.source = context["name1"]
        self.longPath = context["path"]
        self.shortPath = context["shortPath"]
        self.target = context["name2"]
        self.source_begin = context["name1Begin"]
        self.source_end = context["name1End"]
        self.target_begin = context["name2Begin"]
        self.target_end = context["name2End"]

    @property
    def lineColumns(self) -> Tuple[int, int, int, int, int, int, int, int]:
        return (
            self.source_begin["line"],
            self.source_begin["column"],
            self.source_end["line"],
            self.source_end["column"],
            self.target_begin["line"],
            self.target_begin["column"],
            self.target_end["line"],
            self.target_end["column"],
        )

    def __str__(self) -> str:
        return f"{self.source},{self.shortPath},{self.target}"

    # 後方互換性のためのプロパティ
    @property
    def token1(self) -> str:
        return self.source

    @property
    def token2(self) -> str:
        return self.target


class Common:
    internal_delimiter = "|"
    SOS = "<S>"
    EOS = "</S>"
    PAD = "<PAD>"
    UNK = "<UNK>"

    @staticmethod
    def normalize_word(word: str) -> str:
        stripped = re.sub(r"[^a-zA-Z]", "", word)
        if len(stripped) == 0:
            return word.lower()
        else:
            return stripped.lower()

    @staticmethod
    def load_histogram(path: str, max_size: Optional[int] = None) -> Dict[str, int]:
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
    def load_vocab_from_dict(
        word_to_count: Dict[str, int],
        add_values: List[str] = [],
        max_size: Optional[int] = None,
    ) -> Tuple[Dict[str, int], Dict[int, str], int]:
        word_to_index, index_to_word = {}, {}
        current_index = 0
        for value in add_values:
            word_to_index[value] = current_index
            index_to_word[current_index] = value
            current_index += 1
        sorted_counts = [
            (k, word_to_count[k])
            for k in sorted(
                word_to_count.keys(), key=lambda x: word_to_count[x], reverse=True
            )
        ]
        limited_sorted = dict(sorted_counts[:max_size])
        for word, count in limited_sorted.items():
            word_to_index[word] = current_index
            index_to_word[current_index] = word
            current_index += 1
        return word_to_index, index_to_word, current_index

    @staticmethod
    def binary_to_string(binary_string: bytes) -> str:
        return binary_string.decode("utf-8")

    @staticmethod
    def binary_to_string_list(binary_string_list: List[bytes]) -> List[str]:
        return [Common.binary_to_string(w) for w in binary_string_list]

    @staticmethod
    def binary_to_string_matrix(
        binary_string_matrix: List[List[bytes]],
    ) -> List[List[str]]:
        return [Common.binary_to_string_list(l) for l in binary_string_matrix]

    @staticmethod
    def binary_to_string_3d(
        binary_string_tensor: List[List[List[bytes]]],
    ) -> List[List[List[str]]]:
        return [Common.binary_to_string_matrix(l) for l in binary_string_tensor]

    @staticmethod
    def legal_method_names_checker(name: str) -> bool:
        return name not in [Common.UNK, Common.PAD, Common.EOS]

    @staticmethod
    def filter_impossible_names(top_words: List[str]) -> List[str]:
        result = list(filter(Common.legal_method_names_checker, top_words))
        return result

    @staticmethod
    def unique(sequence: List[Any]) -> List[Any]:
        return list(set(sequence))

    @staticmethod
    def parse_results(
        code: Code, pc_info_dict: Dict[Any, Any]
    ) -> List[PredictionResults]:
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
                        # path_contextはPathContextオブジェクトで渡される（model.pyのget_method関数で作成）
                        key = (
                            path_context.source,
                            path_context.path,
                            path_context.target,
                        )
                        if key in pc_info_dict:
                            pc_info = pc_info_dict[key]
                            current_timestep_paths.append(
                                (
                                    path_context.attention,
                                    path_context.vector,
                                    pc_info,
                                    path_context.source_vector,
                                    path_context.target_vector,
                                    path_context.astpath_vector,
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
    def compute_bleu(ref_file_name: str, predicted_file_name: str) -> None:
        with open(predicted_file_name) as predicted_file:
            pipe = subprocess.Popen(
                ["perl", "scripts/multi-bleu.perl", ref_file_name],
                stdin=predicted_file,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
