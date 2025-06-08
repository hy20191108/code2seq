from typing import Tuple

import numpy as np


class PathContext:
    def __init__(
        self,
        source: str,
        short_path: str,
        target: str,
        attention: np.float32,
        vector: np.ndarray,
        source_vector: np.ndarray,
        target_vector: np.ndarray,
        astpath_vector: np.ndarray,
    ) -> None:
        self._source = source
        self._short_path = short_path
        self._target = target
        self._attention: np.float32 = attention
        self._vector = vector
        self._source_vector = source_vector
        self._target_vector = target_vector
        self._astpath_vector = astpath_vector

    def get_key(self) -> Tuple[str, str, str]:
        return (self._source, self._short_path, self._target)

    @property
    def source(self) -> str:
        return self._source

    @property
    def short_path(self) -> str:
        return self._short_path

    @property
    def target(self) -> str:
        return self._target

    @property
    def attention(self) -> np.float32:
        return self._attention

    @property
    def vector(self) -> np.ndarray:
        return self._vector

    @property
    def source_vector(self) -> np.ndarray:
        return self._source_vector

    @property
    def target_vector(self) -> np.ndarray:
        return self._target_vector

    @property
    def astpath_vector(self) -> np.ndarray:
        return self._astpath_vector

    @property
    def path(self) -> str:
        """テスト互換性用のエイリアス"""
        return self._short_path

    @property
    def lineColumns(self) -> str:
        """テスト互換性用の空文字列"""
        return ""

    def validate_vectors(self) -> None:
        """ベクトルの整合性検証"""
        expected_dims = {
            "vector": 320,
            "source_vector": 128,
            "target_vector": 128,
            "astpath_vector": 256,
        }

        for name, expected_dim in expected_dims.items():
            vec = getattr(self, name)
            if vec is None:
                raise ValueError(f"{name} is None")
            if not isinstance(vec, np.ndarray):
                raise TypeError(f"{name} must be numpy array, got {type(vec)}")
            if vec.shape != (expected_dim,):
                raise ValueError(
                    f"{name} dimension mismatch: expected ({expected_dim},), got {vec.shape}"
                )
