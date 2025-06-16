"""
辞書変換修正テスト

common.pyのSingleTimeStepPredictionで発生するnumpy配列→辞書変換の
問題を検証し、修正案をテストします。
"""

import pickle
from typing import Any, List, Optional

import numpy as np


# 問題のあるオリジナル実装（simplified version）
class OriginalSingleTimeStepPrediction:
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
                # 問題：numpy配列をそのまま辞書に格納
                path_context_dict = {
                    "score": attention_score,
                    "vector": vector,  # ← numpy.ndarray
                    "source": getattr(path_context_info, "source", "test_source"),
                    "target": getattr(path_context_info, "target", "test_target"),
                    "path": getattr(path_context_info, "longPath", "test_path"),
                    "source_vector": source_vector,
                    "target_vector": target_vector,
                    "astpath_vector": astpath_vector,
                }
                paths_with_scores.append(path_context_dict)
            self.attention_paths = paths_with_scores


# 修正されたバージョン（numpy配列を保持）
class FixedSingleTimeStepPrediction:
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
                # 修正：numpy配列の型を明示的に保持
                path_context_dict = {
                    "score": attention_score,
                    "vector": np.asarray(vector),  # ← 明示的にnumpy配列として保持
                    "source": getattr(path_context_info, "source", "test_source"),
                    "target": getattr(path_context_info, "target", "test_target"),
                    "path": getattr(path_context_info, "longPath", "test_path"),
                    "source_vector": np.asarray(source_vector)
                    if source_vector is not None
                    else None,
                    "target_vector": np.asarray(target_vector)
                    if target_vector is not None
                    else None,
                    "astpath_vector": np.asarray(astpath_vector)
                    if astpath_vector is not None
                    else None,
                }
                paths_with_scores.append(path_context_dict)
            self.attention_paths = paths_with_scores


class MockPathContextInfo:
    """テスト用のPathContextInfo"""

    def __init__(self, source="test_source", target="test_target", path="test_path"):
        self.source = source
        self.target = target
        self.longPath = path


def create_test_data():
    """テスト用のダミーデータを作成"""
    # numpy配列ベクトルを作成
    vector = np.random.random(320).astype(np.float32)
    source_vector = np.random.random(128).astype(np.float32)
    target_vector = np.random.random(128).astype(np.float32)
    astpath_vector = np.random.random(256).astype(np.float32)

    # PathContextInfo
    pc_info = MockPathContextInfo()

    # attention_pathsデータ
    attention_paths = [
        (
            0.85,  # attention_score
            vector,  # vector
            pc_info,  # path_context_info
            source_vector,  # source_vector
            target_vector,  # target_vector
            astpath_vector,  # astpath_vector
        )
    ]

    return attention_paths, vector, source_vector, target_vector, astpath_vector


def test_original_implementation():
    """オリジナル実装での型変換問題を検証"""
    print("=== オリジナル実装テスト ===")

    attention_paths, vector, source_vector, target_vector, astpath_vector = (
        create_test_data()
    )

    # オリジナル実装でオブジェクト作成
    original = OriginalSingleTimeStepPrediction("test_prediction", attention_paths)

    # pickle直列化/逆直列化
    serialized = pickle.dumps(original)
    deserialized = pickle.loads(serialized)

    # 型チェック
    path_context_dict = deserialized.attention_paths[0]

    print(f"vector型: {type(path_context_dict['vector'])}")
    print(f"source_vector型: {type(path_context_dict['source_vector'])}")
    print(f"target_vector型: {type(path_context_dict['target_vector'])}")
    print(f"astpath_vector型: {type(path_context_dict['astpath_vector'])}")

    # 型変換されているかチェック
    vector_is_numpy = isinstance(path_context_dict["vector"], np.ndarray)
    source_vector_is_numpy = isinstance(path_context_dict["source_vector"], np.ndarray)

    print(f"vector is numpy: {vector_is_numpy}")
    print(f"source_vector is numpy: {source_vector_is_numpy}")

    return vector_is_numpy, source_vector_is_numpy


def test_fixed_implementation():
    """修正実装での型保持を検証"""
    print("\n=== 修正実装テスト ===")

    attention_paths, vector, source_vector, target_vector, astpath_vector = (
        create_test_data()
    )

    # 修正実装でオブジェクト作成
    fixed = FixedSingleTimeStepPrediction("test_prediction", attention_paths)

    # pickle直列化/逆直列化
    serialized = pickle.dumps(fixed)
    deserialized = pickle.loads(serialized)

    # 型チェック
    path_context_dict = deserialized.attention_paths[0]

    print(f"vector型: {type(path_context_dict['vector'])}")
    print(f"source_vector型: {type(path_context_dict['source_vector'])}")
    print(f"target_vector型: {type(path_context_dict['target_vector'])}")
    print(f"astpath_vector型: {type(path_context_dict['astpath_vector'])}")

    # 型が保持されているかチェック
    vector_is_numpy = isinstance(path_context_dict["vector"], np.ndarray)
    source_vector_is_numpy = isinstance(path_context_dict["source_vector"], np.ndarray)

    print(f"vector is numpy: {vector_is_numpy}")
    print(f"source_vector is numpy: {source_vector_is_numpy}")

    return vector_is_numpy, source_vector_is_numpy


def test_dict_access_compatibility():
    """辞書アクセスの互換性をテスト"""
    print("\n=== 辞書アクセス互換性テスト ===")

    attention_paths, vector, source_vector, target_vector, astpath_vector = (
        create_test_data()
    )

    # 修正実装でオブジェクト作成
    fixed = FixedSingleTimeStepPrediction("test_prediction", attention_paths)

    # pickle直列化/逆直列化
    serialized = pickle.dumps(fixed)
    deserialized = pickle.loads(serialized)

    # eye2vecのcontext_model.pyと同様の辞書アクセスをテスト
    path_context = deserialized.attention_paths[0]

    try:
        accessed_vector = path_context["vector"]
        print(f"✅ path_context['vector'] 成功: {type(accessed_vector)}")

        # tobytes()メソッドのテスト（eye2vecで使用）
        if hasattr(accessed_vector, "tobytes"):
            vector_bytes = accessed_vector.tobytes()
            print(f"✅ vector.tobytes() 成功: {len(vector_bytes)} bytes")
        else:
            print("❌ vector.tobytes() 失敗: tobytesBattribute not found")

        return True

    except Exception as e:
        print(f"❌ 辞書アクセス失敗: {e}")
        return False


def main():
    """メイン実行関数"""
    print("辞書変換修正テスト実行中...")

    # オリジナル実装テスト
    orig_vector_numpy, orig_source_numpy = test_original_implementation()

    # 修正実装テスト
    fixed_vector_numpy, fixed_source_numpy = test_fixed_implementation()

    # 辞書アクセステスト
    dict_access_success = test_dict_access_compatibility()

    print("\n=== テスト結果サマリー ===")
    print(f"オリジナル実装 - vector is numpy: {orig_vector_numpy}")
    print(f"オリジナル実装 - source_vector is numpy: {orig_source_numpy}")
    print(f"修正実装 - vector is numpy: {fixed_vector_numpy}")
    print(f"修正実装 - source_vector is numpy: {fixed_source_numpy}")
    print(f"辞書アクセス互換性: {dict_access_success}")

    if fixed_vector_numpy and fixed_source_numpy and dict_access_success:
        print("\n🎉 修正案は有効です！")
        return True
    else:
        print("\n❌ 修正案に問題があります。")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
