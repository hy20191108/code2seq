"""
型変換追跡テストスイート

このテストモジュールは、code2seqシステムのデータフロー全体を通して
numpy配列がリストに変換される具体的なタイミングを特定します。

データフロー追跡ポイント：
1. TensorFlow実行直後（model.py）
2. PathContextオブジェクト作成時（interactive_predict.py）
3. pickle直列化前（flask_app.py）
4. pickle逆直列化後（flask_wrapper.py）
5. 辞書アクセス時（context_model.py）
"""

import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# プロジェクトルートをsys.pathに追加
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import _code2seq
from config import Config
from interactive_predict import InteractivePredictor
from model import Model


class TypeConversionTracker:
    """型変換の詳細な追跡を行うユーティリティクラス"""

    def __init__(self):
        self.trace_log: List[Dict[str, Any]] = []

    def log_vector_state(self, location: str, vector: Any, name: str) -> None:
        """ベクトルの状態をログに記録"""
        vector_info = {
            "location": location,
            "name": name,
            "type": type(vector).__name__,
            "is_numpy": isinstance(vector, np.ndarray),
            "is_list": isinstance(vector, list),
            "shape": getattr(vector, "shape", None),
            "length": len(vector) if hasattr(vector, "__len__") else None,
            "dtype": getattr(vector, "dtype", None),
            "first_5_values": self._get_first_values(vector, 5),
        }
        self.trace_log.append(vector_info)

        # 即座にログ出力
        print(f"[{location}] {name}:")
        print(f"  型: {vector_info['type']}")
        print(f"  numpy配列: {vector_info['is_numpy']}")
        print(f"  リスト: {vector_info['is_list']}")
        if vector_info["shape"]:
            print(f"  形状: {vector_info['shape']}")
        if vector_info["length"]:
            print(f"  長さ: {vector_info['length']}")
        if vector_info["first_5_values"]:
            print(f"  最初の5要素: {vector_info['first_5_values']}")
        print()

    def _get_first_values(self, vector: Any, count: int) -> List[Any]:
        """ベクトルの最初のいくつかの値を取得"""
        try:
            if isinstance(vector, np.ndarray):
                return vector.flat[:count].tolist()
            elif isinstance(vector, list):
                return vector[:count]
            else:
                return []
        except Exception:
            return []

    def print_conversion_summary(self) -> None:
        """変換の要約を表示"""
        print("\n=== 型変換トレース要約 ===")
        for entry in self.trace_log:
            conversion_status = (
                "🔥 LIST!"
                if entry["is_list"]
                else "✅ numpy"
                if entry["is_numpy"]
                else "❓ other"
            )
            print(
                f"{conversion_status} [{entry['location']}] {entry['name']}: {entry['type']}"
            )


class TestTypeConversionTracking:
    """型変換追跡テストクラス"""

    def __init__(self):
        self.predictor = self._initialize_predictor()
        self.tracker = TypeConversionTracker()

    def _initialize_predictor(self) -> InteractivePredictor:
        """予測器を初期化"""
        import sys

        sys.argv = ["test", "--load", "models/java-large-model/model_iter52.release"]

        args = _code2seq.get_args()
        np.random.seed(args.seed)
        import tensorflow as tf

        tf.compat.v1.set_random_seed(args.seed)
        config = Config.get_default_config(args)
        model = Model(config)
        return InteractivePredictor(config, model)

    def test_full_pipeline_tracking(self) -> bool:
        """
        完全なパイプラインを通してベクトル型の変換を追跡
        """
        print("=== 完全パイプライン型変換追跡テスト ===")

        test_code = """
        public class Test {
            public int process(int value) {
                return value * 2;
            }
        }
        """

        try:
            # Step 1: interactive_predict.py の get() メソッドを呼び出し
            print("Step 1: interactive_predict.py 実行開始")
            result = self.predictor.get(test_code)

            # Step 2: 結果の詳細分析
            print("Step 2: 結果の型変換状態確認")
            self._analyze_predictor_result(result)

            # Step 3: pickle直列化/逆直列化テスト
            print("Step 3: pickle直列化/逆直列化テスト")
            self._test_pickle_conversion(result)

            # Step 4: 要約表示
            self.tracker.print_conversion_summary()

            return True

        except Exception as e:
            print(f"❌ 完全パイプライン追跡失敗: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _analyze_predictor_result(self, result: List[Any]) -> None:
        """predictor.get()の結果を詳細分析"""
        print("\n--- predictor.get() 結果分析 ---")

        for method_idx, (method_name, method_vector, path_contexts) in enumerate(
            result
        ):
            print(f"メソッド {method_idx}: {method_name}")

            # メソッドベクトルの追跡
            self.tracker.log_vector_state(
                "predictor_result", method_vector, f"method[{method_idx}].vector"
            )

            # パスコンテキストの追跡（最初の3つのみ）
            for pc_idx, path_context in enumerate(path_contexts[:3]):
                print(f"  PathContext {pc_idx}:")
                print(f"    型: {type(path_context)}")

                # PathContextオブジェクトの属性を直接確認
                if hasattr(path_context, "vector"):
                    self.tracker.log_vector_state(
                        "pathcontext_object",
                        path_context.vector,
                        f"pc[{pc_idx}].vector",
                    )

                if hasattr(path_context, "source_vector"):
                    self.tracker.log_vector_state(
                        "pathcontext_object",
                        path_context.source_vector,
                        f"pc[{pc_idx}].source_vector",
                    )

                if hasattr(path_context, "target_vector"):
                    self.tracker.log_vector_state(
                        "pathcontext_object",
                        path_context.target_vector,
                        f"pc[{pc_idx}].target_vector",
                    )

                if hasattr(path_context, "astpath_vector"):
                    self.tracker.log_vector_state(
                        "pathcontext_object",
                        path_context.astpath_vector,
                        f"pc[{pc_idx}].astpath_vector",
                    )

    def _test_pickle_conversion(self, result: List[Any]) -> None:
        """pickle直列化/逆直列化による型変換をテスト"""
        print("\n--- pickle変換テスト ---")

        try:
            # 直列化前の状態確認
            print("直列化前の状態:")
            if result and len(result) > 0:
                _, _, path_contexts = result[0]
                if path_contexts and len(path_contexts) > 0:
                    pc = path_contexts[0]
                    if hasattr(pc, "vector"):
                        self.tracker.log_vector_state(
                            "before_pickle", pc.vector, "pc.vector"
                        )

            # pickle直列化/逆直列化実行
            print("pickle直列化/逆直列化実行中...")
            serialized = pickle.dumps(result)
            deserialized = pickle.loads(serialized)

            # 逆直列化後の状態確認
            print("逆直列化後の状態:")
            if deserialized and len(deserialized) > 0:
                _, _, path_contexts = deserialized[0]
                if path_contexts and len(path_contexts) > 0:
                    pc = path_contexts[0]
                    if hasattr(pc, "vector"):
                        self.tracker.log_vector_state(
                            "after_pickle", pc.vector, "pc.vector"
                        )
                    if hasattr(pc, "source_vector"):
                        self.tracker.log_vector_state(
                            "after_pickle", pc.source_vector, "pc.source_vector"
                        )
                    if hasattr(pc, "target_vector"):
                        self.tracker.log_vector_state(
                            "after_pickle", pc.target_vector, "pc.target_vector"
                        )
                    if hasattr(pc, "astpath_vector"):
                        self.tracker.log_vector_state(
                            "after_pickle", pc.astpath_vector, "pc.astpath_vector"
                        )

        except Exception as e:
            print(f"❌ pickle変換テスト失敗: {e}")

    def run_all_tests(self) -> bool:
        """全てのテストを実行"""
        print("🔍 型変換追跡テストスイート開始")
        print("=" * 60)

        tests = [
            ("完全パイプライン追跡", self.test_full_pipeline_tracking),
        ]

        passed = 0
        for test_name, test_func in tests:
            print(f"\n🧪 {test_name}テスト実行中...")
            try:
                if test_func():
                    print(f"✅ {test_name}テスト成功")
                    passed += 1
                else:
                    print(f"❌ {test_name}テスト失敗")
            except Exception as e:
                print(f"❌ {test_name}テスト例外: {e}")

        print(f"\n📊 テスト結果: {passed}/{len(tests)} 成功")
        print("=" * 60)

        return passed == len(tests)


def main() -> None:
    """メイン実行関数"""
    print("🚀 型変換追跡テストスイート")
    print("このテストは、numpy配列がリストに変換される箇所を特定します")
    print()

    try:
        tester = TestTypeConversionTracking()
        success = tester.run_all_tests()

        if success:
            print("🎉 全テスト成功")
            sys.exit(0)
        else:
            print("💥 一部テスト失敗")
            sys.exit(1)

    except Exception as e:
        print(f"💥 テスト実行エラー: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
