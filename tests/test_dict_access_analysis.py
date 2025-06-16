"""
辞書アクセス分析テストスイート

このテストモジュールは、eye2vecのcontext_model.pyで実行される
辞書アクセス（path_context["vector"]）の詳細を分析し、
なぜnumpy配列がリストとして返されるのかを解明します。
"""

import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

# プロジェクトルートをsys.pathに追加
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import _code2seq
    from config import Config
    from interactive_predict import InteractivePredictor
    from model import Model
except ImportError as e:
    print(f"インポートエラー: {e}")
    print("必要なモジュールが見つかりません。")
    sys.exit(1)


class TestDictAccessAnalysis:
    """辞書アクセス分析テストクラス"""

    def __init__(self):
        self.predictor = self._initialize_predictor()

    def _initialize_predictor(self):
        """予測器を初期化"""
        import sys

        sys.argv = ["test", "--load", "models/java-large-model/model_iter52.release"]

        args = _code2seq.get_args()
        np.random.seed(args.seed)
        try:
            import tensorflow as tf

            tf.compat.v1.set_random_seed(args.seed)
        except ImportError:
            print("TensorFlowが見つかりません")

        config = Config.get_default_config(args)
        model = Model(config)
        return InteractivePredictor(config, model)

    def analyze_object_structure(self, obj: Any, name: str) -> None:
        """オブジェクトの構造を分析"""
        print(f"\n=== {name} 構造分析 ===")
        print(f"型: {type(obj)}")
        print(f"__getitem__: {hasattr(obj, '__getitem__')}")
        print(f"__dict__: {hasattr(obj, '__dict__')}")

        if hasattr(obj, "__dict__"):
            try:
                obj_dict = obj.__dict__
                print(f"__dict__キー: {list(obj_dict.keys())}")
            except Exception as e:
                print(f"__dict__アクセスエラー: {e}")

    def test_pathcontext_dict_access(self) -> bool:
        """PathContextの辞書アクセスをテスト"""
        print("=== PathContext辞書アクセステスト ===")

        test_code = """
        public class Test {
            public int getValue() {
                return 42;
            }
        }
        """

        try:
            # predictor実行
            result = self.predictor.get(test_code)

            if not result or len(result) == 0:
                print("❌ 結果が空")
                return False

            method_name, method_vector, path_contexts = result[0]

            if not path_contexts or len(path_contexts) == 0:
                print("❌ PathContextが空")
                return False

            # 最初のPathContextを分析
            pc = path_contexts[0]
            self.analyze_object_structure(pc, "PathContext")

            # 辞書アクセステスト
            vector_keys = ["vector", "source_vector", "target_vector", "astpath_vector"]

            for key in vector_keys:
                print(f"\nキー '{key}' のテスト:")

                # パターン1: obj[key]
                try:
                    value1 = pc[key]
                    print(f"  pc['{key}'] ✅: {type(value1)}")
                    if isinstance(value1, (list, np.ndarray)):
                        is_numpy = isinstance(value1, np.ndarray)
                        print(f"    numpy配列: {'✅' if is_numpy else '🔥 リスト'}")
                        if hasattr(value1, "shape"):
                            print(f"    形状: {value1.shape}")
                        elif hasattr(value1, "__len__"):
                            print(f"    長さ: {len(value1)}")
                except Exception as e:
                    print(f"  pc['{key}'] ❌: {e}")

                # パターン2: getattr
                try:
                    value2 = getattr(pc, key)
                    print(f"  getattr(pc, '{key}') ✅: {type(value2)}")
                except Exception as e:
                    print(f"  getattr(pc, '{key}') ❌: {e}")

            return True

        except Exception as e:
            print(f"❌ テスト失敗: {e}")
            import traceback

            traceback.print_exc()
            return False

    def test_pickle_effect(self) -> bool:
        """pickle処理の影響をテスト"""
        print("\n=== pickle処理影響テスト ===")

        test_code = """
        public class PickleTest {
            public String process() {
                return "test";
            }
        }
        """

        try:
            # オリジナル結果
            original = self.predictor.get(test_code)

            # pickle処理
            serialized = pickle.dumps(original)
            deserialized = pickle.loads(serialized)

            # 比較
            if (
                original
                and deserialized
                and len(original) > 0
                and len(deserialized) > 0
            ):
                _, _, orig_pcs = original[0]
                _, _, deser_pcs = deserialized[0]

                if orig_pcs and deser_pcs and len(orig_pcs) > 0 and len(deser_pcs) > 0:
                    orig_pc = orig_pcs[0]
                    deser_pc = deser_pcs[0]

                    print("オリジナル vs デシリアライズ後:")

                    for key in ["vector", "source_vector"]:
                        try:
                            orig_val = orig_pc[key]
                            deser_val = deser_pc[key]

                            orig_is_numpy = isinstance(orig_val, np.ndarray)
                            deser_is_numpy = isinstance(deser_val, np.ndarray)
                            deser_is_list = isinstance(deser_val, list)

                            print(f"\n  {key}:")
                            print(
                                f"    オリジナル: {type(orig_val)} (numpy: {'✅' if orig_is_numpy else '❌'})"
                            )
                            print(
                                f"    デシリアライズ後: {type(deser_val)} (numpy: {'✅' if deser_is_numpy else '❌'}, list: {'🔥' if deser_is_list else '❌'})"
                            )

                            if orig_is_numpy and deser_is_list:
                                print("    🔥 型変換検出: numpy配列 → リスト")
                            elif orig_is_numpy and deser_is_numpy:
                                print("    ✅ 型変換なし: numpy配列のまま")

                        except Exception as e:
                            print(f"    {key}アクセスエラー: {e}")

            return True

        except Exception as e:
            print(f"❌ pickle影響テスト失敗: {e}")
            return False

    def run_tests(self) -> bool:
        """テストを実行"""
        print("🔍 辞書アクセス分析テストスイート開始")
        print("=" * 50)

        tests = [
            self.test_pathcontext_dict_access,
            self.test_pickle_effect,
        ]

        passed = 0
        for i, test_func in enumerate(tests, 1):
            print(f"\n🧪 テスト {i} 実行中...")
            try:
                if test_func():
                    print(f"✅ テスト {i} 成功")
                    passed += 1
                else:
                    print(f"❌ テスト {i} 失敗")
            except Exception as e:
                print(f"❌ テスト {i} 例外: {e}")

        print(f"\n📊 結果: {passed}/{len(tests)} 成功")
        print("=" * 50)

        return passed == len(tests)


def main() -> None:
    """メイン実行関数"""
    print("🚀 辞書アクセス分析テストスイート")
    print('path_context["vector"]でリストが返される原因を調査します')
    print()

    try:
        tester = TestDictAccessAnalysis()
        success = tester.run_tests()

        if success:
            print("🎉 全テスト成功")
        else:
            print("💥 一部テスト失敗")

    except Exception as e:
        print(f"💥 テスト実行エラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
