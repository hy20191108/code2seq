# mypy: disable-error-code="import,call-untyped"
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Add project root to sys.path for shared imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import _code2seq
from config import Config
from interactive_predict import InteractivePredictor
from model import Model


class TestFlaskApp:
    """code2seq flask_app.pyの動作チェック用テストクラス"""

    def __init__(self) -> None:
        """テスト環境を初期化"""
        self.predictor = self._initialize_predictor()

    def _initialize_predictor(self) -> InteractivePredictor:
        """predictorを初期化（flask_app.pyと同じロジック）"""
        # テスト用に適切な引数を設定
        import sys

        sys.argv = ["test", "--load", "models/java-large-model/model_iter52.release"]

        args = _code2seq.get_args()
        np.random.seed(args.seed)
        import tensorflow as tf

        # TensorFlow v1.x互換モードの設定
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.set_random_seed(args.seed)
        config = Config.get_default_config(args)
        model = Model(config)
        return InteractivePredictor(config, model)

    def test_predictor_basic_functionality(self) -> bool:
        """基本的な動作テスト"""
        print("=== 基本動作テスト ===")

        # テスト用のシンプルなJavaコード
        test_code = """
        public class Test {
            public int calculate(int x, int y) {
                return x + y;
            }
        }
        """

        try:
            result = self.predictor.get(test_code)
            print("✅ predictor.get()の実行成功")
            print(f"   結果の型: {type(result)}")
            print(
                f"   結果の長さ: {len(result) if hasattr(result, '__len__') else 'N/A'}"
            )
            return True
        except Exception as e:
            print(f"❌ predictor.get()の実行失敗: {e}")
            return False

    def test_data_structure(self) -> bool:
        """データ構造の検証テスト"""
        print("\n=== データ構造検証テスト ===")

        test_code = """
        public class Calculator {
            public int add(int a, int b) {
                int result = a + b;
                return result;
            }
        }
        """

        try:
            result = self.predictor.get(test_code)

            # 基本構造のチェック
            if not isinstance(result, list):
                print(f"❌ 結果がlistでない: {type(result)}")
                return False

            if len(result) == 0:
                print("❌ 結果が空のlist")
                return False

            print(f"✅ メソッド数: {len(result)}")

            # 各メソッドの構造をチェック
            for i, method_data in enumerate(result):
                if not isinstance(method_data, tuple):
                    print(f"❌ method_data[{i}]がtupleでない: {type(method_data)}")
                    return False

                if len(method_data) != 3:
                    print(f"❌ method_data[{i}]の長さが3でない: {len(method_data)}")
                    return False

                method_name, method_vector, path_contexts = method_data

                print(f"  メソッド{i}: {method_name}")
                print(f"    メソッドベクトル型: {type(method_vector)}")
                print(
                    f"    パスコンテキスト数: {len(path_contexts) if hasattr(path_contexts, '__len__') else 'N/A'}"
                )

                # メソッドベクトルのチェック
                # 注意: interactive_predict.pyではmethod_vectorは-1を返す仕様
                if isinstance(method_vector, np.ndarray):
                    print(f"    メソッドベクトル形状: {method_vector.shape}")
                elif method_vector == -1:
                    print("    メソッドベクトル: -1 (仕様通り)")
                else:
                    print(
                        f"❌ method_vectorが期待される形式でない: {type(method_vector)}, 値: {method_vector}"
                    )
                    return False

                # パスコンテキストのチェック
                if not isinstance(path_contexts, list):
                    print(f"❌ path_contextsがlistでない: {type(path_contexts)}")
                    return False

                return self._check_path_contexts(path_contexts)

            return True

        except Exception as e:
            print(f"❌ データ構造検証失敗: {e}")
            return False

    def _check_path_contexts(self, path_contexts: List[Dict[str, Any]]) -> bool:
        """パスコンテキストの詳細チェック"""
        print("\n  --- パスコンテキスト詳細チェック ---")

        required_keys = [
            "source",
            "path",
            "target",
            "attention",
            "vector",
            "source_vector",
            "target_vector",
            "astpath_vector",
            "lineColumns",
        ]

        for i, pc in enumerate(path_contexts[:3]):  # 最初の3つのみチェック
            print(f"    パスコンテキスト{i}:")

            if not isinstance(pc, dict):
                print(f"      ❌ 辞書でない: {type(pc)}")
                return False

            # 必須キーの存在チェック
            missing_keys = [key for key in required_keys if key not in pc]
            if missing_keys:
                print(f"      ❌ 必須キーが不足: {missing_keys}")
                return False

            # 各フィールドの型と内容をチェック
            print(f"      source: {pc['source']} (型: {type(pc['source'])})")
            print(f"      target: {pc['target']} (型: {type(pc['target'])})")
            print(f"      path: {pc['path'][:50]}... (型: {type(pc['path'])})")
            print(f"      attention: {pc['attention']} (型: {type(pc['attention'])})")

            # ベクトルの検証
            vectors_to_check = [
                ("vector", pc["vector"]),
                ("source_vector", pc["source_vector"]),
                ("target_vector", pc["target_vector"]),
                ("astpath_vector", pc["astpath_vector"]),
            ]

            for vec_name, vec_data in vectors_to_check:
                if vec_data is None:
                    print(f"      ⚠️ {vec_name}: None")
                elif isinstance(vec_data, np.ndarray):
                    print(
                        f"      ✅ {vec_name}: shape={vec_data.shape}, 非ゼロ要素数={np.count_nonzero(vec_data)}"
                    )
                elif isinstance(vec_data, (list, tuple)):
                    vec_array = np.array(vec_data)
                    print(
                        f"      ✅ {vec_name}: shape={vec_array.shape}, 非ゼロ要素数={np.count_nonzero(vec_array)}"
                    )
                else:
                    print(f"      ❌ {vec_name}: 不正な型 {type(vec_data)}")
                    return False

        return True

    def test_vector_quality(self) -> bool:
        """ベクトルの品質テスト"""
        print("\n=== ベクトル品質テスト ===")

        test_code = """
        public class Example {
            public void process(String input) {
                String result = input.toLowerCase();
                System.out.println(result);
            }
        }
        """

        try:
            result = self.predictor.get(test_code)

            if not result:
                print("❌ 結果が空")
                return False

            method_name, method_vector, path_contexts = result[0]

            # メソッドベクトルのチェック
            # 注意: interactive_predict.pyではmethod_vectorは-1を返す仕様
            if method_vector == -1:
                print("✅ メソッドベクトル: -1 (仕様通り)")
            elif isinstance(method_vector, np.ndarray):
                if np.allclose(method_vector, 0):
                    print("⚠️ メソッドベクトルがすべてゼロ")
                else:
                    print(
                        f"✅ メソッドベクトル: 非ゼロ要素数={np.count_nonzero(method_vector)}/{len(method_vector)}"
                    )
            else:
                print(f"⚠️ メソッドベクトルが予期しない形式: {type(method_vector)}")

            # パスコンテキストベクトルの品質チェック
            zero_vectors = {
                "vector": 0,
                "source_vector": 0,
                "target_vector": 0,
                "astpath_vector": 0,
            }
            total_contexts = len(path_contexts)

            for pc in path_contexts:
                for vec_name in zero_vectors.keys():
                    vec_data = pc.get(vec_name)
                    if vec_data is not None:
                        vec_array = (
                            np.array(vec_data)
                            if not isinstance(vec_data, np.ndarray)
                            else vec_data
                        )
                        if np.allclose(vec_array, 0):
                            zero_vectors[vec_name] += 1

            print(f"パスコンテキスト総数: {total_contexts}")
            for vec_name, zero_count in zero_vectors.items():
                non_zero_count = total_contexts - zero_count
                print(f"  {vec_name}: 非ゼロ={non_zero_count}, ゼロ={zero_count}")
                if zero_count == total_contexts:
                    print(f"    ⚠️ {vec_name}がすべてゼロベクトル")
                else:
                    print(f"    ✅ {vec_name}に有効なデータが存在")

            return True

        except Exception as e:
            print(f"❌ ベクトル品質テスト失敗: {e}")
            return False

    def test_pickle_serialization(self) -> bool:
        """pickleシリアライゼーションテスト"""
        print("\n=== Pickleシリアライゼーションテスト ===")

        test_code = """
        public class Serialization {
            public boolean isValid(Object obj) {
                return obj != null;
            }
        }
        """

        try:
            result = self.predictor.get(test_code)

            # pickleでシリアライズ
            serialized = pickle.dumps(result)
            print(f"✅ シリアライズ成功: {len(serialized)} bytes")

            # pickleでデシリアライズ
            deserialized = pickle.loads(serialized)
            print("✅ デシリアライズ成功")

            # データの整合性チェック
            if len(result) != len(deserialized):
                print("❌ データ長が一致しない")
                return False

            for orig, deser in zip(result, deserialized):
                if orig[0] != deser[0]:  # method_name
                    print("❌ メソッド名が一致しない")
                    return False

                # method_vector は -1 の場合があるので適切にチェック
                if isinstance(orig[1], np.ndarray) and isinstance(deser[1], np.ndarray):
                    if not np.allclose(orig[1], deser[1]):
                        print("❌ メソッドベクトルが一致しない")
                        return False
                elif orig[1] != deser[1]:
                    print("❌ メソッドベクトルが一致しない")
                    return False

            print("✅ シリアライゼーション整合性確認完了")
            return True

        except Exception as e:
            print(f"❌ Pickleシリアライゼーションテスト失敗: {e}")
            return False

    def run_all_tests(self) -> None:
        """すべてのテストを実行"""
        print("code2seq flask_app.py 動作チェックテスト開始\n")

        tests = [
            ("基本機能", self.test_predictor_basic_functionality),
            ("データ構造", self.test_data_structure),
            ("ベクトル品質", self.test_vector_quality),
            ("Pickleシリアライゼーション", self.test_pickle_serialization),
        ]

        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"❌ {test_name}テストで例外発生: {e}")
                results.append((test_name, False))

        # 結果サマリー
        print("\n" + "=" * 50)
        print("テスト結果サマリー")
        print("=" * 50)

        passed = 0
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name}: {status}")
            if result:
                passed += 1

        print(f"\n合計: {passed}/{len(results)} テスト通過")

        if passed == len(results):
            print("🎉 すべてのテストが成功しました！")
        else:
            print("⚠️ 一部のテストが失敗しました。詳細を確認してください。")


def main() -> None:
    """メイン関数"""
    try:
        tester = TestFlaskApp()
        tester.run_all_tests()
    except Exception as e:
        print(f"❌ テスト実行中に致命的エラー発生: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
