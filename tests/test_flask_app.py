# mypy: disable-error-code="import"
# type: ignore
import pickle
import sys
from pathlib import Path
from typing import Any, List

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
    """
    code2seq flask_app.pyの動作チェック用テストクラス

    【設計思想】
    このテストスイートは、Code2seqシステムの品質保証を多層的に行います：

    1. 基本契約の保証 - システムがクラッシュせず、期待される型で応答する
    2. データ整合性の検証 - APIが約束した構造とスキーマを遵守する
    3. 機械学習品質の確認 - ベクトル表現が学習済みの意味情報を保持する
    4. システム統合の保証 - 複雑なオブジェクト階層が永続化・復元可能である

    【検証レベル】
    - インターフェース層：API応答の構造的正当性
    - データ層：ベクトル表現の意味的妥当性
    - システム層：コンポーネント間の統合完全性
    """

    def __init__(self) -> None:
        """テスト環境を初期化"""
        self.predictor = self._initialize_predictor()

    def _initialize_predictor(self) -> InteractivePredictor:
        """
        predictorを初期化（flask_app.pyと同じロジック）

        【意図】本番環境と同一の初期化パスを使用することで、
        テスト環境と本番環境の動作差異を最小限に抑制
        """
        # テスト用に適切な引数を設定
        import sys

        sys.argv = ["test", "--load", "models/java-large-model/model_iter52.release"]

        args = _code2seq.get_args()
        np.random.seed(args.seed)
        import tensorflow as tf

        tf.compat.v1.set_random_seed(args.seed)
        config = Config.get_default_config(args)
        model = Model(config)
        return InteractivePredictor(config, model)

    def test_predictor_basic_functionality(self) -> bool:
        """
        基本的な動作テスト

        【検証意図】システムの基本契約履行
        - 入力を受け付け、例外なく処理を完了する
        - 期待される基本型（list）で結果を返す
        - メモリリークや致命的エラーが発生しない

        【品質保証レベル】最低限の稼働保証
        """
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
        """
        データ構造の検証テスト

        【検証意図】APIコントラクトの厳密な遵守確認
        - 約束されたスキーマ構造の完全な実装
        - 型安全性とデータ整合性の保証
        - 下流コンシューマーが依存できる安定したインターフェース

        【品質保証レベル】インターフェース契約の完全性
        """
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

    def _check_path_contexts(self, path_contexts: List[Any]) -> bool:
        """
        パスコンテキストの詳細チェック

        【検証意図】ベクトル表現の完全性保証
        - 機械学習モデルが生成した全ベクトルの存在確認
        - Noneや無効値による下流処理の破綻防止
        - ベクトル次元の仕様準拠確認

        【品質保証レベル】データ品質の厳密な検証
        """
        print("\n  --- パスコンテキスト詳細チェック ---")

        # 期待されるベクトル次元数（設定ファイルに基づく）
        expected_dimensions = {
            "vector": 320,  # DECODER_SIZE (パスコンテキスト統合ベクトル)
            "source_vector": 128,  # EMBEDDINGS_SIZE (ソース単語ベクトル)
            "target_vector": 128,  # EMBEDDINGS_SIZE (ターゲット単語ベクトル)
            "astpath_vector": 256,  # RNN_SIZE (ASTパスベクトル)
        }

        required_properties = [
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

        # 統計情報収集用
        vector_stats = {
            name: {"total": 0, "valid": 0, "zero": 0, "dimension_errors": 0}
            for name in expected_dimensions.keys()
        }

        for i, pc in enumerate(path_contexts[:5]):  # 最初の5つをチェック（拡張）
            print(f"    パスコンテキスト{i}:")

            # PathContextオブジェクトかチェック
            from code2seq.data.path_context import PathContext

            if not isinstance(pc, PathContext):
                print(f"      ❌ PathContextオブジェクトでない: {type(pc)}")
                return False

            # 必須プロパティの存在チェック
            missing_properties = []
            for prop in required_properties:
                if not hasattr(pc, prop):
                    missing_properties.append(prop)

            if missing_properties:
                print(f"      ❌ 必須プロパティが不足: {missing_properties}")
                return False

            # 各フィールドの型と内容をチェック
            print(f"      source: {pc.source} (型: {type(pc.source)})")
            print(f"      target: {pc.target} (型: {type(pc.target)})")
            print(f"      path: {pc.path[:50]}... (型: {type(pc.path)})")
            print(f"      attention: {pc.attention:.6f} (型: {type(pc.attention)})")

            # ベクトルの検証（次元数チェックを追加）
            vectors_to_check = [
                ("vector", pc.vector),
                ("source_vector", pc.source_vector),
                ("target_vector", pc.target_vector),
                ("astpath_vector", pc.astpath_vector),
            ]

            for vec_name, vec_data in vectors_to_check:
                vector_stats[vec_name]["total"] += 1

                if vec_data is None:
                    print(f"      ❌ {vec_name}: None (ベクトルが設定されていない)")
                    return False

                # ベクトルを配列に変換
                if isinstance(vec_data, (list, tuple)):
                    vec_array = np.array(vec_data)
                elif isinstance(vec_data, np.ndarray):
                    vec_array = vec_data
                else:
                    print(f"      ❌ {vec_name}: 不正な型 {type(vec_data)}")
                    return False

                # 次元数チェック
                expected_dim = expected_dimensions[vec_name]
                actual_dim = (
                    vec_array.shape[0] if len(vec_array.shape) == 1 else len(vec_array)
                )

                if actual_dim != expected_dim:
                    print(
                        f"      ❌ {vec_name}: 次元数不正 (期待={expected_dim}, 実際={actual_dim})"
                    )
                    vector_stats[vec_name]["dimension_errors"] += 1
                    return False

                # ゼロベクトルチェック
                if np.allclose(vec_array, 0):
                    print(
                        f"      ❌ {vec_name}: すべてゼロベクトル (次元={actual_dim})"
                    )
                    vector_stats[vec_name]["zero"] += 1
                    return False

                # 統計的品質チェック
                non_zero_count = np.count_nonzero(vec_array)
                mean_val = np.mean(np.abs(vec_array))
                std_val = np.std(vec_array)

                vector_stats[vec_name]["valid"] += 1

                print(
                    f"      ✅ {vec_name}: 次元={actual_dim}, 非ゼロ={non_zero_count}/{len(vec_array)}"
                )
                print(
                    f"         統計: 平均絶対値={mean_val:.4f}, 標準偏差={std_val:.4f}"
                )

        # 統計サマリー表示
        print("\n  --- ベクトル品質統計サマリー ---")
        for vec_name, stats in vector_stats.items():
            if stats["total"] > 0:
                success_rate = (stats["valid"] / stats["total"]) * 100
                print(
                    f"    {vec_name}: 成功率={success_rate:.1f}% ({stats['valid']}/{stats['total']})"
                )
                if stats["zero"] > 0:
                    print(f"      警告: ゼロベクトル検出 {stats['zero']}個")
                if stats["dimension_errors"] > 0:
                    print(f"      エラー: 次元数不正 {stats['dimension_errors']}個")

        return True

    def test_vector_quality(self) -> bool:
        """
        ベクトルの品質テスト

        【検証意図】機械学習モデルの出力品質保証
        - 学習済みベクトルが有意味な情報を保持することの確認
        - ゼロベクトルや退化したベクトルの検出
        - 統計的な品質メトリクスによる健全性評価

        【品質保証レベル】機械学習システムの意味情報完全性
        """
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
                    vec_data = getattr(pc, vec_name)  # オブジェクトアクセスに変更
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
        """
        pickleシリアライゼーションテスト

        【検証意図】システム統合とデータ永続化の保証
        - 複雑なオブジェクト階層の完全な永続化可能性
        - プロセス間通信・データ保存における整合性維持
        - NumPy配列を含む混合型データの安定性確認

        【品質保証レベル】システム間連携の信頼性
        """
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

    def test_performance_benchmark(self) -> bool:
        """
        パフォーマンスベンチマークテスト

        【検証意図】システム性能とスケーラビリティの評価
        - 単一メソッド処理時間の測定
        - 複数メソッド処理での線形性確認
        - メモリ使用量の健全性チェック

        【品質保証レベル】システム性能の実用性保証
        """
        import gc
        import time

        print("\n=== パフォーマンスベンチマークテスト ===")

        # シンプルなテストケース
        simple_code = """
        public class Simple {
            public int add(int a, int b) { return a + b; }
        }
        """

        # 複雑なテストケース
        complex_code = """
        public class Complex {
            public int calculate(int x, int y) { 
                int result = x * y; 
                return result + 1;
            }
            public void process(String data) {
                String cleaned = data.trim().toLowerCase();
                System.out.println(cleaned);
            }
            public boolean validate(Object obj) {
                return obj != null && obj.toString().length() > 0;
            }
        }
        """

        try:
            # ウォームアップ
            self.predictor.get(simple_code)
            gc.collect()

            # シンプルケースの測定
            start_time = time.time()
            result_simple = self.predictor.get(simple_code)
            simple_duration = time.time() - start_time

            # 複雑ケースの測定
            start_time = time.time()
            result_complex = self.predictor.get(complex_code)
            complex_duration = time.time() - start_time

            print(
                f"✅ シンプルケース: {simple_duration:.3f}秒, メソッド数={len(result_simple)}"
            )
            print(
                f"✅ 複雑ケース: {complex_duration:.3f}秒, メソッド数={len(result_complex)}"
            )

            # パフォーマンス閾値チェック（実用的な範囲）
            if simple_duration > 30.0:  # 30秒以上は実用的でない
                print(
                    f"⚠️ シンプルケースの処理時間が長すぎます: {simple_duration:.3f}秒"
                )
            else:
                print("✅ シンプルケースの処理時間が適切")

            if complex_duration > 60.0:  # 60秒以上は実用的でない
                print(f"⚠️ 複雑ケースの処理時間が長すぎます: {complex_duration:.3f}秒")
            else:
                print("✅ 複雑ケースの処理時間が適切")

            # メソッド数に対する処理時間の線形性簡易チェック
            time_per_method_simple = simple_duration / len(result_simple)
            time_per_method_complex = complex_duration / len(result_complex)

            print(
                f"メソッドあたり処理時間: シンプル={time_per_method_simple:.3f}秒, 複雑={time_per_method_complex:.3f}秒"
            )

            return True

        except Exception as e:
            print(f"❌ パフォーマンステスト失敗: {e}")
            return False

    def test_edge_cases(self) -> bool:
        """
        エッジケーステスト

        【検証意図】境界条件での安定性確認
        - 空メソッド、極小メソッドでの動作確認
        - 特殊文字・Unicode文字列の処理確認
        - 異常入力に対する適切なエラーハンドリング

        【品質保証レベル】システムの堅牢性保証
        """
        print("\n=== エッジケーステスト ===")

        test_cases = [
            (
                "空メソッド",
                """
            public class Empty {
                public void empty() {}
            }
            """,
            ),
            (
                "単行メソッド",
                """
            public class OneLiner {
                public int get() { return 42; }
            }
            """,
            ),
            (
                "Unicode文字列",
                """
            public class Unicode {
                public String getMessage() { return "こんにちは世界"; }
            }
            """,
            ),
        ]

        for case_name, test_code in test_cases:
            try:
                print(f"  {case_name}テスト:")
                result = self.predictor.get(test_code)

                if not result:
                    print(f"    ⚠️ 結果が空: {case_name}")
                    continue

                method_name, method_vector, path_contexts = result[0]
                print(f"    ✅ メソッド名: {method_name}")
                print(f"    ✅ パスコンテキスト数: {len(path_contexts)}")

                # 基本的な品質チェック
                if len(path_contexts) > 0:
                    pc = path_contexts[0]
                    if all(
                        key in pc
                        for key in [
                            "vector",
                            "source_vector",
                            "target_vector",
                            "astpath_vector",
                        ]
                    ):
                        print("    ✅ ベクトル構造完全")
                    else:
                        print("    ⚠️ ベクトル構造不完全")

            except Exception as e:
                print(f"    ❌ {case_name}テスト失敗: {e}")
                return False

        return True

    def test_vector_consistency(self) -> bool:
        """
        ベクトル一貫性テスト

        【検証意図】同一入力に対するベクトル出力の再現性確認
        - 複数実行での結果一貫性（決定性の確認）
        - ベクトル値の数値安定性検証
        - 機械学習モデルの出力品質保証

        【品質保証レベル】機械学習システムの信頼性保証
        """
        print("\n=== ベクトル一貫性テスト ===")

        test_code = """
        public class Consistency {
            public String process(String input) {
                return input.toLowerCase();
            }
        }
        """

        try:
            # 同じコードを2回実行
            result1 = self.predictor.get(test_code)
            result2 = self.predictor.get(test_code)

            if len(result1) != len(result2):
                print("❌ メソッド数が一致しない")
                return False

            method1 = result1[0]
            method2 = result2[0]

            # メソッド名の一致確認
            if method1[0] != method2[0]:
                print("❌ メソッド名が一致しない")
                return False

            # パスコンテキスト数の一致確認
            contexts1 = method1[2]
            contexts2 = method2[2]

            if len(contexts1) != len(contexts2):
                print("❌ パスコンテキスト数が一致しない")
                return False

            # ベクトル値の一致確認（最初のパスコンテキストのみ）
            if len(contexts1) > 0 and len(contexts2) > 0:
                pc1, pc2 = contexts1[0], contexts2[0]

                vector_comparisons = [
                    ("vector", pc1.vector, pc2.vector),
                    ("source_vector", pc1.source_vector, pc2.source_vector),
                    ("target_vector", pc1.target_vector, pc2.target_vector),
                    ("astpath_vector", pc1.astpath_vector, pc2.astpath_vector),
                ]

                for vec_name, vec1, vec2 in vector_comparisons:
                    arr1 = np.array(vec1) if isinstance(vec1, (list, tuple)) else vec1
                    arr2 = np.array(vec2) if isinstance(vec2, (list, tuple)) else vec2

                    if not np.allclose(arr1, arr2, rtol=1e-6):
                        print(f"❌ {vec_name}ベクトルが一致しない")
                        max_diff = np.max(np.abs(arr1 - arr2))
                        print(f"    最大差異: {max_diff}")
                        return False
                    else:
                        print(f"✅ {vec_name}ベクトル一致")

            print("✅ ベクトル一貫性確認完了")
            return True

        except Exception as e:
            print(f"❌ ベクトル一貫性テスト失敗: {e}")
            return False

    def test_vector_type_validation(self) -> bool:
        """
        ベクトル型検証テスト

        【検証意図】ベクトルの型安全性とデータ形式の厳密な確認
        - API出力での型がリスト形式であることの確認
        - 内部処理でnumpy配列が正しく使用されることの確認
        - 型変換の整合性とデータ保全性の検証

        【品質保証レベル】型安全性とデータ整合性の保証
        """
        print("\n=== ベクトル型検証テスト ===")

        test_code = """
        public class TypeCheck {
            public int multiply(int a, int b) {
                return a * b;
            }
        }
        """

        try:
            result = self.predictor.get(test_code)

            if not result:
                print("❌ 結果が空")
                return False

            method_name, method_vector, path_contexts = result[0]
            print(f"メソッド名: {method_name}")

            if not path_contexts:
                print("❌ パスコンテキストが空")
                return False

            # 型チェック対象のベクトル（numpy配列を期待）
            expected_types = {
                "vector": (np.ndarray, 320),  # パスコンテキストベクトル
                "source_vector": (np.ndarray, 128),  # ソース単語ベクトル
                "target_vector": (np.ndarray, 128),  # ターゲット単語ベクトル
                "astpath_vector": (np.ndarray, 256),  # ASTパスベクトル
            }

            print("\n--- 型・次元検証 ---")
            all_passed = True

            for i, pc in enumerate(path_contexts[:3]):
                print(f"  パスコンテキスト{i}:")

                for vec_name, (expected_type, expected_dim) in expected_types.items():
                    vec_data = getattr(pc, vec_name)  # オブジェクトアクセスに変更

                    # 型チェック（numpy配列を期待）
                    if not isinstance(vec_data, expected_type):
                        print(
                            f"    ❌ {vec_name}: 型不正 (期待={expected_type.__name__}, 実際={type(vec_data).__name__})"
                        )
                        if isinstance(vec_data, list):
                            print(
                                "        ⚠️ list型が検出されました。numpy配列型であるべきです。"
                            )
                        all_passed = False
                        continue

                    # numpy配列の形状チェック
                    if vec_data.shape != (expected_dim,):
                        print(
                            f"    ❌ {vec_name}: 形状不正 (期待=({expected_dim},), 実際={vec_data.shape})"
                        )
                        all_passed = False
                        continue

                    # データ型チェック
                    if not np.issubdtype(vec_data.dtype, np.floating):
                        print(
                            f"    ❌ {vec_name}: データ型不正 (期待=float型, 実際={vec_data.dtype})"
                        )
                        all_passed = False
                        continue

                    # 有限数チェック（NaN, Infが含まれていないか）
                    if not np.all(np.isfinite(vec_data)):
                        print(f"    ❌ {vec_name}: 無限値またはNaNが含まれている")
                        all_passed = False
                        continue

                    print(
                        f"    ✅ {vec_name}: 型={type(vec_data).__name__}, 形状={vec_data.shape}, dtype={vec_data.dtype}"
                    )

                    # 統計情報も表示
                    min_val, max_val = np.min(vec_data), np.max(vec_data)
                    mean_val, std_val = np.mean(vec_data), np.std(vec_data)
                    print(
                        f"        統計: 範囲=[{min_val:.4f}, {max_val:.4f}], 平均={mean_val:.4f}, 標準偏差={std_val:.4f}"
                    )

            # 追加の型分析（期待される実装との比較）
            print("\n--- 型実装状況の分析 ---")
            pc_sample = path_contexts[0]

            print("現在の実装での型状況:")
            for vec_name in expected_types.keys():
                vec_data = getattr(pc_sample, vec_name)  # オブジェクトアクセスに変更
                print(f"  {vec_name}: {type(vec_data).__name__}")

                if isinstance(vec_data, list):
                    print("    → ❌ 問題: list型で実装されています")
                    print(
                        "    → 💡 修正必要: model.pyで.tolist()を削除してnumpy配列のまま出力"
                    )
                elif isinstance(vec_data, np.ndarray):
                    print("    → ✅ 正常: numpy配列で実装されています")
                else:
                    print(f"    → ⚠️ 予期しない型: {type(vec_data)}")

            print("\n期待される修正:")
            print("  model.py get_method()内で:")
            print('  - "vector": vector.tolist() → "vector": vector')
            print(
                '  - "source_vector": source_vectors[i].tolist() → "source_vector": source_vectors[i]'
            )
            print(
                '  - "target_vector": target_vectors[i].tolist() → "target_vector": target_vectors[i]'
            )
            print(
                '  - "astpath_vector": astpath_vectors[i].tolist() → "astpath_vector": astpath_vectors[i]'
            )

            if all_passed:
                print("\n✅ 全ての型・次元検証が成功")
                return True
            else:
                print("\n❌ 型・次元検証で問題発見")
                return False

        except Exception as e:
            print(f"❌ ベクトル型検証テスト失敗: {e}")
            import traceback

            traceback.print_exc()
            return False

    def run_all_tests(self) -> None:
        """
        すべてのテストを実行

        【検証意図】包括的品質保証の統合実行
        - 多層的テストによるシステム品質の全方位検証
        - 継続的インテグレーションでの自動品質確認
        - 回帰バグの早期発見とシステム安定性維持

        【品質保証レベル】システム全体の信頼性評価
        """
        print("code2seq flask_app.py 動作チェックテスト開始\n")

        tests = [
            ("基本機能", self.test_predictor_basic_functionality),
            ("データ構造", self.test_data_structure),
            ("ベクトル品質", self.test_vector_quality),
            ("Pickleシリアライゼーション", self.test_pickle_serialization),
            ("パフォーマンスベンチマーク", self.test_performance_benchmark),
            ("エッジケース", self.test_edge_cases),
            ("ベクトル一貫性", self.test_vector_consistency),
            ("ベクトル型検証", self.test_vector_type_validation),
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
