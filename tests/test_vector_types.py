"""
型安全性検証テストスイート

このテストモジュールは、code2seqシステムから出力されるベクトルデータの
型安全性を包括的に検証します。特に、pickleの直列化/逆直列化過程で
numpy配列がリストに変換される問題を検出することが主要な目的です。

検証対象：
1. TensorFlowセッション実行直後のベクトル型
2. PathContextオブジェクト作成時のベクトル型
3. pickle直列化/逆直列化後のベクトル型
4. Flask API応答でのベクトル型
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

import _code2seq
from config import Config
from interactive_predict import InteractivePredictor
from model import Model


class VectorTypeValidator:
    """ベクトル型の検証を行うユーティリティクラス"""

    @staticmethod
    def validate_numpy_array(
        vector: Any, expected_shape: tuple[int, ...], name: str
    ) -> bool:
        """numpy配列の型と形状を検証"""
        if not isinstance(vector, np.ndarray):
            print(f"❌ {name}: numpy配列でない (型: {type(vector)})")
            return False

        if vector.shape != expected_shape:
            print(
                f"❌ {name}: 形状が不正 (期待: {expected_shape}, 実際: {vector.shape})"
            )
            return False

        if not np.issubdtype(vector.dtype, np.floating):
            print(f"❌ {name}: 浮動小数点型でない (dtype: {vector.dtype})")
            return False

        return True

    @staticmethod
    def validate_list_conversion(
        original: np.ndarray, converted: Any, name: str
    ) -> bool:
        """numpy配列からリストへの変換を検証"""
        if not isinstance(converted, list):
            print(f"❌ {name}: リストに変換されていない (型: {type(converted)})")
            return False

        if len(converted) != original.size:
            print(
                f"❌ {name}: サイズが不一致 (元: {original.size}, 変換後: {len(converted)})"
            )
            return False

        # 値の一致確認（最初の5要素のみ）
        for i in range(min(5, len(converted))):
            if not np.allclose(float(converted[i]), original.flat[i], rtol=1e-6):
                print(
                    f"❌ {name}: 値が不一致 (インデックス{i}: {converted[i]} vs {original.flat[i]})"
                )
                return False

        return True


class TestVectorTypes:
    """ベクトル型検証テストクラス"""

    def __init__(self) -> None:
        """テスト環境を初期化"""
        self.predictor = self._initialize_predictor()
        self.validator = VectorTypeValidator()

    def _initialize_predictor(self) -> InteractivePredictor:
        """予測器を初期化（test_flask_app.pyと同じロジック）"""
        import sys

        sys.argv = ["test", "--load", "models/java-large-model/model_iter52.release"]

        args = _code2seq.get_args()
        np.random.seed(args.seed)
        import tensorflow as tf

        tf.compat.v1.set_random_seed(args.seed)
        config = Config.get_default_config(args)
        model = Model(config)
        return InteractivePredictor(config, model)

    def test_tensorflow_session_vector_types(self) -> bool:
        """
        TensorFlowセッション実行直後のベクトル型を検証

        この段階では、全てのベクトルがnumpy配列である必要があります。
        """
        print("=== TensorFlowセッション実行後のベクトル型検証 ===")

        test_code = """
        public class TypeTest {
            public int processData(int value) {
                int result = value * 2;
                return result;
            }
        }
        """

        try:
            # model.predict()を直接呼び出してTensorFlowの生出力を確認
            predict_lines, pc_info_dict, code_info_list = (
                self.predictor.path_extractor.extract_paths(test_code)
            )

            # 低レベルのモデル実行（ここではまだ型変換は起こらないはず）
            code_results = self.predictor.model.predict(predict_lines)

            print(f"✅ モデル実行成功（メソッド数: {len(code_results.method_list)}）")

            # 各メソッドのベクトル型を検証
            for method_idx, method in enumerate(code_results.method_list):
                print(f"\n  メソッド {method_idx}: {method.name}")

                for predict_name_idx, predict_name in enumerate(
                    method.predict_name_list[:1]
                ):  # 最初のタイムステップのみ
                    print(f"    予測 {predict_name_idx}:")

                    for pc_idx, path_context in enumerate(
                        predict_name.path_context_list[:3]
                    ):  # 最初の3つのみ
                        print(f"      PathContext {pc_idx}:")

                        # パスコンテキストベクトルの検証
                        if not self.validator.validate_numpy_array(
                            path_context.vector, (320,), f"PathContext[{pc_idx}].vector"
                        ):
                            return False

                        # ソース単語ベクトルの検証（存在する場合）
                        if (
                            hasattr(path_context.source, "vector")
                            and path_context.source.vector is not None
                        ):
                            if not self.validator.validate_numpy_array(
                                path_context.source.vector,
                                (128,),
                                f"PathContext[{pc_idx}].source.vector",
                            ):
                                return False

                        # ターゲット単語ベクトルの検証（存在する場合）
                        if (
                            hasattr(path_context.target, "vector")
                            and path_context.target.vector is not None
                        ):
                            if not self.validator.validate_numpy_array(
                                path_context.target.vector,
                                (128,),
                                f"PathContext[{pc_idx}].target.vector",
                            ):
                                return False

                        # ASTパスベクトルの検証（存在する場合）
                        if (
                            hasattr(path_context, "astpath")
                            and hasattr(path_context.astpath, "vector")
                            and path_context.astpath.vector is not None
                        ):
                            if not self.validator.validate_numpy_array(
                                path_context.astpath.vector,
                                (256,),
                                f"PathContext[{pc_idx}].astpath.vector",
                            ):
                                return False

            print("✅ TensorFlowセッション後のベクトル型検証完了")
            return True

        except Exception as e:
            print(f"❌ TensorFlowセッション後のベクトル型検証失敗: {e}")
            import traceback

            traceback.print_exc()
            return False

    def test_interactive_predict_output_types(self) -> bool:
        """
        InteractivePredictor.get()の出力型を検証

        この段階で型変換が発生する可能性があります。
        """
        print("\n=== InteractivePredictor出力型検証 ===")

        test_code = """
        public class OutputTest {
            public String formatValue(int input) {
                String formatted = "Value: " + input;
                return formatted;
            }
        }
        """

        try:
            result = self.predictor.get(test_code)

            for method_idx, (method_name, method_vector, path_contexts) in enumerate(
                result
            ):
                print(f"\n  メソッド {method_idx}: {method_name}")

                # PathContextオブジェクトの型を検証
                for pc_idx, path_context in enumerate(
                    path_contexts[:3]
                ):  # 最初の3つのみ
                    print(f"    PathContext {pc_idx}:")

                    # PathContextオブジェクトの型確認
                    from data.path_context import PathContext

                    if not isinstance(path_context, PathContext):
                        print(f"❌ PathContextオブジェクトでない: {type(path_context)}")
                        return False

                    # 各ベクトルの型を検証
                    vector_checks = [
                        (path_context.vector, (320,), "vector"),
                        (path_context.source_vector, (128,), "source_vector"),
                        (path_context.target_vector, (128,), "target_vector"),
                        (path_context.astpath_vector, (256,), "astpath_vector"),
                    ]

                    for vector, expected_shape, name in vector_checks:
                        if vector is not None:
                            if not self.validator.validate_numpy_array(
                                vector, expected_shape, f"PathContext[{pc_idx}].{name}"
                            ):
                                return False
                        else:
                            print(f"⚠️  PathContext[{pc_idx}].{name} is None")

            print("✅ InteractivePredictor出力型検証完了")
            return True

        except Exception as e:
            print(f"❌ InteractivePredictor出力型検証失敗: {e}")
            import traceback

            traceback.print_exc()
            return False

    def test_pickle_serialization_types(self) -> bool:
        """
        pickle直列化/逆直列化による型変換を検証

        この段階で numpy.ndarray → list 変換が発生する可能性があります。
        """
        print("\n=== pickle直列化/逆直列化型変換検証 ===")

        test_code = """
        public class PickleTest {
            public boolean validateInput(String data) {
                boolean isValid = data != null && !data.isEmpty();
                return isValid;
            }
        }
        """

        try:
            # オリジナルの結果を取得
            original_result = self.predictor.get(test_code)
            print("✅ オリジナル結果取得完了")

            # pickle直列化/逆直列化を実行
            serialized_data = pickle.dumps(original_result)
            deserialized_result = pickle.loads(serialized_data)
            print("✅ pickle直列化/逆直列化完了")

            # 型変換を検証
            for method_idx, (
                (orig_name, orig_vector, orig_contexts),
                (deser_name, deser_vector, deser_contexts),
            ) in enumerate(zip(original_result, deserialized_result)):
                print(f"\n  メソッド {method_idx}: {orig_name}")

                for pc_idx, (orig_pc, deser_pc) in enumerate(
                    zip(orig_contexts[:3], deser_contexts[:3])  # 最初の3つのみ
                ):
                    print(f"    PathContext {pc_idx}:")

                    # 各ベクトルの型変換を検証
                    vector_pairs = [
                        (orig_pc.vector, deser_pc.vector, "vector"),
                        (
                            orig_pc.source_vector,
                            deser_pc.source_vector,
                            "source_vector",
                        ),
                        (
                            orig_pc.target_vector,
                            deser_pc.target_vector,
                            "target_vector",
                        ),
                        (
                            orig_pc.astpath_vector,
                            deser_pc.astpath_vector,
                            "astpath_vector",
                        ),
                    ]

                    for orig_vec, deser_vec, name in vector_pairs:
                        if orig_vec is not None and deser_vec is not None:
                            print(f"      {name}: {type(orig_vec)} → {type(deser_vec)}")

                            # 型が変換されたかチェック
                            if isinstance(orig_vec, np.ndarray) and isinstance(
                                deser_vec, list
                            ):
                                print(
                                    f"      🔍 {name}: numpy配列がリストに変換されました"
                                )
                                if not self.validator.validate_list_conversion(
                                    orig_vec, deser_vec, f"{name}[{pc_idx}]"
                                ):
                                    return False
                            elif type(orig_vec) != type(deser_vec):
                                print(
                                    f"❌ {name}: 予期しない型変換 {type(orig_vec)} → {type(deser_vec)}"
                                )
                                return False
                            elif isinstance(orig_vec, np.ndarray) and isinstance(
                                deser_vec, np.ndarray
                            ):
                                if not np.allclose(orig_vec, deser_vec, rtol=1e-6):
                                    print(f"❌ {name}: 値が変化しました")
                                    return False
                        elif orig_vec is None and deser_vec is None:
                            continue
                        else:
                            print(
                                f"❌ {name}: Noneの不一致 (orig: {orig_vec is None}, deser: {deser_vec is None})"
                            )
                            return False

            print("✅ pickle直列化/逆直列化型変換検証完了")
            return True

        except Exception as e:
            print(f"❌ pickle直列化/逆直列化型変換検証失敗: {e}")
            import traceback

            traceback.print_exc()
            return False

    def test_flask_api_response_types(self) -> bool:
        """
        Flask API応答での型を検証

        実際のAPIエンドポイントを経由した場合の型変換を検証します。
        """
        print("\n=== Flask API応答型検証 ===")

        test_code = """
        public class APITest {
            public void executeTask(int taskId) {
                System.out.println("Executing task: " + taskId);
            }
        }
        """

        try:
            # Flask Wrapperを使ってAPI経由でデータを取得
            from flask_wrapper import FlaskWrapper

            flask_wrapper = FlaskWrapper()

            # APIからの応答を取得（これが問題の発生源の可能性が高い）
            api_result = flask_wrapper.get(test_code)
            print("✅ Flask API応答取得完了")

            # API応答の構造を検証
            for method_idx, (method_name, method_vector, path_contexts) in enumerate(
                api_result
            ):
                print(f"\n  メソッド {method_idx}: {method_name}")

                for pc_idx, path_context in enumerate(
                    path_contexts[:3]
                ):  # 最初の3つのみ
                    print(f"    PathContext {pc_idx} (type: {type(path_context)}):")

                    # 辞書形式かPathContextオブジェクトかを確認
                    if isinstance(path_context, dict):
                        # 辞書形式の場合
                        vector_keys = [
                            "vector",
                            "source_vector",
                            "target_vector",
                            "astpath_vector",
                        ]
                        for key in vector_keys:
                            if key in path_context and path_context[key] is not None:
                                vec_type = type(path_context[key])
                                print(f"      {key}: {vec_type}")

                                if isinstance(path_context[key], list):
                                    print(
                                        f"      🔍 {key}: リスト形式で受信（長さ: {len(path_context[key])}）"
                                    )
                                elif isinstance(path_context[key], np.ndarray):
                                    print(
                                        f"      ✅ {key}: numpy配列で受信（形状: {path_context[key].shape}）"
                                    )
                                else:
                                    print(f"      ❌ {key}: 予期しない型: {vec_type}")
                                    return False
                    else:
                        # PathContextオブジェクトの場合
                        from data.path_context import PathContext

                        if isinstance(path_context, PathContext):
                            # 各ベクトルの型をチェック
                            vector_attrs = [
                                (path_context.vector, "vector"),
                                (path_context.source_vector, "source_vector"),
                                (path_context.target_vector, "target_vector"),
                                (path_context.astpath_vector, "astpath_vector"),
                            ]

                            for vector, name in vector_attrs:
                                if vector is not None:
                                    vec_type = type(vector)
                                    print(f"      {name}: {vec_type}")

                                    if isinstance(vector, list):
                                        print(
                                            f"      🔍 {name}: リスト形式（長さ: {len(vector)}）"
                                        )
                                    elif isinstance(vector, np.ndarray):
                                        print(
                                            f"      ✅ {name}: numpy配列（形状: {vector.shape}）"
                                        )
                                    else:
                                        print(
                                            f"      ❌ {name}: 予期しない型: {vec_type}"
                                        )
                                        return False
                        else:
                            print(
                                f"❌ PathContextオブジェクトでも辞書でもない: {type(path_context)}"
                            )
                            return False

            print("✅ Flask API応答型検証完了")
            return True

        except Exception as e:
            print(f"❌ Flask API応答型検証失敗: {e}")
            import traceback

            traceback.print_exc()
            return False

    def run_all_tests(self) -> bool:
        """全ての型検証テストを実行"""
        print("==========================================")
        print("    code2seq ベクトル型検証テストスイート")
        print("==========================================")

        tests = [
            ("TensorFlowセッション型検証", self.test_tensorflow_session_vector_types),
            ("InteractivePredictor型検証", self.test_interactive_predict_output_types),
            ("pickle直列化型検証", self.test_pickle_serialization_types),
            ("Flask API型検証", self.test_flask_api_response_types),
        ]

        passed = 0
        total = len(tests)

        for test_name, test_func in tests:
            try:
                result = test_func()
                if result:
                    passed += 1
                    print(f"\n✅ {test_name}: 成功")
                else:
                    print(f"\n❌ {test_name}: 失敗")
            except Exception as e:
                print(f"\n💥 {test_name}: 例外発生 - {e}")

        print("\n==========================================")
        print(f"テスト結果: {passed}/{total} 成功")
        print("==========================================")

        return passed == total


def main() -> None:
    """メイン実行関数"""
    test_runner = TestVectorTypes()
    success = test_runner.run_all_tests()

    if success:
        print("\n🎉 全てのテストが成功しました！")
        exit(0)
    else:
        print("\n💥 一部のテストが失敗しました。")
        exit(1)


if __name__ == "__main__":
    main()
