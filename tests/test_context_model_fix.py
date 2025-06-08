"""
context_model.py修正検証テスト

eye2vecのcontext_model.pyで行ったベクトル型変換修正が
正しく動作するかを検証します。
"""

import sys
from pathlib import Path

import numpy as np

# プロジェクトルートをsys.pathに追加
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# eye2vecモジュールのインポート
from eye2vec.src.eye2vec.context_model import ContextModel


def create_mock_path_context_dict(use_list_vectors: bool = True) -> dict:
    """テスト用のpath_context辞書を作成"""

    # テストデータ作成
    vector_data = np.random.random(320).astype(np.float32)
    source_vector_data = np.random.random(128).astype(np.float32)
    target_vector_data = np.random.random(128).astype(np.float32)
    astpath_vector_data = np.random.random(256).astype(np.float32)

    if use_list_vectors:
        # リスト形式（問題のあるケース）
        vector = vector_data.tolist()
        source_vector = source_vector_data.tolist()
        target_vector = target_vector_data.tolist()
        astpath_vector = astpath_vector_data.tolist()
    else:
        # numpy配列形式（正常ケース）
        vector = vector_data
        source_vector = source_vector_data
        target_vector = target_vector_data
        astpath_vector = astpath_vector_data

    return {
        "source": "getValue",
        "target": "returnStatement",
        "path": "MethodDecl|Body|Return",
        "attention": 0.85,
        "vector": vector,
        "source_vector": source_vector,
        "target_vector": target_vector,
        "astpath_vector": astpath_vector,
        "lineColumns": [1, 5, 1, 12, 2, 8, 2, 15],  # begin/end座標
    }


def test_list_vector_conversion():
    """リスト形式ベクトルの変換テスト"""
    print("=== リスト→numpy配列変換テスト ===")

    # リスト形式のpath_context辞書を作成
    path_context_dict = create_mock_path_context_dict(use_list_vectors=True)

    print(f"変換前 vector型: {type(path_context_dict['vector'])}")
    print(f"変換前 source_vector型: {type(path_context_dict['source_vector'])}")

    try:
        # ContextModel._get_methodを直接テスト
        method = ContextModel._get_method(
            "testMethod",
            np.random.random(128).astype(np.float32),  # method_vector
            [path_context_dict],
        )

        # PathContextオブジェクトを取得
        pc = method.path_context_list[0]

        print(f"変換後 vector型: {type(pc.vector)}")
        print(
            f"変換後 source.vector型: {type(pc.source.vector) if pc.source.vector is not None else 'None'}"
        )
        print(
            f"変換後 target.vector型: {type(pc.target.vector) if pc.target.vector is not None else 'None'}"
        )
        print(
            f"変換後 astpath.vector型: {type(pc.astpath.vector) if pc.astpath.vector is not None else 'None'}"
        )

        # tobytes()メソッドのテスト（これが以前エラーになっていた）
        vector_hash = pc.vec_hash
        print(f"✅ vec_hash取得成功: {vector_hash[:10]}...")

        return True

    except Exception as e:
        print(f"❌ テスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_numpy_vector_preservation():
    """numpy配列形式ベクトルの保持テスト"""
    print("\n=== numpy配列保持テスト ===")

    # numpy配列形式のpath_context辞書を作成
    path_context_dict = create_mock_path_context_dict(use_list_vectors=False)

    original_vector = path_context_dict["vector"].copy()
    print(f"オリジナル vector型: {type(path_context_dict['vector'])}")

    try:
        # ContextModel._get_methodを直接テスト
        method = ContextModel._get_method(
            "testMethod",
            np.random.random(128).astype(np.float32),  # method_vector
            [path_context_dict],
        )

        # PathContextオブジェクトを取得
        pc = method.path_context_list[0]

        print(f"処理後 vector型: {type(pc.vector)}")

        # 値が保持されているかチェック
        if np.allclose(pc.vector, original_vector, rtol=1e-6):
            print("✅ ベクトル値が正しく保持されています")
        else:
            print("❌ ベクトル値が変更されています")
            return False

        # tobytes()メソッドのテスト
        vector_hash = pc.vec_hash
        print(f"✅ vec_hash取得成功: {vector_hash[:10]}...")

        return True

    except Exception as e:
        print(f"❌ テスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_none_vector_handling():
    """None値ベクトルの処理テスト"""
    print("\n=== None値ベクトル処理テスト ===")

    # None値を含むpath_context辞書を作成
    path_context_dict = create_mock_path_context_dict(use_list_vectors=True)
    path_context_dict["source_vector"] = None
    path_context_dict["target_vector"] = None
    path_context_dict["astpath_vector"] = None

    try:
        # ContextModel._get_methodを直接テスト
        method = ContextModel._get_method(
            "testMethod",
            np.random.random(128).astype(np.float32),  # method_vector
            [path_context_dict],
        )

        # PathContextオブジェクトを取得
        pc = method.path_context_list[0]

        print(f"vector型: {type(pc.vector)}")
        print(f"source.vector: {pc.source.vector}")
        print(f"target.vector: {pc.target.vector}")
        print(f"astpath.vector: {pc.astpath.vector}")

        # メインベクトルは変換されているはず
        if isinstance(pc.vector, np.ndarray):
            print("✅ メインベクトルは正しくnumpy配列に変換されています")
        else:
            print("❌ メインベクトルがnumpy配列ではありません")
            return False

        # tobytes()メソッドのテスト
        vector_hash = pc.vec_hash
        print(f"✅ vec_hash取得成功: {vector_hash[:10]}...")

        return True

    except Exception as e:
        print(f"❌ テスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """メイン実行関数"""
    print("context_model.py修正検証テスト実行中...")

    tests = [
        ("リスト→numpy配列変換", test_list_vector_conversion),
        ("numpy配列保持", test_numpy_vector_preservation),
        ("None値処理", test_none_vector_handling),
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

    print("\n" + "=" * 50)
    print(f"テスト結果: {passed}/{total} 成功")
    print("=" * 50)

    if passed == total:
        print("\n🎉 全ての修正が正常に動作しています！")
        print("'list' object has no attribute 'tobytes' エラーは解決されました。")
        return True
    else:
        print("\n❌ 一部の修正に問題があります。")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
