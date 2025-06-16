#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np

# プロジェクトのルートディレクトリをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from code2seq.data.path_context import PathContext


def test_path_context():
    """PathContextクラスの基本動作テスト"""
    print("=== PathContextクラステスト ===")

    # テストデータの作成
    source = "testSource"
    short_path = "testPath"
    target = "testTarget"
    attention = 0.85
    vector = np.random.rand(320).astype(np.float32)
    source_vector = np.random.rand(128).astype(np.float32)
    target_vector = np.random.rand(128).astype(np.float32)
    astpath_vector = np.random.rand(256).astype(np.float32)

    # PathContextオブジェクトの作成
    try:
        pc = PathContext(
            source=source,
            short_path=short_path,
            target=target,
            attention=attention,
            vector=vector,
            source_vector=source_vector,
            target_vector=target_vector,
            astpath_vector=astpath_vector,
        )
        print("✅ PathContextオブジェクトの作成: 成功")
    except Exception as e:
        print(f"❌ PathContextオブジェクトの作成: 失敗 - {e}")
        return False

    # プロパティアクセステスト
    try:
        print(f"✅ source: {pc.source}")
        print(f"✅ target: {pc.target}")
        print(f"✅ path: {pc.path}")  # aliasテスト
        print(f"✅ short_path: {pc.short_path}")
        print(f"✅ attention: {pc.attention}")
        print(f"✅ lineColumns: '{pc.lineColumns}'")  # 空文字列テスト
        print("✅ プロパティアクセス: 成功")
    except Exception as e:
        print(f"❌ プロパティアクセス: 失敗 - {e}")
        return False

    # ベクトルの型と次元チェック
    vectors_to_check = {
        "vector": (pc.vector, 320),
        "source_vector": (pc.source_vector, 128),
        "target_vector": (pc.target_vector, 128),
        "astpath_vector": (pc.astpath_vector, 256),
    }

    for vec_name, (vec_data, expected_dim) in vectors_to_check.items():
        try:
            # 型チェック
            if not isinstance(vec_data, np.ndarray):
                print(f"❌ {vec_name}: 型エラー - {type(vec_data)}")
                return False

            # 次元チェック
            if vec_data.shape != (expected_dim,):
                print(f"❌ {vec_name}: 次元エラー - {vec_data.shape}")
                return False

            # データ型チェック
            if not np.issubdtype(vec_data.dtype, np.floating):
                print(f"❌ {vec_name}: データ型エラー - {vec_data.dtype}")
                return False

            print(
                f"✅ {vec_name}: 型={type(vec_data).__name__}, 次元={vec_data.shape}, dtype={vec_data.dtype}"
            )
        except Exception as e:
            print(f"❌ {vec_name}: ベクトル検証失敗 - {e}")
            return False

    # validate_vectorsメソッドテスト
    try:
        pc.validate_vectors()
        print("✅ validate_vectors(): 成功")
    except Exception as e:
        print(f"❌ validate_vectors(): 失敗 - {e}")
        return False

    # get_keyメソッドテスト
    try:
        key = pc.get_key()
        expected_key = (source, short_path, target)
        if key == expected_key:
            print(f"✅ get_key(): {key}")
        else:
            print(f"❌ get_key(): 期待値={expected_key}, 実際={key}")
            return False
    except Exception as e:
        print(f"❌ get_key(): 失敗 - {e}")
        return False

    print("\n🎉 すべてのテストが成功しました！")
    print("\n--- 設計の確認 ---")
    print("✅ PathContextクラスは4つのベクトルを持ちます：")
    print("  - vector (320D): パスコンテキスト統合ベクトル")
    print("  - source_vector (128D): ソース単語ベクトル")
    print("  - target_vector (128D): ターゲット単語ベクトル")
    print("  - astpath_vector (256D): ASTパスベクトル")
    print("✅ 全てのベクトルはnumpy配列として保持されます")
    print("✅ テスト互換性のため path, lineColumns プロパティも提供されます")

    return True


if __name__ == "__main__":
    test_path_context()
