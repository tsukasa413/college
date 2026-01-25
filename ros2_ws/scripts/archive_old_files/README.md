# このフォルダ内のファイルは新しい統合ファイルに移行されました

## ⚠️ 注意: これらはアーカイブファイルです

このディレクトリ内のファイルは**使用非推奨**です。  
全ての機能は新しい統合ファイルに移行されています。

---

## 🔍 自分のファイルがどこに移行したか確認する方法

### クイックリファレンス

親ディレクトリの **[FILE_MIGRATION_MAP.md](../FILE_MIGRATION_MAP.md)** を参照してください。  
このファイルには全ての旧ファイル→新ファイルのマッピングが記載されています。

```bash
# マッピングファイルを開く
cat ../FILE_MIGRATION_MAP.md

# 特定のファイルを検索
grep "あなたのファイル名" ../FILE_MIGRATION_MAP.md
```

---

## 📋 簡易マッピング表

| 旧ファイル名 | 新ファイルの場所 | カテゴリ |
|------------|----------------|---------|
| `test_cuda_basic.cu` | `../tests/unit_tests/test_cpp_units.cpp` | C++ユニットテスト |
| `test_struct_size.cpp` | `../tests/unit_tests/test_cpp_units.cpp` | C++ユニットテスト |
| `test_all_utils.cpp` | `../tests/unit_tests/test_cpp_units.cpp` | C++ユニットテスト |
| `test_depth_estimation.cpp` | `../tests/unit_tests/test_cpp_units.cpp` | C++ユニットテスト |
| `test_gpu_kernel.cpp` | `../tests/unit_tests/test_cpp_units.cpp` | C++ユニットテスト |
| `test_geometry.py` | `../tests/unit_tests/test_python_units.py` | Pythonユニットテスト |
| `test_depth_estimation_python.py` | `../tests/unit_tests/test_python_units.py` | Pythonユニットテスト |
| `verify_utils.py` | `../tests/unit_tests/test_python_units.py` | Pythonユニットテスト |
| `equivalence_test.py` | `../tests/integration_tests/integration_test_suite.py` | 統合テスト |
| `compare_detailed.py` | `../tests/integration_tests/integration_test_suite.py` | 統合テスト |
| `verify_equivalence.py` | `../tests/verification_tests/verify_implementation_equivalence.py` | 検証テスト |
| `verify_equivalence_minimal.py` | `../tests/verification_tests/verify_implementation_equivalence.py` | 検証テスト |
| `verify_isb_filter.py` | `../tests/verification_tests/verify_implementation_equivalence.py` | 検証テスト |
| `verify_stitcher.py` | `../tests/verification_tests/verify_implementation_equivalence.py` | 検証テスト |
| `analyze_distance_parameterization.py` | `../tests/analysis_tools/analyze_depth_estimation_suite.py` | 解析ツール |
| `analyze_cost_computation.py` | `../tests/analysis_tools/analyze_depth_estimation_suite.py` | 解析ツール |
| `debug_rt_matrix.py` | `../tests/analysis_tools/analyze_depth_estimation_suite.py` | 解析ツール |
| `debug_isb_difference.py` | `../tests/analysis_tools/analyze_depth_estimation_suite.py` | 解析ツール |
| `run_verification.sh` | `../tests/utils/run_unified_tests.sh` | シェルスクリプト |
| `run_verify_isb.sh` | `../tests/utils/run_unified_tests.sh` | シェルスクリプト |

---

## 🚀 新しいテスト実行方法

### 全テストを実行

```bash
cd /home/motoken/college/ros2_ws/scripts
./tests/utils/run_unified_tests.sh
```

### 特定のカテゴリを実行

```bash
# ユニットテストのみ
./tests/utils/run_unified_tests.sh --test-suite unit

# 検証テストのみ
./tests/utils/run_unified_tests.sh --test-suite verification

# 解析ツールのみ
./tests/utils/run_unified_tests.sh --test-suite analysis

# 統合テストのみ
./tests/utils/run_unified_tests.sh --test-suite integration
```

### 個別ファイルを直接実行

```bash
# Pythonユニットテスト
python3 tests/unit_tests/test_python_units.py

# 検証テスト（ミニマルモード）
python3 tests/verification_tests/verify_implementation_equivalence.py --minimal

# 解析ツール
python3 tests/analysis_tools/analyze_depth_estimation_suite.py
```

---

## ❓ よくある質問

### Q: 旧ファイルをまだ使えますか？

**A**: 技術的には実行可能ですが、**非推奨**です。理由：
- メンテナンスされていない
- バグ修正や機能追加が反映されない
- 新しいテストスイートとの互換性がない

### Q: 旧ファイルは削除して良いですか？

**A**: このアーカイブフォルダは参照用に保持することをお勧めします。  
ただし、新しいファイルに完全に移行した後は削除しても問題ありません。

### Q: 旧ファイルの特定の機能が見つからない

**A**: 
1. [FILE_MIGRATION_MAP.md](../FILE_MIGRATION_MAP.md) で検索
2. 新しいファイルのソースコードを確認
3. それでも見つからない場合は、プロジェクト管理者に連絡

### Q: 新しいファイルの使い方がわからない

**A**: 各新ファイルには詳細なdocstringとヘルプが含まれています：
```bash
python3 新ファイル名.py --help
```

---

## 📖 詳細ドキュメント

より詳しい情報は以下を参照：

- **[FILE_MIGRATION_MAP.md](../FILE_MIGRATION_MAP.md)**: 完全な移行マップ
- **[REORGANIZATION_GUIDE.md](../REORGANIZATION_GUIDE.md)**: 再構成の背景と方針
- **[TEST_EXECUTION_REPORT.md](../TEST_EXECUTION_REPORT.md)**: テスト実行結果
- **[DETAILED_TEST_ANALYSIS.md](../DETAILED_TEST_ANALYSIS.md)**: 詳細な解析と考察

---

**最終更新**: 2026-01-25  
**メンテナンス状態**: ⚠️ アーカイブのみ（更新なし）  
**推奨アクション**: 新しい統合ファイルを使用してください