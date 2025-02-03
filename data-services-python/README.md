# data-services-python

## 環境構築ガイド

### 1. システム更新とPythonインストール
```bash
# システムパッケージの更新
sudo apt update

# Python3と関連ツールのインストール
sudo apt install python3 python3-pip python-is-python3
```

### 2. バージョン確認
```bash
# Pythonバージョンの確認
python3 --version

# pipバージョンの確認
pip3 --version
```

### 3. 仮想環境のセットアップ
```bash
# 仮想環境の作成
python3 -m venv myenv

# 仮想環境の有効化
source myenv/bin/activate
```

### 4. 必要なパッケージのインストール
```bash
# データ分析関連パッケージのインストール
pip3 install scikit-learn numpy pandas matplotlib
```

## 注意事項
- `.gitignore`に`myenv/`を追加することを推奨
- `requirements.txt`の生成：
```bash
pip freeze > requirements.txt
```

## 開発環境
- Python 3.x
- scikit-learn
- numpy
- pandas
- matplotlib

この構成により、データ分析やML開発のための基本的な環境が整います。