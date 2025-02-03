# python3 iris_training_and_inference.py --train

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import argparse

#------------------------------------------------------------------------------
# 定数
#------------------------------------------------------------------------------

# モデル関連
MODEL_FILE = 'iris_knn_model.pkl'
N_NEIGHBORS = 3

# データ関連
TEST_SIZE = 0.2
RANDOM_STATE = 42

#------------------------------------------------------------------------------
# データの読み込みと準備
#------------------------------------------------------------------------------

def load_and_prepare_data():
    """Iris データセットを読み込み、特徴量とラベルを返す"""
    iris = load_iris()
    X = iris.data  # 特徴量
    y = iris.target  # ラベル
    return X, y, iris.target_names

#------------------------------------------------------------------------------
# モデルの训练と保存
#------------------------------------------------------------------------------

def train_and_save_model(X, y):
    """モデルを训练し、保存する"""
    # データセットを训练用とテスト用に分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # K-Nearest Neighbors (KNN) モデルを作成
    knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)

    # モデルを训练
    knn.fit(X_train, y_train)

    # テストデータで精度を評価
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"モデルの精度: {accuracy:.2f}")

    # モデルを保存
    joblib.dump(knn, MODEL_FILE)
    print("モデルを训练して保存しました。")

#------------------------------------------------------------------------------
# モデルの読み込みと推論
#------------------------------------------------------------------------------

def load_and_predict_model(X, target_names, new_data):
    """モデルを読み込み、新しいデータに対して推論する"""
    # モデルを読み込み
    knn = joblib.load(MODEL_FILE)
    print("モデルを読み込みました。")

    # 新しいデータに対する推論
    prediction = knn.predict(new_data)

    # 予測結果を表示
    print(f"予測結果: {target_names[prediction][0]}")

#------------------------------------------------------------------------------
# メイン処理
#------------------------------------------------------------------------------

def main(train_model):
    """メイン処理: フラグに応じてモデルの训练または推論を行う"""
    # データの読み込み
    X, y, target_names = load_and_prepare_data()

    # フラグに応じて処理を分岐
    if train_model:
        train_and_save_model(X, y)
    else:
        # 新しいデータを定義
        new_data = [[5.1, 3.5, 1.4, 0.2]]  # 例: setosa に近いデータ
        load_and_predict_model(X, target_names, new_data)

#------------------------------------------------------------------------------
# エントリーポイント
#------------------------------------------------------------------------------

if __name__ == "__main__":
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description="Iris データセットのモデル訓練と推論")
    parser.add_argument(
        "--train",
        action="store_true",
        help="モデルを訓練する場合はこのフラグを指定"
    )
    args = parser.parse_args()

    # メイン処理を実行
    main(args.train)