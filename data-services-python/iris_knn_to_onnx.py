from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# モデルの訓練
iris = load_iris()
X, y = iris.data, iris.target
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# 入力型を定義
initial_type = [('float_input', FloatTensorType([None, 4]))]
# オプションでzipmapを無効にする（出力をtensorとして残す）
options = {id(model): {'zipmap': False}}

# ONNX形式に変換
onnx_model = convert_sklearn(model, initial_types=initial_type, options=options)

# モデルを保存
with open("iris_knn_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("モデルを保存しました。")