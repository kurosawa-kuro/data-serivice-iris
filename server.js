// express 
// get helloworld

import express from 'express';
const app = express();

app.get('/', (req, res) => {
  res.send('Hello World');
});

// Import ONNX Runtime components
import { InferenceSession, Tensor } from 'onnxruntime-node';
import fs from 'fs';

// モデルを非同期で読み込む
let session;
(async () => {
  try {
    session = await InferenceSession.create('iris_knn_model.onnx');
  } catch (e) {
    console.error('Failed to load model:', e);
  }
})();

// Helper function to serialize the output tensor by converting BigInt values to Number
function serializeTensorOutput(tensorOutput) {
  const rawData = tensorOutput.cpuData || tensorOutput.data;
  return Array.from(rawData).map(value =>
    typeof value === 'bigint' ? Number(value) : value
  );
}

app.get('/iris', async (req, res) => {
  if (!session) {
    return res.status(503).send('Model not ready');
  }
  
  // 入力データの準備（Float32Arrayを使用）
  const data = new Float32Array([5.1, 3.5, 1.4, 0.2]);
  const tensor = new Tensor('float32', data, [1, 4]);

  try {
    const inputName = session.inputNames[0];
    console.log('Session input names:', session.inputNames);
    console.log('Session output names:', session.outputNames);
    console.log('Session input shapes:', session.inputShapes);
    console.log('Session output shapes:', session.outputShapes);

    // 推論実行
    const results = await session.run({ [inputName]: tensor });

    // 出力結果をhuman-readableな形式に変換
    // 数値ラベルを直接返すのではなく、クラス名に変換する
    const classes = ["setosa", "versicolor", "virginica"];
    const labelData = serializeTensorOutput(results["label"]);
    const probabilitiesData = serializeTensorOutput(results["probabilities"]);

    // ラベルの数値をそのままクラス名へ変換（例: 0 -> "setosa"）
    const predictedClass = classes[labelData[0]];

    // 確率をパーセンテージ表記に変換
    const probabilities = {};
    for (let i = 0; i < probabilitiesData.length; i++) {
      // 小数部を100倍して四捨五入。例: 1 -> 100%
      probabilities[classes[i]] = `${(probabilitiesData[i] * 100).toFixed(0)}%`;
    }

    // human-readableな結果オブジェクトを作成
    const humanReadableResponse = {
      predicted: predictedClass,
      probabilities: probabilities
    };

    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify(humanReadableResponse));
  } catch (e) {
    console.error('Inference failed:', e);
    res.status(500).send('Inference error');
  }
});

app.listen(3000, () => {
  console.log('Server is running on port 3000');
});