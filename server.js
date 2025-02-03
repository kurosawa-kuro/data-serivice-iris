// Express and ONNX inference server
// This file implements all responsibilities using SRP in a single file

import express from 'express';
import { InferenceSession, Tensor } from 'onnxruntime-node';

const app = express();

// Global session variable for ONNX model
let session = null;

// Classification labels corresponding to output indices.
const CLASSES = ['setosa', 'versicolor', 'virginica'];

/**
 * Loads the ONNX model asynchronously and initializes the global session.
 */
async function loadModel() {
  try {
    session = await InferenceSession.create('iris_knn_model.onnx');
    console.log('Model loaded successfully.');
  } catch (e) {
    console.error('Failed to load model:', e);
  }
}

/**
 * Serializes a tensor output by converting BigInt values to Number.
 * @param {object} tensorOutput - The output tensor from inference.
 * @returns {Array} - Array of converted tensor output values.
 */
function serializeTensorOutput(tensorOutput) {
  const rawData = tensorOutput.cpuData || tensorOutput.data;
  return Array.from(rawData).map(value =>
    typeof value === 'bigint' ? Number(value) : value
  );
}

/**
 * Prepares the input tensor for inference.
 * @returns {Tensor} - The Tensor representing the input data.
 */
function prepareInputTensor() {
  const data = new Float32Array([5.1, 3.5, 1.4, 0.2]);
  return new Tensor('float32', data, [1, 4]);
}

/**
 * Performs model inference using the given input tensor.
 * @param {Tensor} inputTensor - The input tensor.
 * @returns {object} - The raw inference results.
 */
async function performInference(inputTensor) {
  if (!session) {
    throw new Error('Model not loaded');
  }
  const inputName = session.inputNames[0];
  return await session.run({ [inputName]: inputTensor });
}

/**
 * Converts the raw inference output into a human-readable format.
 * @param {object} results - The raw inference results.
 * @returns {object} - The formatted inference result.
 */
function formatInferenceOutput(results) {
  const labelData = serializeTensorOutput(results['label']);
  const probabilitiesData = serializeTensorOutput(results['probabilities']);
  const predictedClass = CLASSES[labelData[0]];

  // Convert probability to percentage string.
  const probabilities = {};
  for (let i = 0; i < probabilitiesData.length; i++) {
    probabilities[CLASSES[i]] = `${(probabilitiesData[i] * 100).toFixed(0)}%`;
  }

  return {
    predicted: predictedClass,
    probabilities: probabilities,
  };
}

/**
 * Sends a JSON response.
 * @param {object} res - The Express response object.
 * @param {object} data - The data to send.
 */
function sendJSONResponse(res, data) {
  res.setHeader('Content-Type', 'application/json');
  res.end(JSON.stringify(data));
}

app.get('/iris', async (req, res) => {
  if (!session) {
    return res.status(503).send('Model not ready');
  }
  
  const inputTensor = prepareInputTensor();
  try {
    const results = await performInference(inputTensor);
    const humanReadableResponse = formatInferenceOutput(results);
    sendJSONResponse(res, humanReadableResponse);
  } catch (e) {
    console.error('Inference failed:', e);
    res.status(500).send('Inference error');
  }
});

// Start the server.
app.listen(3000, () => {
  console.log('Server is running on port 3000');
});

// Load the ONNX model on startup.
loadModel();