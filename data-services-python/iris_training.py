#!/usr/bin/env python3
"""
This script trains a K-Nearest Neighbors classifier on the Iris dataset,
optionally converts the trained model to ONNX format, and performs inference.
It supports the following command-line flags:
  --train      Train the model and save it.
  --convert    Convert the model to ONNX format.
  --predict    Prediction input as a comma-separated list of 4 numeric values.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import argparse

# For ONNX conversion
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------

MODEL_FILE = 'iris_knn_model.pkl'
ONNX_MODEL_FILE = 'iris_knn_model.onnx'
N_NEIGHBORS = 3

TEST_SIZE = 0.2
RANDOM_STATE = 42

#------------------------------------------------------------------------------
# Data loading and preparation
#------------------------------------------------------------------------------

def load_and_prepare_data():
    """
    Load the Iris dataset and return the features, labels, and target names.
    """
    iris = load_iris()
    X = iris.data            # Features
    y = iris.target          # Labels
    return X, y, iris.target_names

#------------------------------------------------------------------------------
# Model training and saving (joblib)
#------------------------------------------------------------------------------

def train_and_save_model(X, y):
    """
    Train the KNN model, evaluate its accuracy, save it as a pickle file,
    and return the trained model.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")

    joblib.dump(knn, MODEL_FILE)
    print("Model trained and saved as pickle.")

    return knn

#------------------------------------------------------------------------------
# ONNX model conversion and saving
#------------------------------------------------------------------------------

def convert_and_save_model(model):
    """
    Convert the given sklearn model to ONNX format and save it.
    """
    # Define the initial type for ONNX conversion (Iris features have 4 columns)
    initial_type = [('float_input', FloatTensorType([None, 4]))]
    # Disable zipmap option so that the output remains a tensor
    options = {id(model): {'zipmap': False}}

    onnx_model = convert_sklearn(model, initial_types=initial_type, options=options)

    with open(ONNX_MODEL_FILE, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print("ONNX model saved.")

#------------------------------------------------------------------------------
# Model loading and inference
#------------------------------------------------------------------------------

def load_and_predict_model(target_names, new_data):
    """
    Load the trained model from file and perform prediction on the provided new_data.
    """
    knn = joblib.load(MODEL_FILE)
    print("Model loaded from pickle.")

    prediction = knn.predict(new_data)
    # Assuming prediction returns an array, extract the first element.
    predicted_class = target_names[prediction[0]]
    print("Prediction:", predicted_class)

#------------------------------------------------------------------------------
# Main processing
#------------------------------------------------------------------------------

def main(train_model, convert_to_onnx, predict_input=None):
    """
    Main processing function:
      - If train_model is True, train and save the model.
        If convert_to_onnx is also True, convert the trained model to ONNX.
      - If train_model is False and convert_to_onnx is True, load an existing model 
        and convert it to ONNX.
      - Otherwise, perform inference using the saved model. If predict_input is provided,
        it will be used as prediction input; otherwise, a default sample input will be used.
    """
    X, y, target_names = load_and_prepare_data()

    if train_model:
        model = train_and_save_model(X, y)
        if convert_to_onnx:
            convert_and_save_model(model)
    elif convert_to_onnx:
        try:
            model = joblib.load(MODEL_FILE)
            print("Loaded model for ONNX conversion from pickle.")
            convert_and_save_model(model)
        except Exception as e:
            print("Error loading model for conversion:", e)
    else:
        if predict_input is not None:
            try:
                # Parse comma-separated numbers into a list of floats
                features = list(map(float, predict_input.split(',')))
                if len(features) != 4:
                    raise ValueError("Exactly 4 numeric features are required for prediction.")
                new_data = [features]
            except Exception as e:
                print("Error parsing prediction input:", e)
                return
        else:
            # Default input if no prediction input is provided
            new_data = [[5.1, 3.5, 1.4, 0.2]]
        load_and_predict_model(target_names, new_data)

#------------------------------------------------------------------------------
# Entry point
#------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Iris KNN model training, conversion to ONNX, and inference"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Set this flag to train the model"
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Set this flag to convert the model to ONNX format"
    )
    parser.add_argument(
        "--predict",
        type=str,
        help="Comma-separated numeric features for prediction, e.g., '5.1,3.5,1.4,0.2'"
    )
    args = parser.parse_args()

    main(args.train, args.convert, args.predict)