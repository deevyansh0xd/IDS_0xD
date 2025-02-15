# models/evaluate_model.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf

def evaluate_model():
    # Load dataset
    data = pd.read_csv('data/kddcup99.csv', header=None)

    # Columns based on KDD CUP 99 dataset
    feature_columns = list(range(41))
    label_column = 41

    # Split features and labels
    X = data[feature_columns].values
    y = data[label_column].values

    # Preprocess labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Load the model
    model = tf.keras.models.load_model('models/xlstm_model.h5')

    # Predict and evaluate
    y_pred = model.predict(X)
    y_pred_classes = y_pred.argmax(axis=1)

    print(classification_report(y, y_pred_classes))

if __name__ == "__main__":
    evaluate_model()
