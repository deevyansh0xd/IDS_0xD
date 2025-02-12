# models/train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from xlstm_model import build_xlstm_model

def train_model():
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

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train model
    model = build_xlstm_model(input_shape=(X_train.shape[1],), num_classes=len(np.unique(y)))
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

    # Save the model
    model.save('models/xlstm_model.h5')
    print("Model saved as 'models/xlstm_model.h5'.")

if __name__ == "__main__":
    train_model()
