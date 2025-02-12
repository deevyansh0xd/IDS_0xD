# models/xlstm_model.py
import tensorflow as tf
from tensorflow.keras import layers, Sequential

def build_xlstm_model(input_shape, num_classes):
    model = Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Embedding(input_dim=500, output_dim=64),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.LSTM(64),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
