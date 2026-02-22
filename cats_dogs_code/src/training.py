"""Model training and evaluation functions for image classification.

The previous tabular utilities were removed. This module now contains
logic to build a simple convolutional neural network using TensorFlow
and to save/load Keras models.
"""

import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)


def build_simple_cnn(input_shape: Tuple[int, int, int] = (224, 224, 3)) -> tf.keras.Model:
    """Construct a minimal CNN suitable for cats-vs-dogs binary classification."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_cnn(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    epochs: int = 5
) -> tf.keras.Model:
    """Train a CNN model with optional validation data."""
    if X_val is not None and y_val is not None:
        model.fit(X_train, y_train, epochs=epochs,
                  validation_data=(X_val, y_val))
    else:
        model.fit(X_train, y_train, epochs=epochs)
    return model


def evaluate_model(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate model performance on test data and return metrics."""
    probs = model.predict(X_test).flatten()
    preds = (probs >= 0.5).astype(int)
    metrics = {
        'accuracy': float(accuracy_score(y_test, preds)),
        'precision': float(precision_score(y_test, preds)),
        'recall': float(recall_score(y_test, preds)),
        'f1_score': float(f1_score(y_test, preds)),
        'roc_auc': float(roc_auc_score(y_test, probs))
    }
    return metrics


def save_model(model: tf.keras.Model, filepath: str):
    """Save Keras model to HDF5 file."""
    model.save(filepath)


def load_model(filepath: str) -> tf.keras.Model:
    """Load Keras model from disk."""
    return tf.keras.models.load_model(filepath)

