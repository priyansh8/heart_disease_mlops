"""Unit tests for CNN training and evaluation"""
import numpy as np
import tensorflow as tf

from src.training import (
    build_simple_cnn,
    train_cnn,
    evaluate_model,
    save_model,
    load_model
)


def make_dummy_data(num_samples=10, image_size=(224, 224, 3)):
    X = np.random.rand(num_samples, *image_size).astype(np.float32)
    y = np.random.randint(0, 2, size=(num_samples,))
    return X, y


def test_build_and_train(tmp_path):
    X, y = make_dummy_data(8)
    # split manually
    X_train, X_val = X[:6], X[6:7]
    y_train, y_val = y[:6], y[6:7]

    model = build_simple_cnn(input_shape=X_train.shape[1:])
    assert isinstance(model, tf.keras.Model)

    model = train_cnn(model, X_train, y_train, X_val, y_val, epochs=1)
    assert hasattr(model, 'predict')


def test_evaluate_and_save(tmp_path):
    X, y = make_dummy_data(6)
    model = build_simple_cnn(input_shape=X.shape[1:])
    model = train_cnn(model, X, y, epochs=1)

    metrics = evaluate_model(model, X, y)
    assert 'accuracy' in metrics
    assert 0.0 <= metrics['accuracy'] <= 1.0

    model_path = tmp_path / "model.h5"
    save_model(model, str(model_path))
    assert model_path.exists()

    loaded = load_model(str(model_path))
    preds1 = model.predict(X)
    preds2 = loaded.predict(X)
    assert np.allclose(preds1, preds2, atol=1e-6)

