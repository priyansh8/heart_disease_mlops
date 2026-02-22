"""
Image classification training pipeline for Cats vs Dogs dataset.
Saves a simple CNN model and configuration necessary for inference.

Features:
- Loads and preprocesses images to 224x224 RGB
- Splits dataset into train/validation/test (80%/10%/10%)
- Applies data augmentation to training set
- Trains CNN model with validation
- Logs parameters, metrics, and artifacts to MLflow
- Saves model in .h5 format
"""
import sys
import os
import json
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# add package path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cats_dogs_code'))

from src.preprocessing import (
    load_images_from_folder,
    split_dataset,
    augment_dataset,
    get_augmentation_metadata
)
from src.training import build_simple_cnn, train_cnn, evaluate_model, save_model


def plot_confusion_matrix(y_true, y_pred, output_path='confusion_matrix.png'):
    """Generate and save a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Cat', 'Dog'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title('Confusion Matrix - Cats vs Dogs')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close(fig)
    return output_path


def plot_training_history(history, output_path='training_curves.png'):
    """Generate and save training loss/accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    axes[0].plot(history.history['loss'], label='Train Loss', marker='o')
    if 'val_loss' in history.history:
        axes[0].plot(history.history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_title('Loss Curve')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy curve
    axes[1].plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    if 'val_accuracy' in history.history:
        axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', marker='s')
    axes[1].set_title('Accuracy Curve')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle('Training History - Cats vs Dogs CNN')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close(fig)
    return output_path


def train_with_history(model, X_train, y_train, X_val, y_val, epochs=5):
    """Train model and return both model and history."""
    if X_val is not None and y_val is not None:
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_data=(X_val, y_val),
            verbose=1
        )
    else:
        history = model.fit(X_train, y_train, epochs=epochs, verbose=1)
    return model, history


def main():
    print("=" * 70)
    print("STARTING IMAGE TRAINING PIPELINE - CATS VS DOGS CLASSIFICATION")
    print("=" * 70)

    # ── MLflow experiment setup ──────────────────────────────────────────────
    mlflow.set_experiment("cats_vs_dogs_classification")

    with mlflow.start_run(run_name=f"cnn_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

        # ── Hyperparameters to log ───────────────────────────────────────────
        EPOCHS = 5
        AUGMENTATION_MULTIPLIER = 2
        AUGMENTATION_PROB = 0.5
        TEST_SIZE = 0.1
        VAL_SIZE = 0.1
        IMAGE_SIZE = 224
        RANDOM_STATE = 42
        DATA_DIR = './cats_dogs_dataset'

        mlflow.log_params({
            "epochs": EPOCHS,
            "augmentation_multiplier": AUGMENTATION_MULTIPLIER,
            "augmentation_prob": AUGMENTATION_PROB,
            "test_size": TEST_SIZE,
            "val_size": VAL_SIZE,
            "image_size": IMAGE_SIZE,
            "random_state": RANDOM_STATE,
            "model_type": "Simple CNN",
            "optimizer": "adam",
            "loss_function": "binary_crossentropy",
            "data_source": DATA_DIR
        })

        # 1. Load and preprocess images
        print("\n[1/6] Loading and preprocessing images...")
        X, y = load_images_from_folder(DATA_DIR)
        print(f"Loaded {len(X)} images")
        if len(X) == 0:
            print("ERROR: No images found in cats_dogs_dataset/")
            print("Ensure it contains subdirectories 'cats/' and 'dogs/' with images")
            return

        mlflow.log_param("total_samples", len(X))

        # 2. Split dataset (80% train, 10% val, 10% test)
        print("\n[2/6] Splitting dataset (80% train, 10% val, 10% test)...")
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
            X, y, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE
        )
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        mlflow.log_params({
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test)
        })

        # 3. Apply data augmentation to training set
        print("\n[3/6] Applying data augmentation to training set...")
        X_train_aug, y_train_aug = augment_dataset(
            X_train, y_train,
            augmentation_multiplier=AUGMENTATION_MULTIPLIER,
            augmentation_prob=AUGMENTATION_PROB,
            random_state=RANDOM_STATE
        )
        print(f"After augmentation - Train: {len(X_train_aug)} (original: {len(X_train)})")
        mlflow.log_param("train_after_augmentation", len(X_train_aug))

        # Log augmentation techniques as tags
        aug_meta = get_augmentation_metadata()
        mlflow.set_tag("augmentation_techniques",
                       ", ".join(aug_meta["augmentation_techniques"]))

        # 4. Build model
        print("\n[4/6] Building CNN model...")
        model = build_simple_cnn(input_shape=X_train_aug.shape[1:])
        print("Model architecture:")
        model.summary()

        # 5. Train model (capture history for curves)
        print("\n[5/6] Training model with augmented data...")
        model, history = train_with_history(
            model, X_train_aug, y_train_aug, X_val, y_val, epochs=EPOCHS
        )

        # Log per-epoch metrics to MLflow
        for epoch_idx, (loss, acc) in enumerate(
            zip(history.history['loss'], history.history['accuracy'])
        ):
            mlflow.log_metrics({
                "train_loss": loss,
                "train_accuracy": acc
            }, step=epoch_idx)
        if 'val_loss' in history.history:
            for epoch_idx, (val_loss, val_acc) in enumerate(
                zip(history.history['val_loss'], history.history['val_accuracy'])
            ):
                mlflow.log_metrics({
                    "val_loss": val_loss,
                    "val_accuracy": val_acc
                }, step=epoch_idx)

        # 6. Evaluate model on test set
        print("\n[6/6] Evaluating model on test set...")
        metrics = evaluate_model(model, X_test, y_test)
        print("Evaluation metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        # Log final test metrics
        mlflow.log_metrics({
            "test_accuracy": metrics["accuracy"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_f1_score": metrics["f1_score"],
            "test_roc_auc": metrics["roc_auc"]
        })

        # 7. Save model artifact
        model_path = 'cat_dog_model.h5'
        save_model(model, model_path)
        print(f"\nModel saved to {model_path}")
        mlflow.log_artifact(model_path, artifact_path="model")

        # Also log model with MLflow's keras integration for model registry
        mlflow.keras.log_model(model, artifact_path="keras_model")

        # 8. Generate and log confusion matrix
        print("Generating confusion matrix...")
        probs = model.predict(X_test).flatten()
        y_pred = (probs >= 0.5).astype(int)
        cm_path = plot_confusion_matrix(y_test, y_pred, 'confusion_matrix.png')
        mlflow.log_artifact(cm_path, artifact_path="plots")
        print(f"Confusion matrix saved: {cm_path}")

        # 9. Generate and log training curves
        print("Generating training curves...")
        curves_path = plot_training_history(history, 'training_curves.png')
        mlflow.log_artifact(curves_path, artifact_path="plots")
        print(f"Training curves saved: {curves_path}")

        # 10. Save and log training config
        config = {
            'timestamp': datetime.now().isoformat(),
            'mlflow_run_id': mlflow.active_run().info.run_id,
            'dataset_info': {
                'total_samples': len(X),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'train_after_augmentation': len(X_train_aug),
                'augmentation_multiplier': AUGMENTATION_MULTIPLIER,
                'image_size': [IMAGE_SIZE, IMAGE_SIZE],
                'image_channels': 3,
                'classes': ['cats', 'dogs'],
                'num_classes': 2
            },
            'augmentation': aug_meta,
            'model_info': {
                'model_type': 'Simple CNN',
                'input_shape': list(X_train_aug.shape[1:]),
                'output_shape': 'binary (0=cats, 1=dogs)'
            },
            'training_metrics': metrics,
            'data_split_ratio': {
                'train': 0.8,
                'validation': 0.1,
                'test': 0.1
            }
        }
        config_path = 'training_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        mlflow.log_artifact(config_path, artifact_path="config")
        print(f"Configuration saved to {config_path}")

        run_id = mlflow.active_run().info.run_id
        print("\n" + "=" * 70)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        print(f"MLflow Run ID: {run_id}")
        print("View results: mlflow ui  →  http://localhost:5000")
        print("=" * 70)


if __name__ == '__main__':
    main()
