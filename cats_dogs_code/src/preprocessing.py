"""Image preprocessing utilities for Cats-vs-Dogs dataset.

This module replaces the previous tabular preprocessing code
used for the heart disease example. Functions here load images,
resize them to the standard 224x224 RGB shape, and prepare
train/validation/test splits for model development.

Includes data augmentation techniques for better generalization.
"""

import os
from pathlib import Path
from typing import Tuple, List, Callable, Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from sklearn.model_selection import train_test_split


def load_images_from_folder(
    folder: str,
    target_size: Tuple[int, int] = (224, 224)
) -> Tuple[np.ndarray, np.ndarray]:
    """Load all images from a directory structure where subfolders
    correspond to class labels (e.g. 'cats/' and 'dogs/').

    Args:
        folder: Root directory containing class subfolders.
        target_size: Desired image size (width, height).

    Returns:
        Tuple of (images_array, labels_array). Images will have shape
        (n_samples, height, width, 3) and dtype float32 scaled to [0,1].
    """
    images: List[np.ndarray] = []
    labels: List[int] = []

    folder_path = Path(folder)
    for label_idx, class_dir in enumerate(sorted(folder_path.iterdir())):
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.glob("*.*"):
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(target_size)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                images.append(arr)
                labels.append(label_idx)
            except Exception:
                # skip unreadable files
                continue

    if images:
        X = np.stack(images)
        y = np.array(labels, dtype=np.int32)
    else:
        X = np.empty((0, target_size[1], target_size[0], 3), dtype=np.float32)
        y = np.empty((0,), dtype=np.int32)

    return X, y


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train/validation/test sets.

    Args:
        X: Feature array
        y: Label array
        test_size: Proportion held out for test set.
        val_size: Proportion of *remaining* data used for validation.
        random_state: Seed for reproducibility.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative,
        random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# helper for unit tests or downstream code

def preprocess_single_image(
    img_path: str,
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """Load and preprocess a single image file, returning a scaled array."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


# ==================== DATA AUGMENTATION FUNCTIONS ====================

def augment_image(
    image: np.ndarray,
    augmentation_prob: float = 0.5
) -> np.ndarray:
    """Apply random augmentation to a single image.
    
    Augmentations include:
    - Random horizontal flip
    - Random vertical flip
    - Random rotation (±15 degrees)
    - Random brightness adjustment
    - Random contrast adjustment
    
    Args:
        image: Input image array with shape (H, W, 3) and values in [0, 1]
        augmentation_prob: Probability of applying each augmentation (0-1)
    
    Returns:
        Augmented image array with same shape and value range
    """
    # Convert back to PIL Image for easier augmentation
    img_uint8 = (image * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_uint8, mode='RGB')
    
    # Random horizontal flip
    if np.random.random() < augmentation_prob:
        img_pil = ImageOps.mirror(img_pil)
    
    # Random vertical flip
    if np.random.random() < augmentation_prob:
        img_pil = ImageOps.flip(img_pil)
    
    # Random rotation (±15 degrees)
    if np.random.random() < augmentation_prob:
        angle = np.random.uniform(-15, 15)
        img_pil = img_pil.rotate(angle, expand=False, fillcolor=(128, 128, 128))
    
    # Random brightness adjustment
    if np.random.random() < augmentation_prob:
        brightness_factor = np.random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Brightness(img_pil)
        img_pil = enhancer.enhance(brightness_factor)
    
    # Random contrast adjustment
    if np.random.random() < augmentation_prob:
        contrast_factor = np.random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Contrast(img_pil)
        img_pil = enhancer.enhance(contrast_factor)
    
    # Convert back to numpy array
    augmented_array = np.asarray(img_pil, dtype=np.float32) / 255.0
    return augmented_array


def augment_dataset(
    X: np.ndarray,
    y: np.ndarray,
    augmentation_multiplier: int = 2,
    augmentation_prob: float = 0.5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply data augmentation to training dataset.
    
    Creates augmented copies of images to increase dataset size.
    
    Args:
        X: Training images array with shape (N, H, W, 3)
        y: Training labels array with shape (N,)
        augmentation_multiplier: Number of augmented copies per original image
        augmentation_prob: Probability of applying each augmentation per image
        random_state: Seed for reproducibility
    
    Returns:
        (X_augmented, y_augmented) with original + augmented samples
    """
    np.random.seed(random_state)
    
    X_augmented_list = [X]  # Start with original images
    y_augmented_list = [y]
    
    # Generate augmented copies
    for _ in range(augmentation_multiplier - 1):
        X_aug = np.stack([
            augment_image(img, augmentation_prob)
            for img in X
        ])
        X_augmented_list.append(X_aug)
        y_augmented_list.append(y.copy())
    
    # Concatenate all augmented and original data
    X_augmented = np.concatenate(X_augmented_list, axis=0)
    y_augmented = np.concatenate(y_augmented_list, axis=0)
    
    # Shuffle the combined dataset
    shuffle_idx = np.random.permutation(len(X_augmented))
    X_augmented = X_augmented[shuffle_idx]
    y_augmented = y_augmented[shuffle_idx]
    
    return X_augmented, y_augmented


def get_augmentation_metadata() -> dict:
    """Return metadata about augmentation techniques used."""
    return {
        "augmentation_techniques": [
            "Horizontal flip",
            "Vertical flip",
            "Rotation (±15°)",
            "Brightness adjustment (0.8-1.2x)",
            "Contrast adjustment (0.8-1.2x)"
        ],
        "default_augmentation_probability": 0.5,
        "default_multiplier": 2,
        "description": "Data augmentation for better model generalization"
    }
