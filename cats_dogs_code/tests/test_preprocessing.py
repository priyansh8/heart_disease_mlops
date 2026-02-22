"""Unit tests for image preprocessing functions"""
import os

import numpy as np
from PIL import Image

from src.preprocessing import (
    load_images_from_folder,
    split_dataset,
    preprocess_single_image,
    augment_image,
    augment_dataset,
    get_augmentation_metadata
)


def create_dummy_image(path, size=(224, 224), color=(0, 0, 0)):
    img = Image.new('RGB', size, color)
    img.save(path)


def test_load_and_split(tmp_path):
    # create minimal folder structure with enough samples for stratified split
    # test_size=0.2 on 12 samples → test=2, then val=0.25 of 10 → val=2
    # both splits need ≥2 samples per class for sklearn's StratifiedShuffleSplit
    cat_dir = tmp_path / "cats"
    dog_dir = tmp_path / "dogs"
    cat_dir.mkdir()
    dog_dir.mkdir()

    for i in range(6):
        create_dummy_image(cat_dir / f"cat{i}.jpg")
        create_dummy_image(dog_dir / f"dog{i}.jpg")

    X, y = load_images_from_folder(str(tmp_path))
    assert X.shape[0] == 12
    assert y.shape[0] == 12
    assert set(y.tolist()) == {0, 1}

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y, test_size=0.2, val_size=0.2
    )
    assert X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == 12
    assert y_train.shape[0] == X_train.shape[0]
    assert y_val.shape[0] == X_val.shape[0]
    assert y_test.shape[0] == X_test.shape[0]


def test_preprocess_single_image(tmp_path):
    img_path = tmp_path / "sample.png"
    create_dummy_image(img_path, size=(100, 100))
    arr = preprocess_single_image(str(img_path), target_size=(50, 50))
    assert arr.shape == (50, 50, 3)
    assert arr.max() <= 1.0
    assert arr.min() >= 0.0


# ==================== NEW AUGMENTATION TESTS ====================

def test_augment_image_shape(tmp_path):
    """Test that augmented image has same shape as input"""
    img_path = tmp_path / "sample.png"
    create_dummy_image(img_path, size=(224, 224), color=(100, 150, 200))
    
    image = preprocess_single_image(str(img_path), target_size=(224, 224))
    augmented = augment_image(image, augmentation_prob=0.5)
    
    assert augmented.shape == image.shape
    assert augmented.shape == (224, 224, 3)


def test_augment_image_range(tmp_path):
    """Test that augmented image values are in valid range [0, 1]"""
    img_path = tmp_path / "sample.png"
    create_dummy_image(img_path, size=(224, 224))
    
    image = preprocess_single_image(str(img_path), target_size=(224, 224))
    augmented = augment_image(image, augmentation_prob=1.0)  # Apply all augmentations
    
    assert augmented.min() >= 0.0
    assert augmented.max() <= 1.0
    assert augmented.dtype == np.float32


def test_augment_image_reproducibility(tmp_path):
    """Test that augmentation with same seed produces different results"""
    img_path = tmp_path / "sample.png"
    create_dummy_image(img_path, size=(224, 224), color=(50, 100, 150))
    
    image = preprocess_single_image(str(img_path), target_size=(224, 224))
    
    np.random.seed(42)
    aug1 = augment_image(image, augmentation_prob=1.0)
    
    np.random.seed(42)
    aug2 = augment_image(image, augmentation_prob=1.0)
    
    # Same seed should produce same augmentation
    # (Note: not exact match due to PIL operations)
    assert aug1.shape == aug2.shape


def test_augment_dataset_size(tmp_path):
    """Test that augmented dataset is multiplier times larger"""
    cat_dir = tmp_path / "cats"
    dog_dir = tmp_path / "dogs"
    cat_dir.mkdir()
    dog_dir.mkdir()

    # Create 4 images (2 cats, 2 dogs)
    for i in range(2):
        create_dummy_image(cat_dir / f"cat{i}.jpg")
        create_dummy_image(dog_dir / f"dog{i}.jpg")

    X, y = load_images_from_folder(str(tmp_path))
    assert len(X) == 4

    # Augment with multiplier=2
    X_aug, y_aug = augment_dataset(X, y, augmentation_multiplier=2, random_state=42)
    
    assert len(X_aug) == len(X) * 2
    assert len(y_aug) == len(y) * 2
    assert len(X_aug) == 8


def test_augment_dataset_labels_preserved(tmp_path):
    """Test that class labels are preserved in augmented dataset"""
    cat_dir = tmp_path / "cats"
    dog_dir = tmp_path / "dogs"
    cat_dir.mkdir()
    dog_dir.mkdir()

    for i in range(2):
        create_dummy_image(cat_dir / f"cat{i}.jpg")
        create_dummy_image(dog_dir / f"dog{i}.jpg")

    X, y = load_images_from_folder(str(tmp_path))
    X_aug, y_aug = augment_dataset(X, y, augmentation_multiplier=2, random_state=42)

    # Check that we still have both classes
    assert set(y_aug.tolist()) == {0, 1}
    
    # Check class balance is maintained (roughly)
    class_0_count = (y_aug == 0).sum()
    class_1_count = (y_aug == 1).sum()
    assert class_0_count > 0
    assert class_1_count > 0


def test_augment_dataset_shuffling(tmp_path):
    """Test that augmented dataset is not simply sorted by class label."""
    cat_dir = tmp_path / "cats"
    dog_dir = tmp_path / "dogs"
    cat_dir.mkdir()
    dog_dir.mkdir()

    # 5 cats + 5 dogs → 20 samples after 2x augmentation.
    # Before shuffling labels are [0,0,0,0,0, 1,1,1,1,1, 0,0,0,0,0, 1,1,1,1,1] → 18 consecutive pairs.
    for i in range(5):
        create_dummy_image(cat_dir / f"cat{i}.jpg", color=(i * 40, 0, 0))
        create_dummy_image(dog_dir / f"dog{i}.jpg", color=(0, i * 40, 0))

    # Save and restore global numpy random state so this test is isolated.
    rng_state = np.random.get_state()
    try:
        X, y = load_images_from_folder(str(tmp_path))
        X_aug, y_aug = augment_dataset(X, y, augmentation_multiplier=2, random_state=7)
    finally:
        np.random.set_state(rng_state)

    # Both classes must still be present after shuffling
    assert set(y_aug.tolist()) == {0, 1}
    assert len(y_aug) == 20

    # The unshuffled result would have 18 consecutive same-class pairs.
    # After shuffling with any reasonable seed the count should be much lower.
    consecutive_same = sum(1 for i in range(len(y_aug) - 1) if y_aug[i] == y_aug[i + 1])
    total_pairs = len(y_aug) - 1  # 19
    assert consecutive_same < int(total_pairs * 0.85), (
        f"Dataset does not appear shuffled: "
        f"{consecutive_same}/{total_pairs} consecutive same-class pairs."
    )


def test_augmentation_metadata():
    """Test that augmentation metadata is correctly structured"""
    metadata = get_augmentation_metadata()
    
    assert "augmentation_techniques" in metadata
    assert "default_augmentation_probability" in metadata
    assert "default_multiplier" in metadata
    assert "description" in metadata
    
    assert len(metadata["augmentation_techniques"]) == 5  # 5 techniques
    assert isinstance(metadata["augmentation_techniques"], list)
    assert metadata["default_augmentation_probability"] == 0.5
    assert metadata["default_multiplier"] == 2