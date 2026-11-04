"""Task 2.1: tf.data Input Pipeline for VisionGuard (23L0713_A2_Ops).

Reads split CSV lists (train.csv, val.csv, test.csv), decodes images to 3-channel RGB,
resizes to target (IMG_SIZE, IMG_SIZE), applies random data augmentation on the training
split only, batches, and prefetches with tf.data.AUTOTUNE.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
SEED = 42


def _pil_decode_rgb(path_tensor: tf.Tensor) -> np.ndarray:
    """Decode any image file (JPEG, PNG, WEBP, CMYK, RGBA, Palette) into uint8 RGB array."""
    path_str = path_tensor.numpy().decode("utf-8")
    with Image.open(path_str) as img:
        rgb = img.convert("RGB")
        return np.asarray(rgb, dtype=np.uint8)


def decode_and_resize(file_path: tf.Tensor, label: tf.Tensor, img_size: tuple[int, int] = IMG_SIZE):
    """Decode image bytes into RGB float32 [0, 255] and resize to img_size."""
    img = tf.py_function(func=_pil_decode_rgb, inp=[file_path], Tout=tf.uint8)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, img_size, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.ensure_shape(img, [img_size[0], img_size[1], 3])
    label = tf.cast(label, tf.float32)
    return img, label


def build_augmentation_layer(seed: int = SEED) -> tf.keras.Sequential:
    """Random augmentation applied exclusively to the training split."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.08, seed=seed),
            tf.keras.layers.RandomZoom(height_factor=(-0.10, 0.10), width_factor=(-0.10, 0.10), seed=seed),
            tf.keras.layers.RandomContrast(0.10, seed=seed),
        ],
        name="visionguard_augmentation",
    )


def create_dataset(
    csv_path: Path | str,
    img_size: tuple[int, int] = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    training: bool = False,
    augment: bool = False,
    seed: int = SEED,
) -> tuple[tf.data.Dataset, pd.DataFrame]:
    """Create a batched, prefetched tf.data.Dataset from a split CSV file."""
    df = pd.read_csv(csv_path)
    paths = df["file_path"].astype(str).tolist()
    labels = df["label"].astype(np.float32).tolist()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, y: decode_and_resize(p, y, img_size=img_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.cache()

    ds = ds.batch(batch_size)
    if training and augment:
        aug_layer = build_augmentation_layer(seed=seed)
        ds = ds.map(
            lambda x, y: (aug_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, df


def save_augmented_batch_figure(
    train_csv: Path = Path("data/splits/train.csv"),
    out_path: Path = Path("reports/figures/task2_1_augmented_batch.png"),
) -> None:
    """Save a 2x4 grid of one augmented training batch for Task 2.1 report evidence."""
    tf.keras.utils.set_random_seed(SEED)
    ds, _ = create_dataset(train_csv, img_size=IMG_SIZE, batch_size=8, training=True, augment=True, seed=SEED)
    images, labels = next(iter(ds))

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.8))
    fig.suptitle(
        f"VisionGuard (23L0713_A2_Ops) — Augmented Training Batch ({IMG_SIZE[0]}x{IMG_SIZE[1]} RGB)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    for idx, ax in enumerate(axes.flat):
        img_np = np.clip(images[idx].numpy(), 0, 255).astype(np.uint8)
        lbl = int(labels[idx].numpy())
        cls_str = "AI-Generated (1)" if lbl == 1 else "Real Art (0)"
        ax.imshow(img_np)
        ax.set_title(f"Sample #{idx + 1}: {cls_str}", fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved augmented training batch visualization to {out_path}")


if __name__ == "__main__":
    save_augmented_batch_figure()
