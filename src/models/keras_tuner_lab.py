"""Task 2.7: Keras Tuner Tool Lab — Toy Part (Fashion-MNIST) & VisionGuard Tuning (8 Trials)."""

import json
import shutil
from pathlib import Path

import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf

from src.models.architectures import build_transfer_model
from src.models.data_pipeline import BATCH_SIZE, IMG_SIZE, SEED, create_dataset


def run_toy_fashion_mnist_tuner() -> pd.DataFrame:
    """Toy part: Tune a tiny network on Fashion-MNIST searching learning_rate and units."""
    print("\n[Task 2.7 Toy] Loading Fashion-MNIST subset & running Keras Tuner RandomSearch ...")
    tf.keras.utils.set_random_seed(SEED)
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.fashion_mnist.load_data()
    # Use a fast subset (4,000 train / 1,000 val) normalized to [0, 1]
    x_tr = (x_train[:4000].astype("float32") / 255.0)[..., np.newaxis]
    y_tr = y_train[:4000]
    x_v = (x_val[:1000].astype("float32") / 255.0)[..., np.newaxis]
    y_v = y_val[:1000]

    def build_toy_model(hp: kt.HyperParameters) -> tf.keras.Model:
        units = hp.Int("units", min_value=32, max_value=128, step=32)
        lr = hp.Choice("learning_rate", values=[1e-2, 3e-3, 1e-3, 3e-4])
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(28, 28, 1)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    tuner_dir = Path("kt_toy_dir")
    if tuner_dir.exists():
        shutil.rmtree(tuner_dir)

    tuner = kt.RandomSearch(
        build_toy_model,
        objective="val_accuracy",
        max_trials=6,
        seed=SEED,
        directory=str(tuner_dir),
        project_name="fashion_mnist_toy",
    )
    tuner.search(x_tr, y_tr, validation_data=(x_v, y_v), epochs=4, batch_size=64, verbose=0)

    rows = []
    for trial in tuner.oracle.get_best_trials(num_trials=6):
        rows.append(
            {
                "Trial ID": trial.trial_id,
                "Units": trial.hyperparameters.get("units"),
                "Learning Rate": trial.hyperparameters.get("learning_rate"),
                "Best Val Accuracy": round(float(trial.score), 4),
            }
        )
    df = pd.DataFrame(rows)
    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "task2_7_toy_tuner_results.csv", index=False)
    print("Fashion-MNIST Toy Tuner Results:")
    print(df.to_string(index=False))
    return df


def run_visionguard_tuner(max_trials: int = 8, epochs_per_trial: int = 3) -> pd.DataFrame:
    """Apply part: Tune learning_rate, dropout, and num_unfrozen_layers for VisionGuard (>= 8 trials)."""
    print(f"\n[Task 2.7 VisionGuard] Running Keras Tuner RandomSearch ({max_trials} trials) ...")
    tf.keras.utils.set_random_seed(SEED)
    train_ds, _ = create_dataset("data/splits/train.csv", img_size=IMG_SIZE, batch_size=BATCH_SIZE, training=True, augment=True)
    val_ds, _ = create_dataset("data/splits/val.csv", img_size=IMG_SIZE, batch_size=BATCH_SIZE, training=False, augment=False)

    def build_vg_hypermodel(hp: kt.HyperParameters) -> tf.keras.Model:
        lr = hp.Choice("learning_rate", values=[1e-3, 5e-4, 1e-4, 5e-5])
        dropout = hp.Choice("dropout_rate", values=[0.20, 0.35, 0.50])
        unfreeze_n = hp.Choice("num_unfrozen_layers", values=[0, 15, 30])
        model, _ = build_transfer_model(
            backbone_name="MobileNetV2",
            img_size=IMG_SIZE,
            learning_rate=lr,
            dropout_rate=dropout,
            unfreeze_last_n=unfreeze_n,
        )
        return model

    tuner_dir = Path("kt_dir")
    if tuner_dir.exists():
        shutil.rmtree(tuner_dir)

    tuner = kt.RandomSearch(
        build_vg_hypermodel,
        objective=kt.Objective("val_auc", direction="max"),
        max_trials=max_trials,
        seed=SEED,
        directory=str(tuner_dir),
        project_name="visionguard_23L0713",
    )

    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_per_trial,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)],
        verbose=1,
    )

    rows = []
    for idx, trial in enumerate(tuner.oracle.get_best_trials(num_trials=max_trials), start=1):
        val_loss_metric = trial.metrics.get_best_value("val_loss")
        val_acc_metric = trial.metrics.get_best_value("val_accuracy")
        rows.append(
            {
                "Rank": idx,
                "Trial ID": trial.trial_id,
                "Learning Rate": trial.hyperparameters.get("learning_rate"),
                "Dropout Rate": trial.hyperparameters.get("dropout_rate"),
                "Unfrozen Layers": trial.hyperparameters.get("num_unfrozen_layers"),
                "Best Val AUC": round(float(trial.score), 4),
                "Best Val Accuracy": round(float(val_acc_metric), 4) if val_acc_metric is not None else None,
                "Best Val Loss": round(float(val_loss_metric), 4) if val_loss_metric is not None else None,
            }
        )

    df = pd.DataFrame(rows)
    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "task2_7_vg_tuner_results.csv", index=False)

    # Compute hyperparameter importance (variance of mean Val AUC across levels of each hyperparameter)
    importance = {}
    for col in ["Learning Rate", "Dropout Rate", "Unfrozen Layers"]:
        group_means = df.groupby(col)["Best Val AUC"].mean()
        importance[col] = round(float(group_means.max() - group_means.min()), 4)

    summary = {
        "best_trial": rows[0] if rows else {},
        "effect_range_on_val_auc": importance,
        "most_influential_setting": max(importance, key=importance.get) if importance else "Learning Rate",
    }
    (out_dir / "task2_7_tuner_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("VisionGuard Keras Tuner Results (8 Trials):")
    print(df.to_string(index=False))
    print("Hyperparameter Effect Range on Val AUC:", importance)
    return df


if __name__ == "__main__":
    run_toy_fashion_mnist_tuner()
    run_visionguard_tuner()
