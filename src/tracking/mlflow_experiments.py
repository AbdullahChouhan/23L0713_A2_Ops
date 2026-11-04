"""Task 3.1 – 3.6: MLflow & DagsHub Experiment Tracking, Comparison, and Model Registry (23L0713_A2_Ops).

All DagsHub credentials and tracking URIs are read strictly from environment variables (.env).
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import mlflow.tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.models.architectures import build_baseline_cnn, build_transfer_model
from src.models.data_pipeline import BATCH_SIZE, IMG_SIZE, SEED, create_dataset

load_dotenv()


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False)
        return res.stdout.strip() or "initial"
    except Exception:
        return "initial"


def get_dvc_version() -> str:
    dvc_file = Path("data/raw.dvc")
    if dvc_file.exists():
        return hashlib.sha256(dvc_file.read_bytes()).hexdigest()[:12]
    return "untracked"


class ManualMLflowEpochLogger(tf.keras.callbacks.Callback):
    """Task 3.3: Custom Keras Callback that manually logs per-epoch metrics to MLflow."""

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if not logs:
            return
        clean_metrics = {k: float(v) for k, v in logs.items() if isinstance(v, (int, float, np.floating))}
        mlflow.log_metrics(clean_metrics, step=epoch + 1)


def run_toy_mlflow_runs(tracking_uri: str, experiment_name: str, label_prefix: str) -> list[dict]:
    """Task 3.1 & 3.2: Train a tiny Keras model on Fashion-MNIST with two different learning rates."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    tf.keras.utils.set_random_seed(SEED)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_tr = (x_train[:2500].astype("float32") / 255.0)[..., np.newaxis]
    y_tr = y_train[:2500]
    x_te = (x_test[:600].astype("float32") / 255.0)[..., np.newaxis]
    y_te = y_test[:600]

    results = []
    for lr in [1e-3, 1e-2]:
        run_name = f"{label_prefix}_FashionMNIST_lr_{lr}"
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags(
                {
                    "project": "23L0713_A2_Ops",
                    "student": "Abdullah Chouhan (23L0713)",
                    "task": "3.1_3.2_Toy_Lab",
                    "dataset": "Fashion-MNIST",
                }
            )
            mlflow.log_params({"learning_rate": lr, "epochs": 4, "batch_size": 64, "hidden_units": 64, "seed": SEED})

            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=(28, 28, 1)),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(64, activation="relu"),
                    tf.keras.layers.Dense(10, activation="softmax"),
                ]
            )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            hist = model.fit(
                x_tr,
                y_tr,
                validation_data=(x_te, y_te),
                epochs=4,
                batch_size=64,
                callbacks=[ManualMLflowEpochLogger()],
                verbose=0,
            )

            # Plot loss curve and attach as artifact
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(range(1, 5), hist.history["loss"], "o-", label="Train Loss")
            ax.plot(range(1, 5), hist.history["val_loss"], "s-", label="Val Loss")
            ax.set_title(f"Toy Fashion-MNIST Loss (LR={lr})")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            plot_path = Path("reports/figures") / f"toy_mlflow_lr_{lr}.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close(fig)

            mlflow.log_artifact(str(plot_path), artifact_path="plots")
            val_acc = float(hist.history["val_accuracy"][-1])
            val_loss = float(hist.history["val_loss"][-1])
            mlflow.log_metrics({"final_val_accuracy": val_acc, "final_val_loss": val_loss})

            results.append(
                {
                    "run_id": run.info.run_id,
                    "run_name": run_name,
                    "tracking_uri": tracking_uri,
                    "learning_rate": lr,
                    "final_val_accuracy": round(val_acc, 4),
                    "final_val_loss": round(val_loss, 4),
                }
            )
            print(f"[Toy MLflow] Logged {run_name} ({run.info.run_id}) -> val_acc={val_acc:.4f}")
    return results


def evaluate_on_dataset(model: tf.keras.Model, ds: tf.data.Dataset, threshold: float = 0.50) -> dict:
    y_true, y_prob = [], []
    for bx, by in ds:
        p = model.predict(bx, verbose=0).reshape(-1)
        y_true.append(by.numpy().reshape(-1))
        y_prob.append(p)
    yt = np.concatenate(y_true)
    yp = np.concatenate(y_prob)
    pred = (yp >= threshold).astype(int)
    return {
        "test_accuracy": round(float(accuracy_score(yt, pred)), 4),
        "test_precision": round(float(precision_score(yt, pred, zero_division=0)), 4),
        "test_recall": round(float(recall_score(yt, pred, zero_division=0)), 4),
        "test_f1": round(float(f1_score(yt, pred, zero_division=0)), 4),
        "test_roc_auc": round(float(roc_auc_score(yt, yp)), 4),
    }


def run_visionguard_experiments_and_registry() -> pd.DataFrame:
    """Tasks 3.3, 3.4, 3.5 & 3.6 on DagsHub MLflow (and local fallback check)."""
    dagshub_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if not dagshub_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is not set in environment/.env!")

    # Verify .env is git-ignored (Task 3.2 evidence)
    ignore_res = subprocess.run(["git", "check-ignore", "-v", ".env"], capture_output=True, text=True, check=False)
    Path("reports/figures/task3_2_git_ignore_proof.txt").write_text(
        f"$ git check-ignore -v .env\n{ignore_res.stdout.strip()}\n"
        f"$ git status --short\n{subprocess.run(['git', 'status', '--short'], capture_output=True, text=True).stdout.strip()}\n",
        encoding="utf-8",
    )

    # 1. Task 3.1 Local Toy Runs & Task 3.2 DagsHub Toy Runs
    local_uri = "sqlite:///mlflow_local.db"
    toy_local = run_toy_mlflow_runs(local_uri, "23L0713_A2_Ops_Toy_Local", "LocalToy")
    toy_dagshub = run_toy_mlflow_runs(dagshub_uri, "23L0713_A2_Ops_Toy_DagsHub", "DagsHubToy")
    pd.DataFrame(toy_local + toy_dagshub).to_csv("reports/figures/task3_1_3_2_toy_runs.csv", index=False)

    # 2. Switch to DagsHub Experiment for VisionGuard (Task 3.3, 3.4, 3.5)
    mlflow.set_tracking_uri(dagshub_uri)
    exp_name = "23L0713_A2_Ops_VisionGuard"
    mlflow.set_experiment(exp_name)

    git_sha = get_git_commit()
    dvc_ver = get_dvc_version()

    # Define 6+ meaningful experiment variations for Task 3.5 (including manual vs autolog in Task 3.3)
    experiment_specs = [
        {
            "run_name": "Run1_Baseline_CNN_160px_lr1e-3_ManualLog",
            "arch": "Baseline_CNN",
            "backbone": "Scratch_CNN",
            "img_size": 160,
            "lr": 1e-3,
            "dropout": 0.35,
            "augment": True,
            "unfreeze_n": 0,
            "epochs": 6,
            "logging_mode": "manual",
            "change_description": "Baseline custom CNN trained from scratch (160x160, LR=1e-3, Augmented)",
        },
        {
            "run_name": "Run2_MobileNetV2_Frozen_160px_lr1e-3_Autolog",
            "arch": "Transfer",
            "backbone": "MobileNetV2",
            "img_size": 160,
            "lr": 1e-3,
            "dropout": 0.35,
            "augment": True,
            "unfreeze_n": 0,
            "epochs": 5,
            "logging_mode": "autolog",
            "change_description": "Changed backbone: Scratch CNN -> Pretrained MobileNetV2 (frozen, Autologged)",
        },
        {
            "run_name": "Run3_MobileNetV2_Frozen_NoAug_160px_lr1e-3",
            "arch": "Transfer",
            "backbone": "MobileNetV2",
            "img_size": 160,
            "lr": 1e-3,
            "dropout": 0.35,
            "augment": False,
            "unfreeze_n": 0,
            "epochs": 5,
            "logging_mode": "manual",
            "change_description": "Changed augmentation: Disabled random training augmentation (augment=False)",
        },
        {
            "run_name": "Run4_MobileNetV2_Unfreeze25_160px_lr1e-4_Champion",
            "arch": "Transfer",
            "backbone": "MobileNetV2",
            "img_size": 160,
            "lr": 1e-4,
            "dropout": 0.35,
            "augment": True,
            "unfreeze_n": 25,
            "epochs": 6,
            "logging_mode": "manual",
            "change_description": "Changed frozen layers & LR: Unfroze top 25 layers with fine-tuning LR=1e-4",
        },
        {
            "run_name": "Run5_MobileNetV2_Unfreeze25_160px_lr1e-3_HighLR",
            "arch": "Transfer",
            "backbone": "MobileNetV2",
            "img_size": 160,
            "lr": 1e-3,
            "dropout": 0.35,
            "augment": True,
            "unfreeze_n": 25,
            "epochs": 5,
            "logging_mode": "manual",
            "change_description": "Changed learning rate: High LR (1e-3) with unfrozen layers degrades pretrained weights",
        },
        {
            "run_name": "Run6_MobileNetV2_Unfreeze25_128px_lr1e-4_SmallRes",
            "arch": "Transfer",
            "backbone": "MobileNetV2",
            "img_size": 128,
            "lr": 1e-4,
            "dropout": 0.35,
            "augment": True,
            "unfreeze_n": 25,
            "epochs": 5,
            "logging_mode": "manual",
            "change_description": "Changed image size: Reduced input resolution from 160x160 -> 128x128",
        },
    ]

    comparison_rows = []
    for spec in experiment_specs:
        tf.keras.utils.set_random_seed(SEED)
        sz = (spec["img_size"], spec["img_size"])
        tr_ds, _ = create_dataset("data/splits/train.csv", img_size=sz, batch_size=BATCH_SIZE, training=True, augment=spec["augment"])
        va_ds, _ = create_dataset("data/splits/val.csv", img_size=sz, batch_size=BATCH_SIZE, training=False, augment=False)
        te_ds, _ = create_dataset("data/splits/test.csv", img_size=sz, batch_size=BATCH_SIZE, training=False, augment=False)

        if spec["logging_mode"] == "autolog":
            mlflow.tensorflow.autolog(log_models=False, disable=False, silent=True)
        else:
            mlflow.tensorflow.autolog(disable=True)

        with mlflow.start_run(run_name=spec["run_name"]) as run:
            mlflow.set_tags(
                {
                    "project": "23L0713_A2_Ops",
                    "student": "Abdullah Chouhan",
                    "roll_number": "23L0713",
                    "git_commit": git_sha,
                    "dvc_data_version": dvc_ver,
                    "seed": str(SEED),
                    "logging_mode": spec["logging_mode"],
                    "meaningful_change": spec["change_description"],
                }
            )
            mlflow.log_params(
                {
                    "architecture": spec["arch"],
                    "backbone": spec["backbone"],
                    "img_size": spec["img_size"],
                    "learning_rate": spec["lr"],
                    "dropout_rate": spec["dropout"],
                    "augment": spec["augment"],
                    "unfrozen_layers": spec["unfreeze_n"],
                    "epochs": spec["epochs"],
                    "batch_size": BATCH_SIZE,
                    "seed": SEED,
                }
            )

            if spec["arch"] == "Baseline_CNN":
                model = build_baseline_cnn(img_size=sz, learning_rate=spec["lr"], dropout_rate=spec["dropout"])
            else:
                model, _ = build_transfer_model(
                    backbone_name=spec["backbone"],
                    img_size=sz,
                    learning_rate=spec["lr"],
                    dropout_rate=spec["dropout"],
                    unfreeze_last_n=spec["unfreeze_n"],
                )

            cbs = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
            if spec["logging_mode"] == "manual":
                cbs.append(ManualMLflowEpochLogger())

            hist = model.fit(tr_ds, validation_data=va_ds, epochs=spec["epochs"], callbacks=cbs, verbose=1)
            test_metrics = evaluate_on_dataset(model, te_ds, threshold=0.50)
            best_val_loss = round(float(min(hist.history["val_loss"])), 4)
            best_val_acc = round(float(max(hist.history["val_accuracy"])), 4)
            best_val_auc = round(float(max(hist.history["val_auc"])), 4)

            mlflow.log_metrics(
                {
                    "best_val_loss": best_val_loss,
                    "best_val_accuracy": best_val_acc,
                    "best_val_auc": best_val_auc,
                    **test_metrics,
                }
            )

            # Task 3.4: Attach required artifacts to the champion run (and key runs)
            if "Champion" in spec["run_name"] or "ManualLog" in spec["run_name"]:
                for art in [
                    Path("reports/figures/task2_5_confusion_matrices.png"),
                    Path("reports/figures/task2_5_roc_curves.png"),
                    Path("data/splits/train.csv"),
                    Path("data/splits/val.csv"),
                    Path("data/splits/test.csv"),
                    Path("requirements.txt"),
                ]:
                    if art.exists():
                        mlflow.log_artifact(str(art), artifact_path="project_artifacts")

            comparison_rows.append(
                {
                    "Run Name": spec["run_name"],
                    "Run ID": run.info.run_id,
                    "Backbone": spec["backbone"],
                    "Img Size": f"{spec['img_size']}x{spec['img_size']}",
                    "LR": spec["lr"],
                    "Augment": spec["augment"],
                    "Unfrozen": spec["unfreeze_n"],
                    "Logging": spec["logging_mode"],
                    "Val Loss": best_val_loss,
                    "Val AUC": best_val_auc,
                    "Test Acc": test_metrics["test_accuracy"],
                    "Test F1": test_metrics["test_f1"],
                    "Test ROC-AUC": test_metrics["test_roc_auc"],
                    "Change Tested": spec["change_description"],
                }
            )
            print(f"[MLflow Run] {spec['run_name']} -> Test F1={test_metrics['test_f1']}, AUC={test_metrics['test_roc_auc']}")

    mlflow.tensorflow.autolog(disable=True)
    comp_df = pd.DataFrame(comparison_rows)
    out_dir = Path("reports/figures")
    comp_df.to_csv(out_dir / "task3_5_experiment_comparison.csv", index=False)

    # Plot comparison bar chart for Task 3.5
    fig, ax = plt.subplots(figsize=(11, 5.2))
    short_names = [r["Run Name"].split("_")[0] + ": " + r["Backbone"] + f" (UF={r['Unfrozen']}, LR={r['LR']})" for r in comparison_rows]
    x_pos = np.arange(len(short_names))
    ax.bar(x_pos - 0.18, comp_df["Test F1"], width=0.35, label="Test F1-Score", color="#1b9e77")
    ax.bar(x_pos + 0.18, comp_df["Test ROC-AUC"], width=0.35, label="Test ROC-AUC", color="#7570b3")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_names, rotation=22, ha="right", fontsize=9)
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("VisionGuard (23L0713_A2_Ops) — Task 3.5 Comparison Across 6 Meaningful Experiment Runs", fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "task3_5_comparison_chart.png", dpi=200)
    plt.close(fig)

    # Task 3.6: Model Registry — Register 2 versions and assign 'champion' alias
    register_champion_models(comp_df, dagshub_uri, local_uri)
    return comp_df


def register_champion_models(comp_df: pd.DataFrame, dagshub_uri: str, local_uri: str) -> None:
    """Task 3.6: Register v1 (Baseline/Frozen) and v2 (Fine-Tuned Champion) in MLflow Model Registry."""
    model_name = "VisionGuard_23L0713_Classifier"
    champion_keras_path = Path("models/champion_model.keras")
    baseline_keras_path = Path("models/baseline_cnn.keras")

    for target_name, uri in [("DagsHub", dagshub_uri), ("Local", local_uri)]:
        try:
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment("23L0713_A2_Ops_Registry")
            client = MlflowClient(tracking_uri=uri)

            v1_model = tf.keras.models.load_model(baseline_keras_path)
            with mlflow.start_run(run_name="Registry_v1_Baseline_CNN") as r1:
                mlflow.keras.log_model(v1_model, artifact_path="model", registered_model_name=model_name)
                if champion_keras_path.exists():
                    mlflow.log_artifact(str(champion_keras_path), artifact_path="keras_file")

            v2_model = tf.keras.models.load_model(champion_keras_path)
            with mlflow.start_run(run_name="Registry_v2_MobileNetV2_Champion") as r2:
                res = mlflow.keras.log_model(v2_model, artifact_path="model", registered_model_name=model_name)
                mlflow.log_artifact(str(champion_keras_path), artifact_path="keras_file")

            versions = client.search_model_versions(f"name='{model_name}'")
            latest_ver = max(int(v.version) for v in versions) if versions else 2
            try:
                client.set_registered_model_alias(model_name, "champion", str(latest_ver))
                alias_status = f"Assigned alias '@champion' -> version {latest_ver}"
            except Exception:
                # Fallback if tracking server uses stage/tags for aliases
                client.set_model_version_tag(model_name, str(latest_ver), "alias", "champion")
                alias_status = f"Tagged version {latest_ver} with alias='champion'"

            print(f"[Task 3.6 Registry - {target_name}] Registered {model_name} (latest v{latest_ver}): {alias_status}")
        except Exception as exc:
            print(f"[WARN] Registry step on {target_name} ({uri}) encountered: {exc}")


if __name__ == "__main__":
    run_visionguard_experiments_and_registry()
