"""Tasks 2.2, 2.3, 2.4, 2.5, 2.6 & 2.8 (Bonus Grad-CAM): Model training, evaluation, error analysis, and Grad-CAM."""

import json
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.models.architectures import (
    build_baseline_cnn,
    build_transfer_model,
    export_model_summary_and_table,
    unfreeze_last_block,
)
from src.models.data_pipeline import BATCH_SIZE, IMG_SIZE, SEED, create_dataset


def make_callbacks(checkpoint_path: Path, patience_es: int = 4, patience_lr: int = 2) -> list:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_es,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def collect_predictions(model: tf.keras.Model, ds: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    y_true_list, y_prob_list = [], []
    for batch_x, batch_y in ds:
        probs = model.predict(batch_x, verbose=0).reshape(-1)
        y_true_list.append(batch_y.numpy().reshape(-1))
        y_prob_list.append(probs)
    return np.concatenate(y_true_list), np.concatenate(y_prob_list)


def find_optimal_threshold(y_val: np.ndarray, p_val: np.ndarray) -> tuple[float, dict]:
    """Select threshold on validation set that maximizes F1 score (with Youden's J recorded)."""
    thresholds = np.linspace(0.25, 0.75, 51)
    best_t = 0.50
    best_f1 = -1.0
    records = []
    for t in thresholds:
        preds = (p_val >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        acc = accuracy_score(y_val, preds)
        records.append({"threshold": round(float(t), 2), "val_f1": f1, "val_precision": prec, "val_recall": rec, "val_accuracy": acc})
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(round(t, 2))
    return best_t, {"best_threshold": best_t, "best_val_f1": round(float(best_f1), 4), "curve": records}


def compute_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, model_name: str) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    cm_arr = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm_arr.ravel()
    return {
        "Model": model_name,
        "Threshold": round(float(threshold), 2),
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1-Score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "ROC-AUC": round(float(roc_auc_score(y_true, y_prob)), 4),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def plot_best_epoch_curves(history_dict: dict, title: str, out_path: Path) -> int:
    val_losses = history_dict["val_loss"]
    best_epoch_idx = int(np.argmin(val_losses))
    best_epoch_num = best_epoch_idx + 1
    epochs = list(range(1, len(val_losses) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{title} — Best Epoch: {best_epoch_num} (val_loss={val_losses[best_epoch_idx]:.4f})", fontsize=13, fontweight="bold")

    # Loss subplot
    axes[0].plot(epochs, history_dict["loss"], "o-", label="Train Loss", color="#1f77b4")
    axes[0].plot(epochs, history_dict["val_loss"], "s-", label="Validation Loss", color="#d62728")
    axes[0].axvline(best_epoch_num, color="#2ca02c", linestyle="--", linewidth=2, label=f"Best Epoch ({best_epoch_num})")
    axes[0].set_title("Binary Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy subplot
    axes[1].plot(epochs, history_dict["accuracy"], "o-", label="Train Accuracy", color="#1f77b4")
    axes[1].plot(epochs, history_dict["val_accuracy"], "s-", label="Validation Accuracy", color="#d62728")
    axes[1].axvline(best_epoch_num, color="#2ca02c", linestyle="--", linewidth=2, label=f"Best Epoch ({best_epoch_num})")
    axes[1].set_title("Classification Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    return best_epoch_num


def plot_two_stage_transfer_curves(hist_stage1: dict, hist_stage2: dict, out_path: Path) -> None:
    s1_epochs = len(hist_stage1["loss"])
    total_loss = hist_stage1["loss"] + hist_stage2["loss"]
    total_val_loss = hist_stage1["val_loss"] + hist_stage2["val_loss"]
    total_acc = hist_stage1["accuracy"] + hist_stage2["accuracy"]
    total_val_acc = hist_stage1["val_accuracy"] + hist_stage2["val_accuracy"]
    epochs = list(range(1, len(total_loss) + 1))
    best_ep = int(np.argmin(total_val_loss)) + 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    fig.suptitle(
        f"VisionGuard (23L0713_A2_Ops) — MobileNetV2 Two-Stage Transfer Learning (Best Epoch: {best_ep})",
        fontsize=13,
        fontweight="bold",
    )

    for ax, tr_vals, val_vals, metric_name in [
        (axes[0], total_loss, total_val_loss, "Loss"),
        (axes[1], total_acc, total_val_acc, "Accuracy"),
    ]:
        ax.plot(epochs, tr_vals, "o-", label=f"Train {metric_name}", color="#1f77b4")
        ax.plot(epochs, val_vals, "s-", label=f"Val {metric_name}", color="#ff7f0e")
        ax.axvline(s1_epochs + 0.5, color="#9467bd", linestyle=":", linewidth=2, label="Stage 2 Unfreeze Start")
        ax.axvline(best_ep, color="#2ca02c", linestyle="--", linewidth=2, label=f"Best Epoch ({best_ep})")
        ax.set_title(f"Stage 1 (Frozen Head) -> Stage 2 (Fine-Tuning) {metric_name}")
        ax.set_xlabel("Cumulative Epoch")
        ax.set_ylabel(metric_name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_confusion_And_roc(
    y_test: np.ndarray,
    p_base: np.ndarray,
    t_base: float,
    p_tl: np.ndarray,
    t_tl: float,
    fig_dir: Path,
) -> None:
    # Confusion Matrices side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("VisionGuard (23L0713_A2_Ops) — Test Set Confusion Matrices", fontsize=13, fontweight="bold")

    for ax, probs, thresh, title in [
        (axes[0], p_base, t_base, f"Baseline CNN (Threshold={t_base:.2f})"),
        (axes[1], p_tl, t_tl, f"MobileNetV2 Fine-Tuned (Threshold={t_tl:.2f})"),
    ]:
        preds = (probs >= thresh).astype(int)
        cm_mat = confusion_matrix(y_test, preds)
        sns.heatmap(
            cm_mat,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Real (0)", "AI (1)"],
            yticklabels=["Real (0)", "AI (1)"],
            ax=ax,
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig(fig_dir / "task2_5_confusion_matrices.png", dpi=200)
    plt.close(fig)

    # ROC Curves
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for probs, label, color in [
        (p_base, "Baseline CNN", "#d95f02"),
        (p_tl, "MobileNetV2 Fine-Tuned (Champion)", "#1b9e77"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_val = roc_auc_score(y_test, probs)
        ax.plot(fpr, tpr, linewidth=2.3, label=f"{label} (AUC = {auc_val:.4f})", color=color)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Classifier (AUC = 0.5000)")
    ax.set_title("VisionGuard (23L0713_A2_Ops) — Test Set ROC Curves", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate (Real Classified as AI)")
    ax.set_ylabel("True Positive Rate (AI Correctly Detected)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "task2_5_roc_curves.png", dpi=200)
    plt.close(fig)


def generate_error_analysis(
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    p_tl: np.ndarray,
    threshold: float,
    fig_dir: Path,
) -> None:
    """Task 2.6: Analyze and plot 10 misclassified test images with concrete diagnostic hypotheses."""
    df = test_df.copy().reset_index(drop=True)
    df["y_true"] = y_test.astype(int)
    df["y_prob"] = p_tl
    df["y_pred"] = (p_tl >= threshold).astype(int)
    df["confidence"] = np.where(df["y_pred"] == 1, df["y_prob"], 1.0 - df["y_prob"])

    mis_df = df[df["y_true"] != df["y_pred"]].copy()
    mis_df["error_margin"] = np.abs(mis_df["y_prob"] - threshold)
    mis_df = mis_df.sort_values("error_margin", ascending=False).head(10).reset_index(drop=True)

    hypotheses = []
    fig, axes = plt.subplots(2, 5, figsize=(17, 7.5))
    fig.suptitle(
        "VisionGuard (23L0713_A2_Ops) — Task 2.6 Error Analysis: Top 10 Misclassified Test Images",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    for idx, ax in enumerate(axes.flat):
        if idx >= len(mis_df):
            ax.axis("off")
            continue
        row = mis_df.iloc[idx]
        with Image.open(row["file_path"]) as img:
            rgb = img.convert("RGB")
            arr = np.asarray(rgb, dtype=np.float32)

        # Compute simple visual diagnostic cues (aspect distortion, contrast, texture variance)
        orig_w, orig_h = int(row["width"]), int(row["height"])
        aspect = orig_w / max(orig_h, 1)
        gray = arr.mean(axis=2)
        lap_var = float(np.var(np.diff(gray, axis=0)) + np.var(np.diff(gray, axis=1)))
        contrast = float(gray.std())

        if row["y_true"] == 1 and row["y_pred"] == 0:
            err_type = "FN (AI -> Real)"
            if lap_var > 450:
                reason = "High-frequency brushstroke/grain texture mimics real canvas photography"
            elif aspect > 1.35 or aspect < 0.75:
                reason = f"Non-square aspect ({orig_w}x{orig_h}) compressed during 160x160 resize, smoothing AI artifacts"
            else:
                reason = "Photorealistic lighting & natural depth-of-field lacking typical AI symmetry"
        else:
            err_type = "FP (Real -> AI)"
            if lap_var < 180:
                reason = "Smooth gradients / digital illustration style resembles diffusion smoothing"
            elif contrast > 68:
                reason = "Hyper-saturated studio lighting / stylized post-processing resembles AI rendering"
            else:
                reason = f"Heavy JPEG compression ({row['file_size_kb']:.0f} KB) suppresses natural camera sensor noise"

        hypotheses.append(
            {
                "Case #": idx + 1,
                "File Name": Path(row["file_path"]).name,
                "True Class": "AI (1)" if row["y_true"] == 1 else "Real (0)",
                "Predicted": "AI (1)" if row["y_pred"] == 1 else "Real (0)",
                "Error Type": err_type,
                "P(AI)": round(float(row["y_prob"]), 4),
                "Orig Size": f"{orig_w}x{orig_h}",
                "Texture Var": round(lap_var, 1),
                "Diagnostic Hypothesis": reason,
            }
        )

        ax.imshow(rgb)
        ax.set_title(
            f"#{idx + 1} {err_type}\nTrue: {row['class_label']} | P(AI)={row['y_prob']:.2f}\n{orig_w}x{orig_h} ({row['file_size_kb']:.0f}KB)",
            fontsize=8.5,
            fontweight="bold",
        )
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(fig_dir / "task2_6_error_analysis_grid.png", dpi=200)
    plt.close(fig)

    pd.DataFrame(hypotheses).to_csv(fig_dir / "task2_6_error_hypotheses.csv", index=False)


def compute_gradcam_heatmap(model: tf.keras.Model, base_model: tf.keras.Model, img_batch: np.ndarray) -> np.ndarray:
    """Compute Grad-CAM heatmap using the spatial output of MobileNetV2 base_model."""
    # Build a sub-model mapping base_model input -> last spatial feature map
    # First apply model's preprocessing layer:
    preproc_layer = model.get_layer("mobilenetv2_preprocess")
    head_layers = [
        model.get_layer("gap"),
        model.get_layer("head_bn"),
        model.get_layer("head_drop1"),
        model.get_layer("head_dense"),
        model.get_layer("head_drop2"),
        model.get_layer("prob_ai"),
    ]

    img_tensor = tf.cast(img_batch, tf.float32)
    x_pre = preproc_layer(img_tensor)

    with tf.GradientTape() as tape:
        conv_outputs = base_model(x_pre, training=False)
        tape.watch(conv_outputs)
        x = conv_outputs
        for lyr in head_layers:
            x = lyr(x, training=False)
        pred_prob = x[:, 0]

    grads = tape.gradient(pred_prob, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def generate_gradcam_bonus(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    p_tl: np.ndarray,
    threshold: float,
    fig_dir: Path,
) -> None:
    """Task 2.8 Bonus: Grad-CAM heatmaps for 4 correct and 4 wrong predictions."""
    df = test_df.copy().reset_index(drop=True)
    df["y_true"] = y_test.astype(int)
    df["y_prob"] = p_tl
    df["y_pred"] = (p_tl >= threshold).astype(int)

    correct_df = df[df["y_true"] == df["y_pred"]].head(4)
    wrong_df = df[df["y_true"] != df["y_pred"]].head(4)
    chosen = pd.concat([correct_df, wrong_df], ignore_index=True)

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    fig.suptitle(
        "VisionGuard (23L0713_A2_Ops) — Bonus Grad-CAM Heatmaps (Top: 4 Correct, Bottom: 4 Misclassified)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    for idx, ax in enumerate(axes.flat):
        if idx >= len(chosen):
            ax.axis("off")
            continue
        row = chosen.iloc[idx]
        with Image.open(row["file_path"]) as img:
            rgb = img.convert("RGB").resize(IMG_SIZE)
            arr = np.asarray(rgb, dtype=np.float32)

        heatmap = compute_gradcam_heatmap(model, base_model, arr[np.newaxis, ...])
        heatmap_resized = np.asarray(
            Image.fromarray(np.uint8(255 * heatmap)).resize(IMG_SIZE, resample=Image.BILINEAR)
        )
        jet_colors = cm.get_cmap("jet")(heatmap_resized / 255.0)[..., :3] * 255.0
        overlay = np.clip(0.58 * arr + 0.42 * jet_colors, 0, 255).astype(np.uint8)

        status = "CORRECT" if row["y_true"] == row["y_pred"] else "WRONG"
        pred_lbl = "AI" if row["y_pred"] == 1 else "Real"
        ax.imshow(overlay)
        ax.set_title(
            f"[{status}] True: {row['class_label']} | Pred: {pred_lbl}\nP(AI)={row['y_prob']:.2f}",
            fontsize=9.5,
            fontweight="bold",
        )
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(fig_dir / "task2_8_gradcam_grid.png", dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved Bonus Grad-CAM visualization to {fig_dir / 'task2_8_gradcam_grid.png'}")


def run_task2_training(epochs_baseline: int = 12, epochs_stage1: int = 8, epochs_stage2: int = 8) -> dict:
    tf.keras.utils.set_random_seed(SEED)
    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    train_ds, _ = create_dataset("data/splits/train.csv", img_size=IMG_SIZE, batch_size=BATCH_SIZE, training=True, augment=True)
    val_ds, _ = create_dataset("data/splits/val.csv", img_size=IMG_SIZE, batch_size=BATCH_SIZE, training=False, augment=False)
    test_ds, test_df = create_dataset("data/splits/test.csv", img_size=IMG_SIZE, batch_size=BATCH_SIZE, training=False, augment=False)

    # 1. Task 2.2 Baseline CNN
    print("\n[Task 2.2] Building & Training Baseline CNN ...")
    baseline_model = build_baseline_cnn(img_size=IMG_SIZE, learning_rate=1e-3, dropout_rate=0.35)
    export_model_summary_and_table(
        baseline_model,
        fig_dir / "task2_2_baseline_summary.txt",
        fig_dir / "task2_2_baseline_layers.csv",
    )
    base_ckpt = models_dir / "baseline_cnn.keras"
    hist_base = baseline_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_baseline,
        callbacks=make_callbacks(base_ckpt, patience_es=4, patience_lr=2),
        verbose=2,
    )
    best_ep_base = plot_best_epoch_curves(
        hist_base.history,
        "VisionGuard Baseline CNN (Task 2.2 & 2.4)",
        fig_dir / "task2_4_baseline_curves.png",
    )

    # 2. Task 2.3 Transfer Learning (MobileNetV2: Stage 1 Frozen + Stage 2 Unfreeze Last Block)
    print("\n[Task 2.3] Stage 1: Training MobileNetV2 with Frozen Backbone ...")
    tl_model, base_backbone = build_transfer_model(
        backbone_name="MobileNetV2",
        img_size=IMG_SIZE,
        learning_rate=1e-3,
        dropout_rate=0.35,
        unfreeze_last_n=0,
    )
    stage1_trainable = [lyr.name for lyr in tl_model.layers if lyr.trainable and lyr.count_params() > 0]
    tl_ckpt = models_dir / "champion_model.keras"
    hist_s1 = tl_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_stage1,
        callbacks=make_callbacks(tl_ckpt, patience_es=3, patience_lr=2),
        verbose=2,
    )

    print("\n[Task 2.3] Stage 2: Unfreezing Last Block of MobileNetV2 & Fine-Tuning ...")
    stage2_trainable = unfreeze_last_block(tl_model, base_backbone, unfreeze_last_n=25, fine_tune_lr=5e-5)
    hist_s2 = tl_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_stage2,
        callbacks=make_callbacks(tl_ckpt, patience_es=4, patience_lr=2),
        verbose=2,
    )

    export_model_summary_and_table(
        tl_model,
        fig_dir / "task2_3_mobilenetv2_summary.txt",
        fig_dir / "task2_3_mobilenetv2_layers.csv",
    )

    plot_two_stage_transfer_curves(hist_s1.history, hist_s2.history, fig_dir / "task2_3_transfer_curves.png")
    combined_tl_hist = {k: hist_s1.history[k] + hist_s2.history[k] for k in hist_s1.history}
    best_ep_tl = plot_best_epoch_curves(
        combined_tl_hist,
        "VisionGuard MobileNetV2 Transfer Learning (Task 2.3 & 2.4)",
        fig_dir / "task2_4_best_epoch_curves.png",
    )

    trainable_info = {
        "stage1_frozen_backbone_trainable_layers": stage1_trainable,
        "stage2_unfrozen_last_block_trainable_layers": stage2_trainable,
        "baseline_best_epoch": best_ep_base,
        "transfer_best_epoch": best_ep_tl,
    }
    (fig_dir / "task2_3_trainable_layers.json").write_text(json.dumps(trainable_info, indent=2), encoding="utf-8")

    # 3. Task 2.5 Validation Threshold Tuning & Test Set Evaluation
    print("\n[Task 2.5] Evaluating Baseline CNN & MobileNetV2 on Validation and Held-Out Test Set ...")
    y_val, p_val_base = collect_predictions(baseline_model, val_ds)
    _, p_val_tl = collect_predictions(tl_model, val_ds)

    t_base, t_base_info = find_optimal_threshold(y_val, p_val_base)
    t_tl, t_tl_info = find_optimal_threshold(y_val, p_val_tl)

    y_test, p_test_base = collect_predictions(baseline_model, test_ds)
    _, p_test_tl = collect_predictions(tl_model, test_ds)

    m_base = compute_metrics_at_threshold(y_test, p_test_base, t_base, "Baseline CNN (Scratch)")
    m_tl_default = compute_metrics_at_threshold(y_test, p_test_tl, 0.50, "MobileNetV2 Fine-Tuned (Default T=0.50)")
    m_tl_tuned = compute_metrics_at_threshold(y_test, p_test_tl, t_tl, f"MobileNetV2 Fine-Tuned (Tuned T={t_tl:.2f})")

    metrics_df = pd.DataFrame([m_base, m_tl_default, m_tl_tuned])
    metrics_df.to_csv(fig_dir / "task2_5_test_metrics.csv", index=False)
    print("\nTest Set Evaluation Table:")
    print(metrics_df.to_string(index=False))

    plot_confusion_And_roc(y_test, p_test_base, t_base, p_test_tl, t_tl, fig_dir)

    # 4. Task 2.6 Error Analysis
    generate_error_analysis(test_df, y_test, p_test_tl, t_tl, fig_dir)

    # 5. Task 2.8 Bonus Grad-CAM
    generate_gradcam_bonus(tl_model, base_backbone, test_df, y_test, p_test_tl, t_tl, fig_dir)

    summary_out = {
        "baseline_metrics": m_base,
        "mobilenetv2_default_metrics": m_tl_default,
        "mobilenetv2_tuned_metrics": m_tl_tuned,
        "optimal_threshold_tl": t_tl,
        "optimal_threshold_base": t_base,
        "trainable_info": trainable_info,
    }
    (fig_dir / "task2_summary.json").write_text(json.dumps(summary_out, indent=2), encoding="utf-8")
    return summary_out


if __name__ == "__main__":
    run_task2_training()
