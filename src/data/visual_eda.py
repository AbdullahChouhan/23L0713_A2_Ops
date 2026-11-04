"""Task 1.3: Visual Exploratory Data Analysis (EDA) for VisionGuard (23L0713_A2_Ops).

Generates:
  1. Grid of sample images per class (AI vs Real) with original dimensions and format annotated.
  2. Histograms of image width, height, and file size (KB) per class with median markers.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image


def generate_eda_figures(audit_csv: Path = Path("data/audit_table.csv")) -> None:
    df = pd.read_csv(audit_csv)
    clean_df = df[~df["is_corrupt"] & ~df["is_duplicate"]].copy()

    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sample Grid per Class (2 rows x 5 columns: Top row = AI-Generated, Bottom row = Real)
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    fig.suptitle(
        "VisionGuard (23L0713_A2_Ops) — Sample Images per Class (Top: AI-Generated, Bottom: Real Art)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    for row_idx, cls_label in enumerate(["AI", "Real"]):
        cls_subset = clean_df[clean_df["class_label"] == cls_label]
        chosen = cls_subset.sample(n=min(5, len(cls_subset)), random_state=42)
        for col_idx, (_, row) in enumerate(chosen.iterrows()):
            ax = axes[row_idx, col_idx]
            with Image.open(row["file_path"]) as img:
                rgb_img = img.convert("RGB")
                ax.imshow(rgb_img)
            ax.set_title(
                f"{cls_label} | {int(row['width'])}x{int(row['height'])}\n"
                f"{row['image_format']} ({row['color_mode']}) • {row['file_size_kb']:.0f} KB",
                fontsize=9,
            )
            ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    grid_path = out_dir / "task1_3_sample_grid.png"
    plt.savefig(grid_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved sample image grid to {grid_path}")

    # 2. Histograms of Width, Height, and File Size per Class
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(
        "VisionGuard (23L0713_A2_Ops) — Distributions of Width, Height, and File Size by Class",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    palette = {"AI": "#d95f02", "Real": "#1b9e77"}
    metrics = [
        ("width", "Image Width (pixels)", "Distribution of Image Width by Class"),
        ("height", "Image Height (pixels)", "Distribution of Image Height by Class"),
        ("file_size_kb", "File Size (KB)", "Distribution of File Size (KB) by Class"),
    ]

    for ax, (col, xlabel, title) in zip(axes, metrics):
        sns.histplot(
            data=clean_df,
            x=col,
            hue="class_label",
            palette=palette,
            bins=30,
            kde=True,
            alpha=0.55,
            ax=ax,
        )
        for cls_label, color in palette.items():
            med = clean_df[clean_df["class_label"] == cls_label][col].median()
            ax.axvline(med, color=color, linestyle="--", linewidth=1.8, label=f"{cls_label} Median: {med:.0f}")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Image Count", fontsize=10)
        ax.legend(fontsize=8)

    plt.tight_layout()
    hist_path = out_dir / "task1_3_distributions_hist.png"
    plt.savefig(hist_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved distribution histograms to {hist_path}")


if __name__ == "__main__":
    generate_eda_figures()
