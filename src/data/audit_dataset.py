"""Task 1.2: Dataset audit script for VisionGuard (23L0713_A2_Ops).

Walks the dataset directory and builds a comprehensive table for every file:
  - class_name, label_id (1=AI, 0=Real)
  - file_size_bytes, file_size_kb
  - width, height, color_mode, image_format
  - sha256 content hash
  - is_corrupt (unreadable or truncated), is_duplicate (exact SHA-256 match)
Also generates stratified train/val/test split CSVs from clean, unique images.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split

ImageFile.LOAD_TRUNCATED_IMAGES = False


CLASS_MAPPING = {
    "AiArtData": ("AI", 1),
    "RealArt": ("Real", 0),
}


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_image(filepath: Path) -> dict:
    size_bytes = filepath.stat().st_size
    sha256_hash = compute_sha256(filepath)
    record = {
        "file_path": filepath.as_posix(),
        "file_name": filepath.name,
        "file_size_bytes": size_bytes,
        "file_size_kb": round(size_bytes / 1024.0, 2),
        "width": None,
        "height": None,
        "color_mode": None,
        "image_format": None,
        "sha256": sha256_hash,
        "is_corrupt": False,
        "corrupt_reason": "",
    }
    if size_bytes == 0:
        record["is_corrupt"] = True
        record["corrupt_reason"] = "Zero-byte file"
        return record

    try:
        with Image.open(filepath) as img:
            img_format = img.format
            color_mode = img.mode
            width, height = img.size
            img.verify()
        # Re-open and force full pixel decoding to catch truncated/broken streams
        with Image.open(filepath) as img:
            img.load()
            _ = img.convert("RGB")
        record["width"] = int(width)
        record["height"] = int(height)
        record["color_mode"] = str(color_mode)
        record["image_format"] = str(img_format)
    except Exception as exc:
        record["is_corrupt"] = True
        record["corrupt_reason"] = f"{type(exc).__name__}: {exc}"

    return record


def run_audit(raw_dir: Path = Path("data/raw")) -> pd.DataFrame:
    rows = []
    for folder_name, (class_label, label_id) in CLASS_MAPPING.items():
        class_dir = raw_dir / folder_name
        if not class_dir.exists():
            continue
        for file_path in sorted(class_dir.rglob("*")):
            if file_path.is_file():
                info = inspect_image(file_path)
                info["class_folder"] = folder_name
                info["class_label"] = class_label
                info["label"] = label_id
                rows.append(info)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No files found in {raw_dir}. Run download_dataset.py first.")

    # Mark exact content duplicates via SHA-256 hash
    df["is_duplicate"] = df.duplicated(subset=["sha256"], keep="first")
    all_dup_mask = df.duplicated(subset=["sha256"], keep=False)
    df["duplicate_group_size"] = df.groupby("sha256")["sha256"].transform("count")

    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "audit_table.csv", index=False)

    # Compute summary per class and overall
    valid_df = df[~df["is_corrupt"]].copy()
    summary_rows = []
    for cls_name, grp in list(df.groupby("class_label")) + [("Total / Overall", df)]:
        v_grp = grp[~grp["is_corrupt"]]
        summary_rows.append(
            {
                "Class": cls_name,
                "Total Files": int(len(grp)),
                "Corrupt / Unreadable": int(grp["is_corrupt"].sum()),
                "Exact Duplicates (Removed)": int(grp["is_duplicate"].sum()),
                "Duplicate Files Involved": int(grp.duplicated(subset=["sha256"], keep=False).sum()),
                "Clean Unique Images": int((~grp["is_corrupt"] & ~grp["is_duplicate"]).sum()),
                "Width Min (px)": int(v_grp["width"].min()) if not v_grp.empty else 0,
                "Width Median (px)": float(v_grp["width"].median()) if not v_grp.empty else 0.0,
                "Width Max (px)": int(v_grp["width"].max()) if not v_grp.empty else 0,
                "Height Min (px)": int(v_grp["height"].min()) if not v_grp.empty else 0,
                "Height Median (px)": float(v_grp["height"].median()) if not v_grp.empty else 0.0,
                "Height Max (px)": int(v_grp["height"].max()) if not v_grp.empty else 0,
                "Size Min (KB)": round(float(v_grp["file_size_kb"].min()), 2) if not v_grp.empty else 0.0,
                "Size Median (KB)": round(float(v_grp["file_size_kb"].median()), 2) if not v_grp.empty else 0.0,
                "Size Max (KB)": round(float(v_grp["file_size_kb"].max()), 2) if not v_grp.empty else 0.0,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    reports_dir = Path("reports/figures")
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(reports_dir / "task1_2_audit_summary.csv", index=False)

    # Save duplicate and corrupt details
    dups_df = df[all_dup_mask].sort_values("sha256")
    dups_df.to_csv(reports_dir / "task1_2_duplicates.csv", index=False)
    corrupt_df = df[df["is_corrupt"]]
    corrupt_df.to_csv(reports_dir / "task1_2_corrupt.csv", index=False)

    # Also save format & colour mode breakdown
    fmt_counts = valid_df.groupby(["class_label", "image_format", "color_mode"]).size().reset_index(name="count")
    fmt_counts.to_csv(reports_dir / "task1_2_formats_modes.csv", index=False)

    # Create clean RGB standardized cache if needed by tf.io.decode_image (e.g., WEBP/TIFF/CMYK -> PNG/JPG)
    # Wait: let's check which formats tf.io.decode_image supports natively (JPEG, PNG, BMP, GIF).
    # For any clean unique image whose format is not in (JPEG, PNG, BMP, GIF) or mode not in (RGB, L, RGBA),
    # we also write a clean RGB PNG copy under data/processed_rgb/ or re-save in place?
    # Keeping data/raw untouched for DVC integrity, we store paths in splits and handle decoding safely in tf.data!
    clean_df = df[~df["is_corrupt"] & ~df["is_duplicate"]].copy().reset_index(drop=True)

    # Create deterministic 70/15/15 stratified splits (seed=42)
    train_df, temp_df = train_test_split(
        clean_df,
        test_size=0.30,
        stratify=clean_df["label"],
        random_state=42,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=42,
    )

    splits_dir = Path("data/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)
    cols = ["file_path", "class_label", "label", "width", "height", "file_size_kb", "image_format", "color_mode", "sha256"]
    train_df[cols].to_csv(splits_dir / "train.csv", index=False)
    val_df[cols].to_csv(splits_dir / "val.csv", index=False)
    test_df[cols].to_csv(splits_dir / "test.csv", index=False)

    audit_report = {
        "project": "23L0713_A2_Ops",
        "total_files": int(len(df)),
        "corrupt_files_count": int(df["is_corrupt"].sum()),
        "duplicate_files_removed": int(df["is_duplicate"].sum()),
        "clean_unique_files": int(len(clean_df)),
        "splits": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "formats": valid_df["image_format"].value_counts().to_dict(),
        "color_modes": valid_df["color_mode"].value_counts().to_dict(),
        "summary_table": summary_rows,
    }
    (reports_dir / "task1_2_audit_report.json").write_text(json.dumps(audit_report, indent=2), encoding="utf-8")

    print("=" * 80)
    print("VisionGuard Dataset Audit Summary (23L0713_A2_Ops)")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("-" * 80)
    print("Formats & Color Modes Breakdown:")
    print(fmt_counts.to_string(index=False))
    print("-" * 80)
    print(f"Splits created (seed=42): Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    if not dups_df.empty:
        print(f"Detected {df['is_duplicate'].sum()} duplicate files across {dups_df['sha256'].nunique()} SHA-256 groups.")
    if not corrupt_df.empty:
        print(f"Detected {len(corrupt_df)} corrupt/unreadable files.")
    print("=" * 80)
    return df


if __name__ == "__main__":
    run_audit()
