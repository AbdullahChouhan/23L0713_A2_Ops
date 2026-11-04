"""Task 1.1: Environment verification & GPU check for 23L0713_A2_Ops."""

import datetime
import importlib.metadata
import platform
import sys
from pathlib import Path

import tensorflow as tf


def main() -> None:
    packages = [
        "tensorflow",
        "keras",
        "keras-tuner",
        "mlflow",
        "dagshub",
        "dvc",
        "dvc-s3",
        "flask",
        "sqlalchemy",
        "pillow",
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "pytest",
        "ruff",
    ]

    lines = [
        "=" * 72,
        "VisionGuard Environment Verification — Project: 23L0713_A2_Ops",
        "Student: Abdullah Chouhan (Roll No: 23L0713)",
        f"Timestamp: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"Python Executable: {sys.executable}",
        f"Python Version: {sys.version.splitlines()[0]}",
        f"Platform: {platform.platform()}",
        "=" * 72,
        f"{'Package':<22} | {'Installed Version':<20}",
        "-" * 72,
    ]

    for pkg in packages:
        try:
            ver = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            ver = "NOT INSTALLED"
        lines.append(f"{pkg:<22} | {ver:<20}")

    gpus = tf.config.list_physical_devices("GPU")
    cpus = tf.config.list_physical_devices("CPU")
    lines.extend(
        [
            "-" * 72,
            f"TensorFlow Built with CUDA : {tf.test.is_built_with_cuda()}",
            f"Physical CPU Devices       : {len(cpus)} -> {cpus}",
            f"Physical GPU Devices       : {len(gpus)} -> {gpus if gpus else 'None (CPU-only execution active)'}",
            "=" * 72,
        ]
    )

    output = "\n".join(lines)
    print(output)

    out_path = Path("reports/figures/task1_1_env_versions.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
