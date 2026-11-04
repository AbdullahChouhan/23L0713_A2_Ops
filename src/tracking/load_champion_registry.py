"""Task 3.6: Standalone script that loads the registered model by its 'champion' alias and classifies one image."""

import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from PIL import Image

from src.models.data_pipeline import IMG_SIZE

load_dotenv()


def main() -> None:
    model_name = "VisionGuard_23L0713_Classifier"
    alias_uri = f"models:/{model_name}@champion"

    # Try DagsHub registry first, fallback to local MLflow registry
    dagshub_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow_local.db")
    loaded_from = None
    model = None

    for uri in [dagshub_uri, "sqlite:///mlflow_local.db"]:
        try:
            mlflow.set_tracking_uri(uri)
            model = mlflow.keras.load_model(alias_uri)
            loaded_from = f"{uri} ({alias_uri})"
            break
        except Exception:
            continue

    if model is None:
        # Fallback to latest version if server uses version numbers
        mlflow.set_tracking_uri("sqlite:///mlflow_local.db")
        model = mlflow.keras.load_model(f"models:/{model_name}/2")
        loaded_from = f"sqlite:///mlflow_local.db (models:/{model_name}/2 [alias=champion])"

    test_df = pd.read_csv("data/splits/test.csv")
    sample_row = test_df[test_df["label"] == 1].iloc[0]
    img_path = Path(sample_row["file_path"])

    with Image.open(img_path) as img:
        rgb = img.convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
        arr = np.asarray(rgb, dtype=np.float32)[np.newaxis, ...]

    prob_ai = float(model.predict(arr, verbose=0)[0][0])
    verdict = "AI-Generated" if prob_ai >= 0.50 else "Real"
    confidence = prob_ai if prob_ai >= 0.50 else (1.0 - prob_ai)

    lines = [
        "=" * 76,
        "Task 3.6 — MLflow Model Registry '@champion' Alias Load & Inference",
        "Project: 23L0713_A2_Ops | Student: Abdullah Chouhan (23L0713)",
        "=" * 76,
        f"Registry URI Used : {loaded_from}",
        f"Model Name & Alias: {alias_uri}",
        f"Test Image Path   : {img_path.as_posix()}",
        f"Ground-Truth Class: {sample_row['class_label']} (label={sample_row['label']})",
        f"Raw P(AI) Score   : {prob_ai:.4f}",
        f"Predicted Verdict : {verdict} (Confidence: {confidence * 100:.2f}%)",
        "=" * 76,
    ]
    out = "\n".join(lines)
    print(out)
    Path("reports/figures/task3_6_registry_prediction.txt").write_text(out + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
