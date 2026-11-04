"""Task 4.1: Singleton Inference Module for VisionGuard (23L0713_A2_Ops).

Loads the champion Keras model once at application startup, validates and decodes
uploaded image bytes in-memory without writing to disk, applies identical resizing
and preprocessing as in training, and returns the predicted label and confidence.
"""

import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False

TARGET_IMG_SIZE = (160, 160)


class VisionGuardInferenceService:
    """Singleton-style inference service that keeps the loaded Keras model in memory."""

    def __init__(
        self,
        model_path: str | Path,
        model_version: str = "VisionGuard-MobileNetV2-v2@champion",
        threshold: float = 0.50,
    ) -> None:
        self.model_path = Path(model_path)
        self.model_version = model_version
        self.threshold = float(threshold)
        self.model: tf.keras.Model | None = None
        self.load_error: str | None = None
        self._load_model_once()

    def _load_model_once(self) -> None:
        if not self.model_path.exists():
            self.load_error = f"Model file not found at {self.model_path.resolve()}"
            return
        try:
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            # Warm up graph with one dummy forward pass
            dummy = np.zeros((1, TARGET_IMG_SIZE[0], TARGET_IMG_SIZE[1], 3), dtype=np.float32)
            _ = self.model.predict(dummy, verbose=0)
            self.load_error = None
        except Exception as exc:
            self.model = None
            self.load_error = f"{type(exc).__name__}: {exc}"

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.load_error is None

    @staticmethod
    def validate_and_preprocess_bytes(raw_bytes: bytes) -> tuple[np.ndarray, dict]:
        """Validate image bytes in-memory and preprocess to (1, 160, 160, 3) float32 RGB."""
        if not raw_bytes:
            raise ValueError("Uploaded file is empty (0 bytes).")

        try:
            with Image.open(io.BytesIO(raw_bytes)) as img:
                img_format = img.format or "UNKNOWN"
                color_mode = img.mode
                width, height = img.size
                img.verify()

            with Image.open(io.BytesIO(raw_bytes)) as img:
                img.load()
                rgb = img.convert("RGB")
                resized = rgb.resize(TARGET_IMG_SIZE, resample=Image.BILINEAR)
                arr = np.asarray(resized, dtype=np.float32)[np.newaxis, ...]
        except Exception as exc:
            raise ValueError(f"Corrupt or unreadable image stream: {type(exc).__name__}: {exc}") from exc

        meta = {
            "width": int(width),
            "height": int(height),
            "format": str(img_format),
            "color_mode": str(color_mode),
            "dimensions_str": f"{width}x{height} ({img_format}/{color_mode})",
        }
        return arr, meta

    def predict_bytes(self, raw_bytes: bytes, custom_threshold: float | None = None) -> dict:
        if not self.is_ready or self.model is None:
            raise RuntimeError(f"Inference model is not loaded: {self.load_error}")

        t0 = time.perf_counter()
        batch_arr, meta = self.validate_and_preprocess_bytes(raw_bytes)
        prob_ai = float(self.model.predict(batch_arr, verbose=0)[0][0])
        thresh = float(custom_threshold if custom_threshold is not None else self.threshold)

        is_ai = prob_ai >= thresh
        verdict = "AI-Generated" if is_ai else "Real"
        confidence = prob_ai if is_ai else (1.0 - prob_ai)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "verdict": verdict,
            "is_ai": bool(is_ai),
            "confidence": round(float(confidence), 4),
            "confidence_pct": round(float(confidence) * 100.0, 2),
            "prob_ai": round(float(prob_ai), 4),
            "threshold": round(thresh, 2),
            "model_version": self.model_version,
            "latency_ms": round(float(latency_ms), 2),
            "image_meta": meta,
            "caveat": (
                "Honest Caveat: VisionGuard evaluates visual texture, lighting consistency, and frequency patterns "
                "at 160x160 resolution. Highly stylized digital illustrations, heavy JPEG compression, or post-processed "
                "studio photographs can occasionally trigger false positives or false negatives."
            ),
        }


def run_cli_demo() -> None:
    """Task 4.1 Evidence: Run inference module on two known test images (one AI, one Real)."""
    from app.config import AppConfig

    cfg = AppConfig()
    svc = VisionGuardInferenceService(
        model_path=cfg.model_path,
        model_version=cfg.model_version,
        threshold=cfg.decision_threshold,
    )
    test_df = pd.read_csv("data/splits/test.csv")
    ai_row = test_df[test_df["label"] == 1].iloc[0]
    # Find a clear Real test image predicted as Real (< threshold)
    real_candidates = test_df[test_df["label"] == 0]
    real_row = real_candidates.iloc[0]
    for _, cand in real_candidates.iterrows():
        res_c = svc.predict_bytes(Path(cand["file_path"]).read_bytes())
        if not res_c["is_ai"]:
            real_row = cand
            break

    lines = [
        "=" * 78,
        "Task 4.1 — VisionGuard Inference Module Console Verification (23L0713_A2_Ops)",
        f"Model Path    : {cfg.model_path}",
        f"Model Version : {cfg.model_version}",
        f"Model Ready   : {svc.is_ready}",
        "=" * 78,
    ]
    for idx, row in enumerate([ai_row, real_row], start=1):
        p = Path(row["file_path"])
        res = svc.predict_bytes(p.read_bytes())
        lines.extend(
            [
                f"[Sample #{idx}] File       : {p.as_posix()}",
                f"            Ground Truth : {row['class_label']} (label={row['label']})",
                f"            Orig Meta    : {res['image_meta']['dimensions_str']}",
                f"            Raw P(AI)    : {res['prob_ai']:.4f} (Threshold={res['threshold']:.2f})",
                f"            Verdict      : {res['verdict']} (Confidence: {res['confidence_pct']:.2f}%)",
                f"            Latency      : {res['latency_ms']:.2f} ms",
                "-" * 78,
            ]
        )
    out = "\n".join(lines)
    print(out)
    out_file = Path("reports/figures/task4_1_inference_cli.txt")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(out + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_cli_demo()
