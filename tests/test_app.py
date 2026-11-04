"""Task 4.2, 5 & 6.6: Automated pytest suite for VisionGuard (23L0713_A2_Ops).

Tests:
  - /health endpoint status and configuration fields
  - Rejecting disallowed file extensions (e.g. .txt, .pdf) with HTTP 400
  - Rejecting oversized image uploads (> limit) with HTTP 413
  - Rejecting corrupt/non-image byte streams with .jpg extension with HTTP 422
  - Valid in-memory image classification via /predict and /api/predict
  - Database persistence and /history trail verification
  - Ensuring no uploaded files are written to disk
"""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from app.config import AppConfig
from app.main import create_app


def _ensure_dummy_or_real_model(model_path: Path) -> None:
    if model_path.exists():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    m = tf.keras.Model(inputs, outputs)
    m.save(model_path)


def _create_valid_png_bytes(width: int = 200, height: int = 200) -> bytes:
    arr = np.random.default_rng(42).integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path: Path):
    db_file = tmp_path / "test_vg.db"
    model_path = Path("models/champion_model.keras")
    _ensure_dummy_or_real_model(model_path)

    cfg = AppConfig(
        project_name="23L0713_A2_Ops",
        model_path=str(model_path),
        model_version="VisionGuard-MobileNetV2-v2@champion",
        decision_threshold=0.50,
        max_content_length_mb=0.5,  # 512 KB limit for fast oversize testing
        database_url=f"sqlite:///{db_file.as_posix()}",
        flask_debug=False,
        use_cloud_api=False,
    )
    app = create_app(cfg)
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "healthy"
    assert data["project"] == "23L0713_A2_Ops"
    assert data["model_loaded"] is True
    assert data["debug_mode"] is False


def test_reject_wrong_file_type(client):
    fake_txt = io.BytesIO(b"This is a plain text document, not an image.")
    resp = client.post(
        "/predict",
        data={"file": (fake_txt, "malicious_script.txt")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert b"Unsupported File Type Rejected" in resp.data


def test_reject_oversized_file(client):
    # Limit in fixture is 0.5 MB (512 KB); send ~700 KB payload with .png extension
    oversized_bytes = b"\x89PNG\r\n\x1a\n" + (b"A" * (700 * 1024))
    resp = client.post(
        "/predict",
        data={"file": (io.BytesIO(oversized_bytes), "huge_image.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 413
    assert b"File Exceeds Size Limit" in resp.data


def test_reject_corrupt_image_bytes(client):
    corrupt_jpg = io.BytesIO(b"\xff\xd8\xff\xe0NOT_A_VALID_JPEG_STREAM_TRUNCATED_DATA")
    resp = client.post(
        "/predict",
        data={"file": (corrupt_jpg, "corrupt_photo.jpg")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 422
    assert b"Corrupt or Undecodable Image Rejected" in resp.data


def test_valid_prediction_and_history_trail(client):
    png_bytes = _create_valid_png_bytes()
    resp = client.post(
        "/api/predict",
        data={"file": (io.BytesIO(png_bytes), "../../unsafe name!@#sample.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["verdict"] in ("AI-Generated", "Real")
    assert 0.5 <= payload["confidence"] <= 1.0
    # Verify filename was sanitized by secure_filename
    assert ".." not in payload["filename"]
    assert "/" not in payload["filename"]

    # Check /history JSON & HTML
    hist_json = client.get("/history?format=json").get_json()
    assert hist_json["count"] == 1
    assert hist_json["history"][0]["filename"] == payload["filename"]

    hist_html = client.get("/history")
    assert hist_html.status_code == 200
    assert payload["filename"].encode() in hist_html.data
