"""Task 4.2, 4.3, 4.4, 5, 7 & 8: VisionGuard Flask Web Application & REST API (23L0713_A2_Ops)."""

import base64
import datetime
import io
import os
import time
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template, request
from PIL import Image
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from app.config import AppConfig
from app.database import DatabaseManager
from app.inference import VisionGuardInferenceService

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def create_app(custom_config: AppConfig | None = None) -> Flask:
    cfg = custom_config or AppConfig()
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = cfg.max_content_length_bytes
    app.config["VG_CONFIG"] = cfg

    # Initialize Database Trail (Task 5)
    db_manager = DatabaseManager(cfg.database_url)
    try:
        db_manager.init_db()
    except Exception as exc:
        app.logger.error(f"Database initialization failed: {exc}")

    # Load Champion Keras Model Once at Startup (Task 4.1)
    inference_service = VisionGuardInferenceService(
        model_path=cfg.model_path,
        model_version=cfg.model_version,
        threshold=cfg.decision_threshold,
    )

    app.extensions["db_manager"] = db_manager
    app.extensions["inference_service"] = inference_service

    def _validate_upload(file_storage) -> tuple[str | None, bytes | None, tuple[str, str, int] | None]:
        """Validate uploaded file extension, size, and image stream without writing to disk."""
        if file_storage is None or not file_storage.filename:
            return None, None, ("No File Selected", "Please select an image file before submitting.", 400)

        raw_filename = file_storage.filename
        ext = Path(raw_filename).suffix.lower()
        safe_name = secure_filename(raw_filename) or f"upload{ext}"
        if not Path(safe_name).suffix and ext:
            safe_name = f"{safe_name}{ext}"

        if ext not in ALLOWED_EXTENSIONS:
            return (
                None,
                None,
                (
                    "Unsupported File Type Rejected",
                    f"File '{safe_name}' has disallowed extension '{ext or 'none'}'. "
                    f"Only {', '.join(sorted(ALLOWED_EXTENSIONS))} image files are accepted.",
                    400,
                ),
            )

        raw_bytes = file_storage.read()
        if len(raw_bytes) == 0:
            return None, None, ("Empty File Rejected", f"Uploaded file '{safe_name}' contains 0 bytes.", 400)

        if len(raw_bytes) > cfg.max_content_length_bytes:
            size_mb = len(raw_bytes) / (1024 * 1024)
            return (
                None,
                None,
                (
                    "File Exceeds Size Limit",
                    f"File '{safe_name}' ({size_mb:.2f} MB) exceeds the maximum allowed limit of {cfg.max_content_length_mb:.1f} MB.",
                    413,
                ),
            )

        try:
            VisionGuardInferenceService.validate_and_preprocess_bytes(raw_bytes)
        except ValueError as exc:
            return (
                None,
                None,
                (
                    "Corrupt or Undecodable Image Rejected",
                    f"File '{safe_name}' has an image extension but could not be decoded as a valid image. ({exc})",
                    422,
                ),
            )

        return safe_name, raw_bytes, None

    def _make_data_uri(raw_bytes: bytes) -> str:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            rgb = img.convert("RGB")
            rgb.thumbnail((420, 420))
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"

    @app.errorhandler(413)
    @app.errorhandler(RequestEntityTooLarge)
    def handle_file_too_large(_err):
        title = "File Exceeds Size Limit"
        msg = f"The uploaded file exceeds the maximum allowed size limit of {cfg.max_content_length_mb:.1f} MB."
        if request.path.startswith("/api/"):
            return jsonify({"error": title, "message": msg, "status_code": 413, "project": cfg.project_name}), 413
        return render_template("error.html", error_title=title, error_message=msg, status_code=413, config=cfg), 413

    @app.route("/", methods=["GET"])
    def index():
        recent_logs = []
        try:
            recent_logs = db_manager.get_recent_logs(limit=5)
        except Exception:
            pass
        return render_template("index.html", config=cfg, recent_logs=recent_logs)

    @app.route("/predict", methods=["POST"])
    def predict_view():
        file_obj = request.files.get("file")
        safe_name, raw_bytes, err = _validate_upload(file_obj)
        if err is not None:
            title, msg, code = err
            return render_template("error.html", error_title=title, error_message=msg, status_code=code, config=cfg), code

        size_kb = len(raw_bytes) / 1024.0
        use_remote = cfg.use_cloud_api or (request.form.get("inference_mode") == "cloud" and bool(cfg.cloud_api_url))

        try:
            if use_remote and cfg.cloud_api_url:
                t0 = time.perf_counter()
                api_endpoint = cfg.cloud_api_url.rstrip("/") + "/api/predict"
                resp = requests.post(
                    api_endpoint,
                    files={"file": (safe_name, raw_bytes, "application/octet-stream")},
                    headers={"X-API-Key": cfg.cloud_api_key},
                    timeout=30,
                )
                resp.raise_for_status()
                result = resp.json()
                result["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
                inference_source = f"cloud_api ({cfg.cloud_api_url})"
            else:
                result = inference_service.predict_bytes(raw_bytes)
                inference_source = "local_model"
        except Exception as exc:
            return (
                render_template(
                    "error.html",
                    error_title="Inference Service Error",
                    error_message=f"Could not complete model inference: {exc}",
                    status_code=500,
                    config=cfg,
                ),
                500,
            )

        log_entry = db_manager.log_prediction(
            filename=safe_name,
            file_size_kb=size_kb,
            image_dimensions=result["image_meta"]["dimensions_str"],
            verdict=result["verdict"],
            confidence=result["confidence"],
            prob_ai=result["prob_ai"],
            threshold=result["threshold"],
            model_version=result["model_version"],
            inference_source=inference_source,
            latency_ms=result["latency_ms"],
        )

        preview_uri = _make_data_uri(raw_bytes)
        return render_template(
            "result.html",
            result=result,
            log_entry=log_entry,
            preview_uri=preview_uri,
            config=cfg,
        )

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        """Task 7 & 8: Public REST API endpoint for Postman and remote Flask integration."""
        if os.getenv("REQUIRE_API_KEY", "false").lower() in ("1", "true", "yes"):
            provided_key = request.headers.get("X-API-Key", "")
            if provided_key != cfg.cloud_api_key:
                return jsonify({"error": "Unauthorized", "message": "Invalid or missing X-API-Key header."}), 401

        file_obj = request.files.get("file")
        safe_name, raw_bytes, err = _validate_upload(file_obj)
        if err is not None:
            title, msg, code = err
            return jsonify({"error": title, "message": msg, "status_code": code, "project": cfg.project_name}), code

        size_kb = len(raw_bytes) / 1024.0
        result = inference_service.predict_bytes(raw_bytes)
        log_entry = db_manager.log_prediction(
            filename=safe_name,
            file_size_kb=size_kb,
            image_dimensions=result["image_meta"]["dimensions_str"],
            verdict=result["verdict"],
            confidence=result["confidence"],
            prob_ai=result["prob_ai"],
            threshold=result["threshold"],
            model_version=result["model_version"],
            inference_source="api_endpoint",
            latency_ms=result["latency_ms"],
        )
        return (
            jsonify(
                {
                    "project": cfg.project_name,
                    "student": f"{cfg.student_name} ({cfg.roll_number})",
                    "filename": safe_name,
                    "file_size_kb": round(size_kb, 2),
                    **result,
                    "log_id": log_entry["id"],
                    "logged_at": log_entry["created_at"],
                }
            ),
            200,
        )

    @app.route("/history", methods=["GET"])
    def history_view():
        logs = db_manager.get_recent_logs(limit=100)
        if request.args.get("format") == "json":
            return jsonify({"project": cfg.project_name, "count": len(logs), "history": logs}), 200
        return render_template("history.html", logs=logs, config=cfg)

    @app.route("/health", methods=["GET"])
    def health_check():
        db_ok, db_msg = db_manager.check_health()
        model_ok = inference_service.is_ready
        overall_ok = db_ok and model_ok
        payload = {
            "status": "healthy" if overall_ok else "unhealthy",
            "project": cfg.project_name,
            "student": f"{cfg.student_name} ({cfg.roll_number})",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "model_loaded": model_ok,
            "model_path": cfg.model_path,
            "model_version": cfg.model_version,
            "model_error": inference_service.load_error,
            "decision_threshold": cfg.decision_threshold,
            "database_status": db_msg,
            "use_cloud_api": cfg.use_cloud_api,
            "cloud_api_url": cfg.cloud_api_url or None,
            "debug_mode": app.debug,
        }
        return jsonify(payload), (200 if overall_ok else 503)

    return app


app = create_app()

if __name__ == "__main__":
    cfg = AppConfig()
    # Task 4.4: Always run with debug=False in production mode
    from waitress import serve

    print(f"[VisionGuard {cfg.project_name}] Starting production WSGI server on http://0.0.0.0:{cfg.port} (debug=False)")
    serve(app, host="0.0.0.0", port=cfg.port)
