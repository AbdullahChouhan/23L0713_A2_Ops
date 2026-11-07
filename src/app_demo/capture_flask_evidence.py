"""Capture Task 4, 5, 7 & 8 validation responses, API payloads, and visual report figures (23L0713_A2_Ops)."""

import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from app.config import AppConfig
from app.main import create_app


def render_terminal_box_figure(title: str, subtitle: str, text_blocks: list[tuple[str, str]], out_path: Path) -> None:
    """Render terminal/HTTP response evidence into a high-DPI captioned figure for the PDF report."""
    n = len(text_blocks)
    fig, axes = plt.subplots(n, 1, figsize=(13, 3.0 * n))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"{title}\n{subtitle}", fontsize=12, fontweight="bold", y=0.99)

    for ax, (block_title, block_body) in zip(axes, text_blocks):
        ax.set_facecolor("#0f172a")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#334155")
            spine.set_linewidth(1.5)
        ax.set_title(block_title, loc="left", fontsize=10.5, fontweight="bold", color="#1e293b", pad=6)
        ax.text(
            0.02,
            0.92,
            block_body,
            transform=ax.transAxes,
            fontsize=8.6,
            fontfamily="monospace",
            color="#e2e8f0",
            verticalalignment="top",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path = out_dir / "evidence_history.db"
    if db_path.exists():
        try:
            db_path.unlink()
        except OSError:
            pass

    cfg = AppConfig(
        project_name="23L0713_A2_Ops",
        model_path="models/champion_model.keras",
        model_version="VisionGuard-MobileNetV2-v2@champion",
        decision_threshold=0.50,
        max_content_length_mb=0.5,  # 512 KB cap to demonstrate Case 2 oversize rejection cleanly
        database_url=f"sqlite:///{db_path.as_posix()}",
        flask_debug=False,
    )
    app = create_app(cfg)
    client = app.test_client()

    # 1. Task 4.4 Health check response
    h_resp = client.get("/health")
    h_json = h_resp.get_json()
    (out_dir / "task4_4_health_response.json").write_text(json.dumps(h_json, indent=2), encoding="utf-8")
    render_terminal_box_figure(
        "VisionGuard (23L0713_A2_Ops) — Task 4.4 Production Health Check (/health)",
        "Student: Abdullah Chouhan (23L0713) | WSGI Production Server (FLASK_DEBUG=0)",
        [
            (
                "GET http://localhost:5000/health -> HTTP 200 OK",
                json.dumps(h_json, indent=2),
            )
        ],
        out_dir / "task4_4_health_check.png",
    )

    # 2. Task 4.2 Three Failure Cases (Wrong Extension, Oversized File, Corrupt Image Stream)
    r_ext = client.post(
        "/api/predict",
        data={"file": (io.BytesIO(b"Plain text content"), "notes_document.txt")},
        content_type="multipart/form-data",
    )
    r_size = client.post(
        "/api/predict",
        data={"file": (io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"X" * (650 * 1024)), "oversized_banner.png")},
        content_type="multipart/form-data",
    )
    r_corrupt = client.post(
        "/api/predict",
        data={"file": (io.BytesIO(b"\xff\xd8\xff\xe0CORRUPTED_TRUNCATED_BYTES"), "corrupted_upload.jpg")},
        content_type="multipart/form-data",
    )

    failure_blocks = [
        (
            f"Failure Case 1: Disallowed File Extension (.txt) -> HTTP {r_ext.status_code}",
            json.dumps(r_ext.get_json(), indent=2),
        ),
        (
            f"Failure Case 2: File Over Size Limit (> 0.5 MB) -> HTTP {r_size.status_code}",
            json.dumps(r_size.get_json(), indent=2),
        ),
        (
            f"Failure Case 3: Undecodable / Corrupt Image Stream (.jpg) -> HTTP {r_corrupt.status_code}",
            json.dumps(r_corrupt.get_json(), indent=2),
        ),
    ]
    render_terminal_box_figure(
        "VisionGuard (23L0713_A2_Ops) — Task 4.2 Input Validation & Friendly Error Handling",
        "Student: Abdullah Chouhan (23L0713) | Zero Disk Writes (In-Memory Validation)",
        failure_blocks,
        out_dir / "task4_2_three_failure_cases.png",
    )

    # 3. Task 4.3, 5, 7 & 8: Multiple Test Inputs Classified via Live Render Cloud API & Local Flask
    cloud_url = "https://two3l0713-a2-ops.onrender.com"
    cfg_full = AppConfig(
        project_name="23L0713_A2_Ops",
        model_path="models/champion_model.keras",
        model_version="VisionGuard-MobileNetV2-v2@champion",
        decision_threshold=0.50,
        max_content_length_mb=10.0,
        database_url=f"sqlite:///{db_path.as_posix()}",
        flask_debug=False,
        cloud_api_url=cloud_url,
        use_cloud_api=True,
    )
    app_full = create_app(cfg_full)
    client_full = app_full.test_client()

    test_df = pd.read_csv("data/splits/test.csv")
    svc = app_full.extensions["inference_service"]
    ai_picks, real_picks = [], []
    for _, r in test_df[test_df["label"] == 1].iterrows():
        if svc.predict_bytes(Path(r["file_path"]).read_bytes())["is_ai"]:
            ai_picks.append(r)
        if len(ai_picks) == 3:
            break
    for _, r in test_df[test_df["label"] == 0].iterrows():
        if not svc.predict_bytes(Path(r["file_path"]).read_bytes())["is_ai"]:
            real_picks.append(r)
        if len(real_picks) == 3:
            break
    sample_rows = pd.DataFrame(ai_picks + real_picks).reset_index(drop=True)

    api_responses = []
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        f"VisionGuard (23L0713_A2_Ops) — Task 7 & 8: Multiple Inputs Evaluated via Live Render API ({cloud_url})",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )

    import requests as ext_requests

    for idx, (_, row) in enumerate(sample_rows.iterrows()):
        img_path = Path(row["file_path"])
        raw_bytes = img_path.read_bytes()

        # Task 7: Direct call to live Render Cloud API endpoint
        live_r = ext_requests.post(
            f"{cloud_url}/api/predict",
            files={"file": (img_path.name, raw_bytes, "application/octet-stream")},
            timeout=45,
        )
        payload = live_r.json()
        api_responses.append(payload)

        # Task 8: Also call local Flask /predict view (which forwards to cloud_url and logs to local DB)
        _ = client_full.post(
            "/predict",
            data={"file": (io.BytesIO(raw_bytes), img_path.name), "inference_mode": "cloud"},
            content_type="multipart/form-data",
        )

        ax = axes.flat[idx]
        with Image.open(img_path) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(
            f"Cloud Log #{payload['log_id']} | File: {payload['filename'][:20]}\n"
            f"Verdict: {payload['verdict']} ({payload['confidence_pct']:.1f}%)\n"
            f"P(AI)={payload['prob_ai']:.4f} • {payload['latency_ms']:.1f} ms",
            fontsize=9,
            fontweight="bold",
            color="#c2410c" if payload["verdict"] == "AI-Generated" else "#047857",
        )
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_dir / "task8_multiple_inputs_results.png", dpi=200)
    plt.close(fig)

    (out_dir / "task7_api_postman_responses.json").write_text(json.dumps(api_responses, indent=2), encoding="utf-8")

    # Render Postman / API Request-Response Evidence Figure for Task 7
    render_terminal_box_figure(
        f"VisionGuard (23L0713_A2_Ops) — Task 7 Live Render Cloud API Response ({cloud_url}/api/predict)",
        "POST https://two3l0713-a2-ops.onrender.com/api/predict | Student: Abdullah Chouhan (23L0713)",
        [
            (
                f"Sample 1 (AI-Generated Input: {api_responses[0]['filename']}) -> HTTP 200 OK",
                json.dumps(api_responses[0], indent=2),
            ),
            (
                f"Sample 4 (Real Art Input: {api_responses[3]['filename']}) -> HTTP 200 OK",
                json.dumps(api_responses[3], indent=2),
            ),
        ],
        out_dir / "task7_api_test_evidence.png",
    )

    # Save History Trail Table for Task 5 & Task 8
    hist_resp = client_full.get("/history?format=json").get_json()
    hist_df = pd.DataFrame(hist_resp["history"])
    hist_df.to_csv(out_dir / "task5_database_history_trail.csv", index=False)
    print("[INFO] Captured Flask validation, API responses, and database history trail:")
    print(hist_df[["id", "created_at", "filename", "verdict", "confidence_pct", "prob_ai", "model_version", "latency_ms"]].to_string(index=False))


if __name__ == "__main__":
    main()
