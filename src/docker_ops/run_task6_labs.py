"""Task 6.1 – 6.5: Automated Docker Warm-Up, Image Size Analysis, Compose Verification & Troubleshooting (23L0713_A2_Ops)."""

import json
import socket
import subprocess
import time
from pathlib import Path

import pandas as pd
import requests


def run_cmd(cmd: list[str], cwd: Path = Path(".")) -> str:
    cmd_str = " ".join(cmd)
    res = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    out = (res.stdout + "\n" + res.stderr).strip()
    return f"$ {cmd_str}\n{out if out else '(exit 0)'}\n"


def run_task6_1_warmup(fig_dir: Path) -> None:
    print("[Task 6.1] Running Docker warm-up exercises ...")
    vol_dir = Path("toy_labs/docker_warmup_html").resolve()
    vol_dir.mkdir(parents=True, exist_ok=True)
    (vol_dir / "index.html").write_text(
        "<h1>VisionGuard Docker Warm-up (23L0713_A2_Ops - Abdullah Chouhan)</h1>\n",
        encoding="utf-8",
    )

    subprocess.run(["docker", "rm", "-f", "vg_warmup_nginx"], capture_output=True, check=False)
    logs = [
        "=" * 78,
        "Task 6.1 — Docker Warm-Up Execution Log (Project: 23L0713_A2_Ops)",
        "Student: Abdullah Chouhan (23L0713)",
        "=" * 78,
    ]
    logs.append(run_cmd(["docker", "pull", "nginx:alpine"]))
    logs.append(
        run_cmd(
            [
                "docker",
                "run",
                "-d",
                "--name",
                "vg_warmup_nginx",
                "-p",
                "8088:80",
                "-v",
                f"{vol_dir}:/usr/share/nginx/html:ro",
                "nginx:alpine",
            ]
        )
    )
    time.sleep(2)
    try:
        r = requests.get("http://127.0.0.1:8088", timeout=5)
        logs.append(f"$ curl http://127.0.0.1:8088\nHTTP {r.status_code}: {r.text.strip()}\n")
    except Exception as exc:
        logs.append(f"HTTP check note: {exc}\n")

    logs.append(run_cmd(["docker", "ps", "--filter", "name=vg_warmup_nginx"]))
    logs.append(run_cmd(["docker", "logs", "vg_warmup_nginx"]))
    logs.append(
        run_cmd(
            [
                "docker",
                "exec",
                "vg_warmup_nginx",
                "sh",
                "-c",
                "whoami && uname -a && cat /usr/share/nginx/html/index.html",
            ]
        )
    )
    logs.append(run_cmd(["docker", "images", "nginx:alpine"]))
    logs.append(run_cmd(["docker", "rm", "-f", "vg_warmup_nginx"]))
    logs.append(run_cmd(["docker", "rmi", "nginx:alpine"]))

    (fig_dir / "task6_1_docker_warmup.txt").write_text("\n".join(logs), encoding="utf-8")
    print("[Task 6.1] Saved warm-up log to reports/figures/task6_1_docker_warmup.txt")


def run_task6_2_and_6_4_image_analysis(fig_dir: Path) -> None:
    print("[Task 6.2 & 6.4] Building optimized and unoptimized Docker images & analyzing layers ...")
    logs = [
        "=" * 78,
        "Task 6.2 & 6.4 — Dockerfile Verification, Non-Root User & Image Size Optimization",
        "Project: 23L0713_A2_Ops | Student: Abdullah Chouhan (23L0713)",
        "=" * 78,
    ]
    logs.append(run_cmd(["docker", "build", "-t", "23l0713_a2_ops_web:latest", "-f", "Dockerfile", "."]))
    logs.append(run_cmd(["docker", "run", "--rm", "23l0713_a2_ops_web:latest", "sh", "-c", "whoami && id && pwd"]))
    logs.append(run_cmd(["docker", "build", "-t", "23l0713_a2_ops_web:unoptimized", "-f", "Dockerfile.unoptimized", "."]))
    logs.append(
        run_cmd(
            [
                "docker",
                "images",
                "--format",
                "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.Size}}",
                "23l0713_a2_ops_web",
            ]
        )
    )
    logs.append(run_cmd(["docker", "history", "--no-trunc=false", "23l0713_a2_ops_web:latest"]))
    logs.append(run_cmd(["docker", "history", "--no-trunc=false", "23l0713_a2_ops_web:unoptimized"]))

    # Remove unoptimized image after recording metrics to save disk space
    subprocess.run(["docker", "rmi", "23l0713_a2_ops_web:unoptimized"], capture_output=True, check=False)
    (fig_dir / "task6_2_6_4_image_analysis.txt").write_text("\n".join(logs), encoding="utf-8")
    print("[Task 6.2 & 6.4] Saved image analysis log to reports/figures/task6_2_6_4_image_analysis.txt")


def run_task6_3_and_6_5_compose_and_troubleshooting(fig_dir: Path) -> None:
    print("[Task 6.3 & 6.5] Starting Docker Compose stack, testing health/DB isolation, and troubleshooting ...")
    logs = [
        "=" * 78,
        "Task 6.3 & 6.5 — Docker Compose Stack, Network Isolation & Troubleshooting Log",
        "Project: 23L0713_A2_Ops | Student: Abdullah Chouhan (23L0713)",
        "=" * 78,
    ]
    logs.append(run_cmd(["docker", "compose", "up", "-d", "--build"]))

    # Wait for both services to become healthy
    for _ in range(18):
        ps_out = subprocess.run(["docker", "compose", "ps"], capture_output=True, text=True).stdout
        if "healthy" in ps_out.lower() and ps_out.lower().count("healthy") >= 2:
            break
        time.sleep(3)

    logs.append(run_cmd(["docker", "compose", "ps"]))

    # Proof that DB port 5432 is closed from host outside
    logs.append(run_cmd(["docker", "compose", "port", "db", "5432"]))
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.5)
    port_open = s.connect_ex(("127.0.0.1", 5432)) == 0
    s.close()
    logs.append(f"$ python socket check 127.0.0.1:5432 from host -> Port 5432 reachable from outside? {port_open} (Expected: False)\n")

    # Query /health and submit 2 test images into Compose stack
    try:
        health_resp = requests.get("http://127.0.0.1:5000/health", timeout=10)
        logs.append(f"$ GET http://127.0.0.1:5000/health (HTTP {health_resp.status_code})\n{json.dumps(health_resp.json(), indent=2)}\n")

        test_df = pd.read_csv("data/splits/test.csv")
        for idx in range(2):
            row = test_df.iloc[idx]
            p = Path(row["file_path"])
            with p.open("rb") as f:
                pred_r = requests.post("http://127.0.0.1:5000/api/predict", files={"file": (p.name, f, "image/jpeg")}, timeout=20)
            logs.append(f"$ POST http://127.0.0.1:5000/api/predict ({p.name}) -> HTTP {pred_r.status_code}\n{json.dumps(pred_r.json(), indent=2)}\n")

        hist_r = requests.get("http://127.0.0.1:5000/history?format=json", timeout=10)
        logs.append(f"$ GET http://127.0.0.1:5000/history?format=json -> HTTP {hist_r.status_code}\n{json.dumps(hist_r.json(), indent=2)}\n")
    except Exception as exc:
        logs.append(f"Compose HTTP verification note: {exc}\n")

    # Task 6.5: Demonstrate shell inside each service, stop, start, restart
    logs.append("\n--- [Task 6.5] Shell inside web and db containers + start/stop/restart ---")
    logs.append(run_cmd(["docker", "compose", "exec", "-T", "web", "sh", "-c", "whoami && id && ls -lh /app/models"]))
    logs.append(
        run_cmd(
            [
                "docker",
                "compose",
                "exec",
                "-T",
                "db",
                "psql",
                "-U",
                "visionguard_user",
                "-d",
                "visionguard_db",
                "-c",
                "SELECT id, filename, verdict, confidence, model_version, created_at FROM prediction_logs;",
            ]
        )
    )
    logs.append(run_cmd(["docker", "compose", "restart", "web"]))
    logs.append(run_cmd(["docker", "compose", "stop", "web"]))
    logs.append(run_cmd(["docker", "compose", "start", "web"]))

    # Task 6.5 Intentional Break -> Diagnose from logs -> Fix
    logs.append("\n--- [Task 6.5] Intentional Failure Simulation: Broken Database Password ---")
    env_path = Path(".env")
    orig_env = env_path.read_text(encoding="utf-8")
    try:
        # Temporarily point web to a wrong DB password while pgdata volume retains original password
        broken_env = orig_env.replace("POSTGRES_PASSWORD=vg_secure_pass_23L0713", "POSTGRES_PASSWORD=WRONG_BROKEN_PASSWORD_999")
        env_path.write_text(broken_env, encoding="utf-8")
        logs.append(run_cmd(["docker", "compose", "up", "-d", "web"]))
        time.sleep(3)
        try:
            bad_h = requests.get("http://127.0.0.1:5000/health", timeout=8)
            logs.append(f"$ GET http://127.0.0.1:5000/health during failure -> HTTP {bad_h.status_code}\n{json.dumps(bad_h.json(), indent=2)}\n")
        except Exception as e_bad:
            logs.append(f"Health check during failure: {e_bad}\n")
        logs.append(run_cmd(["docker", "compose", "logs", "--tail", "25", "web"]))
        logs.append(run_cmd(["docker", "compose", "logs", "--tail", "15", "db"]))
    finally:
        # Restore valid .env and recover web container
        env_path.write_text(orig_env, encoding="utf-8")
        logs.append("\n--- [Task 6.5] Fix Applied: Restored POSTGRES_PASSWORD in .env and recreated web service ---")
        logs.append(run_cmd(["docker", "compose", "up", "-d", "web"]))
        time.sleep(4)
        try:
            good_h = requests.get("http://127.0.0.1:5000/health", timeout=8)
            logs.append(f"$ GET http://127.0.0.1:5000/health after recovery -> HTTP {good_h.status_code}\n{json.dumps(good_h.json(), indent=2)}\n")
        except Exception:
            pass

    (fig_dir / "task6_3_6_5_compose_and_troubleshooting.txt").write_text("\n".join(logs), encoding="utf-8")
    print("[Task 6.3 & 6.5] Saved Compose & troubleshooting log to reports/figures/task6_3_6_5_compose_and_troubleshooting.txt")


if __name__ == "__main__":
    out = Path("reports/figures")
    out.mkdir(parents=True, exist_ok=True)
    run_task6_1_warmup(out)
    run_task6_2_and_6_4_image_analysis(out)
    run_task6_3_and_6_5_compose_and_troubleshooting(out)
