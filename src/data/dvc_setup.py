"""Task 1.4: DVC (Data Version Control) — Toy Lab & Project Dataset Tracking (23L0713_A2_Ops).

1. Toy Lab (in isolated folder `toy_labs/dvc_toy_work`):
   - Initializes Git and DVC
   - Creates `samples.csv` (v1: 3 rows), tracks with `dvc add samples.csv`, commits v1 in Git
   - Modifies `samples.csv` (v2: 6 rows), runs `dvc add samples.csv`, commits v2 in Git
   - Configures remote storage (both DagsHub S3 remote and local remote store), pushes data (`dvc push`)
   - Returns to older version (`git checkout HEAD~1 samples.csv.dvc && dvc checkout`), verifies v1 content
   - Returns to v2 (`git checkout main && dvc checkout`)
   - Deletes local `samples.csv` and `.dvc/cache`, then restores via `dvc pull`
2. Project Dataset Tracking:
   - Initializes DVC in project root, tracks `data/raw` (`data/raw.dvc`), configures DagsHub remote.
"""

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DVC_CMD = [str(Path(".venv/Scripts/python.exe").resolve()), "-m", "dvc"]


def _safe_rmtree(path: Path) -> None:
    if not path.exists():
        return

    def _on_rm_error(func, p, _exc_info):
        os.chmod(p, stat.S_IWRITE)
        func(p)

    shutil.rmtree(path, onerror=_on_rm_error)


def run_cmd(cmd: list[str], cwd: Path, log_lines: list[str]) -> str:
    cmd_str = " ".join(cmd)
    log_lines.append(f"\n$ {cmd_str}")
    res = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    out = (res.stdout + "\n" + res.stderr).strip()
    log_lines.append(out if out else "(exit 0, no output)")
    return out


def run_toy_dvc_lab() -> None:
    base_dir = Path("toy_labs")
    work_dir = base_dir / "dvc_toy_work"
    remote_dir = base_dir / "dvc_local_remote"

    _safe_rmtree(work_dir)
    _safe_rmtree(remote_dir)

    work_dir.mkdir(parents=True, exist_ok=True)
    remote_dir.mkdir(parents=True, exist_ok=True)

    log_lines = [
        "=" * 78,
        "Task 1.4 — DVC Toy Lab Execution Log (Project: 23L0713_A2_Ops)",
        "Student: Abdullah Chouhan (23L0713)",
        "=" * 78,
    ]

    run_cmd(["git", "init", "-b", "main"], work_dir, log_lines)
    run_cmd(["git", "config", "user.name", "Abdullah Chouhan"], work_dir, log_lines)
    run_cmd(["git", "config", "user.email", "l230713@lhr.nu.edu.pk"], work_dir, log_lines)
    run_cmd([*DVC_CMD, "init"], work_dir, log_lines)
    run_cmd(["git", "add", ".dvc", ".dvcignore"], work_dir, log_lines)
    run_cmd(["git", "commit", "-m", "Initialize DVC in toy lab (23L0713_A2_Ops)"], work_dir, log_lines)

    # Configure local remote and DagsHub S3-compatible remote
    run_cmd([*DVC_CMD, "remote", "add", "-d", "localstore", str(remote_dir.resolve())], work_dir, log_lines)
    dagshub_s3_endpoint = "https://dagshub.com/AbdullahChouhan/23L0713_A2_Ops.s3"
    run_cmd([*DVC_CMD, "remote", "add", "origin", "s3://dvc"], work_dir, log_lines)
    run_cmd([*DVC_CMD, "remote", "modify", "origin", "endpointurl", dagshub_s3_endpoint], work_dir, log_lines)

    # Version 1 of CSV
    csv_file = work_dir / "samples.csv"
    v1_content = "image_id,class_label,width,height\nimg_001.jpg,AI,512,512\nimg_002.jpg,Real,800,600\nimg_003.jpg,AI,1024,1024\n"
    csv_file.write_text(v1_content, encoding="utf-8")
    log_lines.append("\n--- [Step 1] Created samples.csv (v1: 3 rows) ---")
    log_lines.append(v1_content.strip())

    run_cmd([*DVC_CMD, "add", "samples.csv"], work_dir, log_lines)
    log_lines.append("\n--- [Pointer File v1: samples.csv.dvc] ---")
    if (work_dir / "samples.csv.dvc").exists():
        log_lines.append((work_dir / "samples.csv.dvc").read_text(encoding="utf-8").strip())

    run_cmd(["git", "add", "samples.csv.dvc", ".gitignore", ".dvc/config"], work_dir, log_lines)
    run_cmd(["git", "commit", "-m", "Track samples.csv v1 (3 rows) with DVC"], work_dir, log_lines)
    run_cmd(["git", "tag", "v1.0"], work_dir, log_lines)

    # Version 2 of CSV
    v2_content = (
        v1_content
        + "img_004.jpg,Real,640,480\nimg_005.jpg,AI,768,768\nimg_006.jpg,Real,1200,900\n"
    )
    csv_file.write_text(v2_content, encoding="utf-8")
    log_lines.append("\n--- [Step 2] Modified samples.csv (v2: 6 rows) ---")
    log_lines.append(v2_content.strip())

    run_cmd([*DVC_CMD, "add", "samples.csv"], work_dir, log_lines)
    log_lines.append("\n--- [Pointer File v2: samples.csv.dvc] ---")
    if (work_dir / "samples.csv.dvc").exists():
        log_lines.append((work_dir / "samples.csv.dvc").read_text(encoding="utf-8").strip())

    run_cmd(["git", "add", "samples.csv.dvc"], work_dir, log_lines)
    run_cmd(["git", "commit", "-m", "Update samples.csv to v2 (6 rows) with DVC"], work_dir, log_lines)
    run_cmd(["git", "tag", "v2.0"], work_dir, log_lines)

    # Push both versions to remote storage
    run_cmd([*DVC_CMD, "push", "-r", "localstore"], work_dir, log_lines)
    token = os.getenv("DAGSHUB_TOKEN", "")
    if token:
        subprocess.run([*DVC_CMD, "remote", "modify", "--local", "origin", "access_key_id", token], cwd=str(work_dir), check=False)
        subprocess.run([*DVC_CMD, "remote", "modify", "--local", "origin", "secret_access_key", token], cwd=str(work_dir), check=False)
        run_cmd([*DVC_CMD, "push", "-r", "origin"], work_dir, log_lines)

    # Return to older version v1
    log_lines.append("\n--- [Step 3] Returning to older version (v1.0) via Git + DVC checkout ---")
    run_cmd(["git", "checkout", "v1.0", "samples.csv.dvc"], work_dir, log_lines)
    run_cmd([*DVC_CMD, "checkout"], work_dir, log_lines)
    log_lines.append("--- Content of samples.csv after checking out v1.0 ---")
    log_lines.append(csv_file.read_text(encoding="utf-8").strip())

    # Restore v2 pointer, delete local data & cache, and pull back from remote
    log_lines.append("\n--- [Step 4] Restoring v2.0 pointer, deleting local data & cache, and running dvc pull ---")
    run_cmd(["git", "checkout", "main", "samples.csv.dvc"], work_dir, log_lines)
    if csv_file.exists():
        csv_file.unlink()
    _safe_rmtree(work_dir / ".dvc" / "cache")
    log_lines.append(f"samples.csv exists after deletion? {csv_file.exists()}")
    run_cmd([*DVC_CMD, "pull", "-r", "localstore"], work_dir, log_lines)
    log_lines.append(f"samples.csv exists after dvc pull? {csv_file.exists()}")
    log_lines.append("--- Content of samples.csv after dvc pull ---")
    if csv_file.exists():
        log_lines.append(csv_file.read_text(encoding="utf-8").strip())

    run_cmd(["git", "log", "--oneline", "--decorate"], work_dir, log_lines)

    out_path = Path("reports/figures/task1_4_dvc_toy_log.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"[INFO] Saved DVC toy lab log to {out_path}")


def setup_project_dvc() -> None:
    root = Path(".")
    log_lines = [
        "=" * 78,
        "Task 1.4 — Project Dataset DVC Tracking Log (23L0713_A2_Ops)",
        "=" * 78,
    ]
    if not (root / ".git").exists():
        run_cmd(["git", "init", "-b", "main"], root, log_lines)
        run_cmd(["git", "config", "user.name", "Abdullah Chouhan"], root, log_lines)
        run_cmd(["git", "config", "user.email", "l230713@lhr.nu.edu.pk"], root, log_lines)

    if not (root / ".dvc").exists():
        run_cmd([*DVC_CMD, "init"], root, log_lines)

    dagshub_s3_endpoint = "https://dagshub.com/AbdullahChouhan/23L0713_A2_Ops.s3"
    run_cmd([*DVC_CMD, "remote", "add", "-f", "-d", "origin", "s3://dvc"], root, log_lines)
    run_cmd([*DVC_CMD, "remote", "modify", "origin", "endpointurl", dagshub_s3_endpoint], root, log_lines)

    local_dvc_store = Path("toy_labs/project_dvc_remote").resolve()
    local_dvc_store.mkdir(parents=True, exist_ok=True)
    run_cmd([*DVC_CMD, "remote", "add", "-f", "localstore", str(local_dvc_store)], root, log_lines)

    token = os.getenv("DAGSHUB_TOKEN", "")
    if token:
        subprocess.run([*DVC_CMD, "remote", "modify", "--local", "origin", "access_key_id", token], cwd=str(root), check=False)
        subprocess.run([*DVC_CMD, "remote", "modify", "--local", "origin", "secret_access_key", token], cwd=str(root), check=False)

    if (root / "data" / "raw").exists():
        run_cmd([*DVC_CMD, "add", "data/raw"], root, log_lines)
        if (root / "data" / "raw.dvc").exists():
            pointer_text = (root / "data" / "raw.dvc").read_text(encoding="utf-8")
            log_lines.append("\n--- [Contents of data/raw.dvc pointer file] ---")
            log_lines.append(pointer_text.strip())

    out_path = Path("reports/figures/task1_4_project_dvc_log.txt")
    out_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"[INFO] Project DVC tracking complete. Pointer file saved at data/raw.dvc")


if __name__ == "__main__":
    run_toy_dvc_lab()
    setup_project_dvc()
