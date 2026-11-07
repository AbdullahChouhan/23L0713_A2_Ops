"""Task 6.6: Push Red (Failing) and Green (Passing) commits to https://github.com/AbdullahChouhan/23L0713_A2_ToyCI."""

import os
import shutil
import stat
import subprocess
import time
from pathlib import Path


def _safe_rmtree(path: Path) -> None:
    if not path.exists():
        return

    def _on_rm_error(func, p, _exc_info):
        os.chmod(p, stat.S_IWRITE)
        func(p)

    shutil.rmtree(path, onerror=_on_rm_error)


def run_cmd(cmd: list[str], cwd: Path, logs: list[str]) -> str:
    res = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    out = ((res.stdout or "") + "\n" + (res.stderr or "")).strip()
    logs.append(f"$ {' '.join(cmd)}\n{out if out else '(exit 0)'}")
    return out


def main() -> None:
    repo_url = "https://github.com/AbdullahChouhan/23L0713_A2_ToyCI.git"
    work_dir = Path("toy_labs/toy_ci_repo_work")
    _safe_rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    wf_dir = work_dir / ".github" / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)

    # Copy matrix workflow file
    src_wf = Path("toy_labs/github_actions_toy/.github/workflows/toy_ci.yml")
    (wf_dir / "toy_ci.yml").write_text(src_wf.read_text(encoding="utf-8"), encoding="utf-8")
    (work_dir / "README.md").write_text(
        "# 23L0713_A2_ToyCI — Task 6.6 GitHub Actions Matrix Toy Lab\n\n"
        "[![Toy Matrix CI (23L0713_A2_Ops)](https://github.com/AbdullahChouhan/23L0713_A2_ToyCI/actions/workflows/toy_ci.yml/badge.svg)](https://github.com/AbdullahChouhan/23L0713_A2_ToyCI/actions/workflows/toy_ci.yml)\n\n"
        "Student: Abdullah Chouhan (`23L0713`) | Main Project: `23L0713_A2_Ops`\n",
        encoding="utf-8",
    )

    logs = [
        "=" * 78,
        "Task 6.6 — GitHub Actions Toy CI Repository Execution Log (23L0713_A2_ToyCI)",
        f"Remote Repository: {repo_url}",
        "=" * 78,
    ]

    run_cmd(["git", "init", "-b", "main"], work_dir, logs)
    run_cmd(["git", "config", "user.name", "Abdullah Chouhan"], work_dir, logs)
    run_cmd(["git", "config", "user.email", "l230713@lhr.nu.edu.pk"], work_dir, logs)
    run_cmd(["git", "remote", "add", "origin", repo_url], work_dir, logs)

    # Commit 1: Intentionally Failing Test (Red Run)
    failing_test = (
        '"""Task 6.6 Toy CI Test File (Commit 1: Intentional Failure for Red Run)."""\n\n\n'
        "def normalize_confidence(prob_ai: float, threshold: float = 0.50) -> tuple[str, float]:\n"
        "    if prob_ai >= threshold:\n"
        '        return "AI-Generated", round(prob_ai, 4)\n'
        '    return "Real", round(1.0 - prob_ai, 4)\n\n\n'
        "def test_ai_prediction():\n"
        "    label, conf = normalize_confidence(0.85, threshold=0.50)\n"
        '    assert label == "AI-Generated"\n'
        "    # Intentional wrong assertion (0.99 instead of 0.85) to trigger RED CI run:\n"
        "    assert conf == 0.99\n"
    )
    (work_dir / "test_toy.py").write_text(failing_test, encoding="utf-8")
    run_cmd(["git", "add", "."], work_dir, logs)
    run_cmd(["git", "commit", "-m", "Commit 1 (Red Run): Add matrix CI workflow and intentionally failing test"], work_dir, logs)
    run_cmd(["git", "push", "-u", "-f", "origin", "main"], work_dir, logs)

    print("[Task 6.6] Pushed Commit 1 (Red Run). Waiting 6 seconds before pushing Commit 2 (Green Run)...")
    time.sleep(6)

    # Commit 2: Fixed Passing Test (Green Run)
    passing_test = Path("toy_labs/github_actions_toy/test_toy.py").read_text(encoding="utf-8")
    (work_dir / "test_toy.py").write_text(passing_test, encoding="utf-8")
    run_cmd(["git", "add", "test_toy.py"], work_dir, logs)
    run_cmd(["git", "commit", "-m", "Commit 2 (Green Run): Fix confidence assertion in test_toy.py"], work_dir, logs)
    run_cmd(["git", "push", "origin", "main"], work_dir, logs)
    run_cmd(["git", "log", "--oneline"], work_dir, logs)

    _safe_rmtree(work_dir)

    out_path = Path("reports/figures/task6_6_toy_ci_log.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(logs) + "\n", encoding="utf-8")
    print(f"[Task 6.6] Saved Toy CI execution log to {out_path}")


if __name__ == "__main__":
    main()
