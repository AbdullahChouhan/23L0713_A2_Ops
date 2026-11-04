"""Download the Kaggle dataset 'cashbowman/ai-generated-images-vs-real-images' into data/raw."""

import shutil
from pathlib import Path

import kagglehub


def download_and_prepare(target_dir: Path = Path("data/raw")) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    ai_dir = target_dir / "AiArtData"
    real_dir = target_dir / "RealArt"

    if ai_dir.exists() and real_dir.exists() and any(ai_dir.iterdir()) and any(real_dir.iterdir()):
        print(f"[INFO] Dataset already present in {target_dir.resolve()}")
        return target_dir

    print("[INFO] Downloading dataset via kagglehub: cashbowman/ai-generated-images-vs-real-images ...")
    cache_path = Path(kagglehub.dataset_download("cashbowman/ai-generated-images-vs-real-images"))
    print(f"[INFO] Downloaded to cache: {cache_path}")

    # Copy class folders into data/raw
    for sub in cache_path.rglob("*"):
        if sub.is_dir() and sub.name in ("AiArtData", "RealArt"):
            dest = target_dir / sub.name
            if not dest.exists():
                print(f"[INFO] Copying {sub} -> {dest}")
                shutil.copytree(sub, dest)

    print(f"[INFO] Dataset ready at {target_dir.resolve()}")
    return target_dir


if __name__ == "__main__":
    download_and_prepare()
