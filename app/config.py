"""Task 4.4: Environment-driven configuration for VisionGuard (23L0713_A2_Ops)."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    project_name: str = os.getenv("PROJECT_NAME", "23L0713_A2_Ops")
    student_name: str = "Abdullah Chouhan"
    roll_number: str = "23L0713"
    model_path: str = os.getenv("MODEL_PATH", "models/champion_model.keras")
    model_version: str = os.getenv("MODEL_VERSION", "VisionGuard-MobileNetV2-v2@champion")
    decision_threshold: float = float(os.getenv("DECISION_THRESHOLD", "0.50"))
    max_content_length_mb: float = float(os.getenv("MAX_CONTENT_LENGTH_MB", "5.0"))
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///visionguard_history.db")
    flask_debug: bool = os.getenv("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")
    port: int = int(os.getenv("PORT", "5000"))
    cloud_api_url: str = os.getenv("CLOUD_API_URL", "").strip()
    cloud_api_key: str = os.getenv("CLOUD_API_KEY", "vg_api_secret_23L0713").strip()
    use_cloud_api: bool = os.getenv("USE_CLOUD_API", "false").lower() in ("1", "true", "yes")

    @property
    def max_content_length_bytes(self) -> int:
        return int(self.max_content_length_mb * 1024 * 1024)
