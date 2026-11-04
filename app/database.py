"""Task 5: SQLAlchemy Database Trail for VisionGuard (23L0713_A2_Ops).

Supports PostgreSQL (in Docker Compose / Cloud) and SQLite (for local development/testing).
Every prediction logs: what was decided, when, by which model, and how confidently.
"""

import datetime
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), nullable=False)
    filename = Column(String(255), nullable=False)
    file_size_kb = Column(Float, nullable=False)
    image_dimensions = Column(String(64), nullable=False)
    verdict = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=False)
    prob_ai = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    model_version = Column(String(128), nullable=False)
    inference_source = Column(String(64), nullable=False, default="local_model")
    latency_ms = Column(Float, nullable=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.created_at else "",
            "filename": self.filename,
            "file_size_kb": round(self.file_size_kb, 2),
            "image_dimensions": self.image_dimensions,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 4),
            "confidence_pct": round(self.confidence * 100.0, 2),
            "prob_ai": round(self.prob_ai, 4),
            "threshold": round(self.threshold, 2),
            "model_version": self.model_version,
            "inference_source": self.inference_source,
            "latency_ms": round(self.latency_ms, 2),
        }


class DatabaseManager:
    def __init__(self, database_url: str) -> None:
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self.database_url = database_url
        self.engine = create_engine(database_url, connect_args=connect_args, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)

    def init_db(self) -> None:
        Base.metadata.create_all(bind=self.engine)

    def check_health(self) -> tuple[bool, str]:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True, "connected"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def log_prediction(
        self,
        filename: str,
        file_size_kb: float,
        image_dimensions: str,
        verdict: str,
        confidence: float,
        prob_ai: float,
        threshold: float,
        model_version: str,
        inference_source: str,
        latency_ms: float,
    ) -> dict:
        with self.SessionLocal() as session:
            entry = PredictionLog(
                filename=filename,
                file_size_kb=file_size_kb,
                image_dimensions=image_dimensions,
                verdict=verdict,
                confidence=confidence,
                prob_ai=prob_ai,
                threshold=threshold,
                model_version=model_version,
                inference_source=inference_source,
                latency_ms=latency_ms,
            )
            session.add(entry)
            session.commit()
            session.refresh(entry)
            return entry.to_dict()

    def get_recent_logs(self, limit: int = 100) -> list[dict]:
        with self.SessionLocal() as session:
            rows = (
                session.query(PredictionLog)
                .order_by(PredictionLog.created_at.desc(), PredictionLog.id.desc())
                .limit(limit)
                .all()
            )
            return [r.to_dict() for r in rows]
