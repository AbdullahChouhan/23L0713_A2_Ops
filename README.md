# VisionGuard — AI vs Real Image Classification (`23L0713_A2_Ops`)

[![VisionGuard CI (23L0713_A2_Ops)](https://github.com/AbdullahChouhan/23L0713_A2_Ops/actions/workflows/ci.yml/badge.svg)](https://github.com/AbdullahChouhan/23L0713_A2_Ops/actions/workflows/ci.yml)
[![DagsHub Repository](https://img.shields.io/badge/DagsHub-23L0713__A2__Ops-orange)](https://dagshub.com/AbdullahChouhan/23L0713_A2_Ops)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.18](https://img.shields.io/badge/TensorFlow-2.18.0-FF6F00.svg)](https://www.tensorflow.org/)

**Student**: Abdullah Chouhan  
**Roll Number**: `23L0713`  
**Project Name**: `23L0713_A2_Ops`  
**GitHub Repository**: [https://github.com/AbdullahChouhan/23L0713_A2_Ops](https://github.com/AbdullahChouhan/23L0713_A2_Ops)  
**Toy CI Repository (Task 6.6)**: [https://github.com/AbdullahChouhan/23L0713_A2_ToyCI](https://github.com/AbdullahChouhan/23L0713_A2_ToyCI)  
**DagsHub Repository**: [https://dagshub.com/AbdullahChouhan/23L0713_A2_Ops](https://dagshub.com/AbdullahChouhan/23L0713_A2_Ops)  
**Live Cloud API & Web Service (Render)**: [https://two3l0713-a2-ops.onrender.com](https://two3l0713-a2-ops.onrender.com)  
**Live Health Endpoint**: [https://two3l0713-a2-ops.onrender.com/health](https://two3l0713-a2-ops.onrender.com/health)

---

## Overview

**VisionGuard (`23L0713_A2_Ops`)** is an end-to-end MLOps system that detects whether an uploaded image is **AI-Generated** or **Real** (`RealArt` vs `AiArtData`). It encompasses:

1. **Data Understanding & Versioning (Task 1)**: Automated SHA-256 dataset audit (`src/data/audit_dataset.py`), corruption & duplicate detection, visual EDA (`src/data/visual_eda.py`), and DVC tracking (`data/raw.dvc`) backed by DagsHub storage.
2. **Model Development with TensorFlow/Keras (Task 2)**: Batched & prefetched `tf.data` input pipeline (`src/models/data_pipeline.py`), a custom 3-block Baseline CNN (`src/models/architectures.py`), two-stage transfer learning with `MobileNetV2`, validation-tuned decision thresholding, error analysis on 10 misclassified test samples, Keras Tuner hyperparameter optimization (`src/models/keras_tuner_lab.py`), and Grad-CAM visual explanations (`src/models/train_and_eval.py`).
3. **Experiment Tracking & Model Registry with MLflow & DagsHub (Task 3)**: Manual callback logging vs TensorFlow autologging, multi-run comparison across 6+ architectural/hyperparameter changes, artifact tracking, and model versioning with `@champion` alias resolution (`src/tracking/mlflow_experiments.py`, `src/tracking/load_champion_registry.py`).
4. **Flask Web Application & Database Trail (Task 4 & 5)**: Singleton model loading at startup (`app/inference.py`), strict in-memory upload validation (`.jpg`/`.png`/`.webp`/`.bmp`, size caps, Pillow stream verification, `secure_filename`), honest model caveats, and permanent PostgreSQL/SQLite request logging (`app/database.py`, `app/main.py`).
5. **Containerization & CI/CD (Task 6)**: Slim `python:3.11-slim` + `tensorflow-cpu` `Dockerfile` running as non-root `appuser` with container `HEALTHCHECK`, multi-container `docker-compose.yml` (`web` + isolated `postgres:16-alpine` `db` with named volume `pgdata`), and automated GitHub Actions CI (`.github/workflows/ci.yml`).
6. **Cloud API Deployment & Local Flask Integration (Task 7 & 8)**: Public REST API endpoint (`POST /api/predict`) callable from Postman and integrated with the local Flask web UI (`USE_CLOUD_API=true`).

---

## Quick Start

### 1. Local Environment Setup
```bash
uv venv --python 3.11 .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
cp .env.example .env
```

### 2. Data Audit, Training & MLflow Tracking
```bash
python -m src.data.env_check
python -m src.data.download_dataset
python -m src.data.audit_dataset
python -m src.data.visual_eda
python -m src.models.train_and_eval
python -m src.models.keras_tuner_lab
python -m src.tracking.mlflow_experiments
python -m src.tracking.load_champion_registry
```

### 3. Run Production Flask Server Locally
```bash
python -m app.main
```

### 4. Run Full Stack with Docker Compose
```bash
docker compose up --build -d
docker compose ps
```

### 5. Run Linter & Automated Tests
```bash
ruff check app/ tests/
pytest tests/ -v
```
