"""Generate the complete PDF report '23L0713_VisionGuard.pdf' for Assignment 2 (23L0713_A2_Ops)."""

import datetime
import json
import os
import subprocess
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fpdf import FPDF

load_dotenv()


def sanitize_text(text: str) -> str:
    """Ensure text uses Latin-1 compatible glyphs for standard PDF core fonts."""
    replacements = {
        "—": "--",
        "–": "-",
        "•": "*",
        "→": "->",
        "←": "<-",
        "≥": ">=",
        "≤": "<=",
        "×": "x",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "⚠️": "[!]",
        "🖼️": "[IMG]",
        "🏃": "[RUN]",
        "🧪": "[EXP]",
        "\u202f": " ",
        "\u00a0": " ",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", errors="replace").decode("latin-1")


class VisionGuardReportPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(90, 100, 115)
        self.cell(
            0,
            6,
            sanitize_text("VisionGuard MLOps Report | Project: 23L0713_A2_Ops | Abdullah Chouhan (23L0713)"),
            align="L",
        )
        self.ln(8)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 110, 125)
        self.cell(0, 8, sanitize_text(f"Page {self.page_no()} of {{nb}} — 23L0713_VisionGuard.pdf"), align="C")

    def chapter_title(self, title: str):
        self.ln(3)
        self.set_fill_color(15, 23, 42)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 13)
        self.cell(0, 9, sanitize_text(f"  {title}"), fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)
        self.set_text_color(20, 25, 35)

    def subtask_heading(self, heading: str):
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(29, 78, 216)
        self.cell(0, 7, sanitize_text(heading), new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(20, 25, 35)

    def body_paragraph(self, label: str, text: str):
        self.set_font("Helvetica", "B", 9.5)
        self.write(5, sanitize_text(f"{label}: "))
        self.set_font("Helvetica", "", 9.5)
        self.write(5, sanitize_text(text) + "\n")
        self.ln(1.5)

    def code_block(self, title: str, code_text: str, max_lines: int = 28):
        lines = code_text.strip().splitlines()
        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f"... ({len(code_text.splitlines()) - max_lines} more lines omitted for brevity)"]
        body = "\n".join(lines)
        self.set_font("Helvetica", "B", 8.5)
        self.set_text_color(51, 65, 85)
        self.cell(0, 5, sanitize_text(title), new_x="LMARGIN", new_y="NEXT")
        self.set_fill_color(241, 245, 249)
        self.set_text_color(15, 23, 42)
        self.set_font("Courier", "", 7.5)
        self.multi_cell(0, 4.0, sanitize_text(body), border=1, fill=True)
        self.ln(2)

    def add_figure(self, img_path: Path, caption: str, width: float = 175.0):
        if not img_path.exists():
            return
        if self.get_y() > 205:
            self.add_page()
        x_pos = (210.0 - width) / 2.0
        self.image(str(img_path), x=x_pos, w=width)
        self.ln(1.5)
        self.set_font("Helvetica", "I", 8.5)
        self.set_text_color(71, 85, 105)
        self.multi_cell(0, 4.5, sanitize_text(caption), align="C")
        self.set_text_color(20, 25, 35)
        self.ln(2.5)

    def add_dataframe_table(self, df: pd.DataFrame, caption: str):
        if df.empty:
            return
        self.set_font("Helvetica", "B", 8.5)
        self.set_text_color(30, 41, 59)
        self.cell(0, 5.5, sanitize_text(caption), new_x="LMARGIN", new_y="NEXT")

        cols = list(df.columns)
        col_w = 190.0 / len(cols)
        self.set_fill_color(226, 232, 240)
        self.set_font("Helvetica", "B", 7.2)
        for c in cols:
            self.cell(col_w, 5.5, sanitize_text(str(c)[:20]), border=1, fill=True, align="C")
        self.ln()

        self.set_font("Helvetica", "", 7.0)
        for _, row in df.iterrows():
            for c in cols:
                val = str(row[c])
                self.cell(col_w, 5.0, sanitize_text(val[:22]), border=1, align="C")
            self.ln()
        self.ln(2.5)


def get_git_commit_full() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        return res.stdout.strip() or "initial-commit"
    except Exception:
        return "initial-commit"


def build_pdf_report(output_pdf: Path = Path("23L0713_VisionGuard.pdf")) -> None:
    fig_dir = Path("reports/figures")
    pdf = VisionGuardReportPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Front Matter / Cover Page
    pdf.add_page()
    pdf.set_fill_color(15, 23, 42)
    pdf.rect(10, 12, 190, 42, style="F")
    pdf.set_y(19)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 9, "VisionGuard: AI vs. Real Image Classification", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11.5)
    pdf.cell(0, 7, "End-to-End MLOps Pipeline, Experiment Tracking, Containerization & Cloud Deployment", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 10.5)
    pdf.cell(0, 7, "Project Identifier: 23L0713_A2_Ops | FAST-NUCES MLOps Assignment 2", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_y(60)
    pdf.set_text_color(20, 25, 35)
    commit_hash = get_git_commit_full()
    cloud_url = os.getenv("CLOUD_API_URL") or "https://two3l0713-a2-ops.onrender.com"

    front_matter_df = pd.DataFrame(
        [
            {"Field": "Student Name", "Value": "Abdullah Chouhan"},
            {"Field": "Roll Number", "Value": "23L0713"},
            {"Field": "Project Name (Everywhere)", "Value": "23L0713_A2_Ops"},
            {"Field": "GitHub Repository", "Value": "https://github.com/AbdullahChouhan/23L0713_A2_Ops"},
            {"Field": "DagsHub Repository", "Value": "https://dagshub.com/AbdullahChouhan/23L0713_A2_Ops"},
            {"Field": "MLflow Tracking URI", "Value": "https://dagshub.com/AbdullahChouhan/23L0713_A2_Ops.mlflow"},
            {"Field": "Live Cloud API URL", "Value": cloud_url},
            {"Field": "Graded Git Commit Hash", "Value": commit_hash[:16]},
            {"Field": "Generated Timestamp", "Value": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S PKT")},
        ]
    )
    pdf.add_dataframe_table(front_matter_df, "Front Matter — Student, Repository & Deployment Metadata")

    # CHAPTER 1: TASK 1
    pdf.chapter_title("Chapter 1 — Task 1: Data Understanding and Versioning (7 Marks)")

    pdf.subtask_heading("Task 1.1: Isolated Python Environment & Hardware Verification")
    pdf.body_paragraph(
        "What We Did",
        "Created an isolated Python 3.11.18 virtual environment (.venv) using uv (uv venv --python 3.11 .venv) and installed pinned versions of TensorFlow (2.18.0), Keras (3.15.1), Keras-Tuner (1.4.7), MLflow (2.19.0), DagsHub (0.5.9), DVC (3.58.0) with dvc-s3 (3.2.0), Flask (3.1.0), SQLAlchemy (2.0.36), Pillow (11.0.0), Scikit-Learn (1.6.0), and Pytest (8.3.4).",
    )
    pdf.body_paragraph(
        "Why & How It Works",
        "Isolating dependencies in a virtual environment prevents version collisions across host packages and guarantees deterministic builds between local development, GitHub Actions CI, and Docker containers. We wrote src/data/env_check.py using importlib.metadata and tf.config.list_physical_devices('GPU') to record exact versions and verify hardware acceleration.",
    )
    pdf.body_paragraph(
        "What Went Wrong & Fix",
        "The host system defaulted to Python 3.14.7, for which precompiled TensorFlow 2.18 wheels do not yet exist (pip index versions tensorflow returned no distribution). We resolved this cleanly by provisioning a dedicated CPython 3.11.18 environment via uv venv --python 3.11 .venv.",
    )
    env_txt = fig_dir / "task1_1_env_versions.txt"
    if env_txt.exists():
        pdf.code_block("Terminal Evidence (Task 1.1): python -m src.data.env_check", env_txt.read_text(encoding="utf-8"))

    pdf.subtask_heading("Task 1.2: Dataset Audit, SHA-256 Deduplication & Corrupt File Detection")
    pdf.body_paragraph(
        "What We Did",
        "Developed src/data/audit_dataset.py to walk data/raw (AiArtData -> AI [1], RealArt -> Real [0]), inspecting all 975 files and recording class, byte size, width, height, colour mode, format, and SHA-256 content digest.",
    )
    pdf.body_paragraph(
        "Why & How It Works",
        "File names can be misleading; only cryptographic content hashing (SHA-256 over 64 KB chunks) reliably detects exact duplicates across or within classes that would otherwise cause train/test data leakage. Similarly, calling both PIL.Image.verify() and img.load() catches zero-byte or truncated streams before training starts.",
    )
    pdf.body_paragraph(
        "Audit Findings & What Went Wrong",
        "Out of 975 total images (539 AI, 436 Real), 0 files were unreadable/corrupt, and 1 exact SHA-256 duplicate pair (2 files in AiArtData) was detected and deduplicated, leaving 974 clean unique images (538 AI, 436 Real). Furthermore, the audit revealed mixed colour modes: RGB (896), RGBA (49), Palette 'P' with byte transparency (23), and Grayscale 'L' (7). We standardized all images to 3-channel RGB during decoding and created stratified 70/15/15 splits (Train=681, Val=146, Test=147, seed=42).",
    )
    audit_csv = fig_dir / "task1_2_audit_summary.csv"
    if audit_csv.exists():
        audit_df = pd.read_csv(audit_csv)[
            [
                "Class",
                "Total Files",
                "Corrupt / Unreadable",
                "Exact Duplicates (Removed)",
                "Clean Unique Images",
                "Width Min (px)",
                "Width Median (px)",
                "Width Max (px)",
                "Height Min (px)",
                "Height Median (px)",
                "Height Max (px)",
                "Size Min (KB)",
                "Size Median (KB)",
                "Size Max (KB)",
            ]
        ]
        pdf.add_dataframe_table(audit_df, "Table 1.2a: Dataset Audit Summary per Class (Counts, Width, Height, File Size)")

    fmt_csv = fig_dir / "task1_2_formats_modes.csv"
    if fmt_csv.exists():
        pdf.add_dataframe_table(pd.read_csv(fmt_csv), "Table 1.2b: Image Format and Colour Mode Breakdown by Class")

    pdf.subtask_heading("Task 1.3: Visual Exploratory Data Analysis (EDA)")
    pdf.body_paragraph(
        "Sample Grid Interpretation",
        "Figure 1.3a displays 5 randomly sampled AI-generated images (top row) and 5 Real Art/photographic images (bottom row). AI images frequently exhibit square 1:1 aspect ratios (e.g., 512x512 or 1024x1024 from Stable Diffusion/Midjourney) and hyper-smooth skin/background gradients, whereas Real images exhibit diverse photographic framing, natural sensor grain, and physical brushwork.",
    )
    pdf.add_figure(
        fig_dir / "task1_3_sample_grid.png",
        "Figure 1.3a: Sample images per class (Top: AI-Generated, Bottom: Real) annotated with original dimensions, format, colour mode, and file size.",
    )
    pdf.body_paragraph(
        "Histograms Interpretation",
        "Figure 1.3b shows the distributions of width, height, and file size (KB) by class. AI images have a higher median width (1024 px vs. 920 px for Real) and higher median height (800 px vs. 720 px), clustering tightly around standard diffusion resolutions, while Real images show a wider right tail reaching up to 8495x7091 px and 9956 KB.",
    )
    pdf.add_figure(
        fig_dir / "task1_3_distributions_hist.png",
        "Figure 1.3b: Histograms and KDE curves of image width, height, and file size (KB) per class with dashed median lines.",
    )

    pdf.subtask_heading("Task 1.4: Data Version Control (DVC) — Toy Lab & Project Dataset Tracking")
    pdf.body_paragraph(
        "What We Did & Why",
        "In toy_labs/dvc_toy_work, we initialized Git and DVC, created samples.csv (v1: 3 rows), tracked it via dvc add samples.csv, committed samples.csv.dvc (tag v1.0), updated samples.csv to v2 (6 rows), committed v2.0, pushed to remote storage, checked out v1.0 pointer + dvc checkout to verify rollback, and finally deleted local data/cache and restored via dvc pull. We then tracked data/raw (975 files, ~481 MB) in the main repository with data/raw.dvc and configured DagsHub S3 remote.",
    )
    pdf.body_paragraph(
        "What the .dvc Pointer File Holds & Why Large Data Does Not Belong in Git",
        "The small YAML pointer file (samples.csv.dvc / data/raw.dvc) stores only lightweight metadata: the content MD5 hash (e.g., 8f9c2...dir), total byte size, number of files (nfiles: 975), and relative path. Large binary datasets do not belong in Git because Git packs every version into .git/objects, causing repository bloat, slow clones, and hitting GitHub's 100 MB file limit. DVC keeps Git lightweight while storing content-addressed blobs in remote object storage (DagsHub S3).",
    )
    dvc_toy_txt = fig_dir / "task1_4_dvc_toy_log.txt"
    if dvc_toy_txt.exists():
        pdf.code_block("Terminal Evidence (Task 1.4a): Toy DVC Lab (v1 -> v2 -> checkout v1 -> delete -> dvc pull)", dvc_toy_txt.read_text(encoding="utf-8"), max_lines=32)
    dvc_proj_txt = fig_dir / "task1_4_project_dvc_log.txt"
    if dvc_proj_txt.exists():
        pdf.code_block("Terminal Evidence (Task 1.4b): Project data/raw.dvc Pointer & Remote Setup", dvc_proj_txt.read_text(encoding="utf-8"), max_lines=22)
    git_push_img = fig_dir / "user_git_push_screenshot.png"
    if git_push_img.exists():
        pdf.add_figure(
            git_push_img,
            "Figure 1.4: Terminal screenshot of initial commit push to https://github.com/AbdullahChouhan/23L0713_A2_Ops.git.",
            width=165.0,
        )

    # CHAPTER 2: TASK 2
    pdf.add_page()
    pdf.chapter_title("Chapter 2 — Task 2: Model Development with TensorFlow/Keras (14 Marks + Bonus)")

    pdf.subtask_heading("Task 2.1: tf.data Input Pipeline & Data Augmentation")
    pdf.body_paragraph(
        "Pipeline Stages, Image Size Choice & Aspect Ratio Impact",
        "Implemented in src/models/data_pipeline.py: (1) tf.data.Dataset.from_tensor_slices reads file paths and binary labels from train/val/test CSVs; (2) full-buffer shuffle(seed=42) on training split; (3) parallel RGB decoding and bilinear resize to 160x160; (4) .cache() in memory after decoding; (5) .batch(32); (6) random augmentation (RandomFlip('horizontal'), RandomRotation(0.08), RandomZoom(+-10%), RandomContrast(0.10)) applied ONLY to training batches; (7) .prefetch(tf.data.AUTOTUNE) overlapping CPU preprocessing with model execution. We chose 160x160 because it is an exact native ImageNet resolution for MobileNetV2 that balances spatial artifact retention with fast CPU training. Direct bilinear resizing warps non-square images to 1:1 aspect ratio, which standardizes input tensors without cropping away border watermarks or background cues.",
    )
    pdf.add_figure(
        fig_dir / "task2_1_augmented_batch.png",
        "Figure 2.1: One augmented training batch (8 samples at 160x160 RGB) showing random horizontal flips, rotations, zoom, and contrast jitter.",
    )

    pdf.subtask_heading("Task 2.2: Baseline CNN Architecture from Scratch")
    pdf.body_paragraph(
        "Architecture, Layer Justification & Loss/Optimizer Choice",
        "Built in src/models/architectures.py (VisionGuard_Baseline_CNN, 102,401 total parameters): Rescaling(1/255) normalizes pixel intensities to [0, 1]; three progressive convolutional blocks (32 -> 64 -> 128 filters, 3x3 kernels) extract edges, textures, and structural patterns; BatchNormalization stabilizes internal covariate shift; ReLU provides non-linearity; MaxPooling2D(2x2) halves spatial dimensions and provides translation invariance; GlobalAveragePooling2D collapses 20x20x128 feature maps into a 128-D vector (avoiding massive Flatten parameters); Dropout (0.175 -> 0.35) prevents co-adaptation; and Dense(1, activation='sigmoid') outputs P(AI). Trained with BinaryCrossentropy loss, Adam(lr=1e-3), and Accuracy, Precision, Recall, and ROC-AUC metrics.",
    )
    base_layers_csv = fig_dir / "task2_2_baseline_layers.csv"
    if base_layers_csv.exists():
        pdf.add_dataframe_table(pd.read_csv(base_layers_csv), "Table 2.2: Baseline CNN Per-Layer Output Shape and Parameter Count")

    pdf.subtask_heading("Task 2.3 & 2.4: Two-Stage Transfer Learning (MobileNetV2) & Training Discipline")
    pdf.body_paragraph(
        "Two-Stage Protocol & Pretrained Preprocessing Justification",
        "We loaded MobileNetV2(include_top=False, weights='imagenet', input_shape=(160,160,3)) preceded by Rescaling(1/127.5, offset=-1.0), which maps [0, 255] to [-1, 1] exactly as expected by mobilenet_v2.preprocess_input. Matching pretraining normalization is mandatory because ImageNet weights are calibrated for zero-centered [-1, 1] activations. In Stage 1 (Epochs 1-6), the MobileNetV2 base was frozen and only the custom classification head (head_bn, head_dense, prob_ai) was trained at lr=1e-3. In Stage 2 (Epochs 7-14), we unfroze the top 25 layers of the last inverted residual blocks (block_14_project through Conv_1, keeping BatchNormalization layers frozen) and fine-tuned at a 20x smaller learning rate (lr=5e-5) so pretrained representations adapted smoothly without catastrophic forgetting.",
    )
    pdf.body_paragraph(
        "Training Discipline & Best Epoch",
        "We fixed global seeds (tf.keras.utils.set_random_seed(42)) and attached EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True), ModelCheckpoint(save_best_only=True), and ReduceLROnPlateau(factor=0.5, patience=2). Baseline CNN peaked at Epoch 1 (val_loss=0.6867), whereas MobileNetV2 Transfer Learning achieved its best validation loss at cumulative Epoch 8 (Stage 2 fine-tuning, val_loss=0.5312).",
    )
    pdf.add_figure(
        fig_dir / "task2_3_transfer_curves.png",
        "Figure 2.3: Two-stage MobileNetV2 training and validation curves showing Stage 1 (Frozen Head) to Stage 2 (Fine-Tuning) transition and Best Epoch (8).",
    )
    pdf.add_figure(
        fig_dir / "task2_4_best_epoch_curves.png",
        "Figure 2.4: MobileNetV2 Loss and Accuracy curves with the optimal checkpoint epoch explicitly marked.",
    )

    pdf.subtask_heading("Task 2.5: Held-Out Test Set Evaluation & Decision Threshold Justification")
    pdf.body_paragraph(
        "Evaluation Results & Which Error Is Worse for VisionGuard",
        "Table 2.5 compares the scratch Baseline CNN against the fine-tuned MobileNetV2 model on the held-out test set (147 images). Fine-tuning MobileNetV2 boosted Test ROC-AUC from 0.5827 to 0.8122 (+22.95 percentage points) and Test Accuracy from 52.38% to 70.75% (F1=0.7456 at T=0.50). For VisionGuard's product mission (protecting publishers and users from synthetic media deception), a False Negative (calling an AI-generated fake photo 'Real') allows synthetic misinformation or copyright-infringing AI art to pass verification undetected, whereas a False Positive (flagging a real image as potentially AI) triggers secondary human review. Lowering the threshold to T=0.25 (maximizing validation F1) increases AI Recall to 88.89% (F1=0.7539), while T=0.50 provides balanced precision (71.59%) and recall (77.78%).",
    )
    test_m_csv = fig_dir / "task2_5_test_metrics.csv"
    if test_m_csv.exists():
        pdf.add_dataframe_table(pd.read_csv(test_m_csv), "Table 2.5: Held-Out Test Set Metrics for Baseline CNN and MobileNetV2")
    pdf.add_figure(
        fig_dir / "task2_5_confusion_matrices.png",
        "Figure 2.5a: Test set Confusion Matrices for Baseline CNN and Fine-Tuned MobileNetV2.",
        width=165.0,
    )
    pdf.add_figure(
        fig_dir / "task2_5_roc_curves.png",
        "Figure 2.5b: Test set ROC Curves comparing Baseline CNN (AUC=0.5827) vs. MobileNetV2 Champion (AUC=0.8122).",
        width=135.0,
    )

    pdf.subtask_heading("Task 2.6: Error Analysis of 10 Misclassified Test Images")
    pdf.body_paragraph(
        "Error Patterns & Hypotheses",
        "We inspected the top 10 highest-confidence misclassifications on the test set (Figure 2.6 and Table 2.6). Three dominant failure modes emerged: (1) Heavy JPEG compression in web-scraped real photographs strips natural high-frequency camera sensor noise, making smooth regions resemble diffusion outputs; (2) Real digital vector illustrations and abstract paintings lack photographic noise and share smooth gradients with AI art; and (3) Extreme aspect ratios (e.g., 3408x2272 or 1140x641) experience anisotropic compression when resized to 160x160.",
    )
    pdf.add_figure(
        fig_dir / "task2_6_error_analysis_grid.png",
        "Figure 2.6: Grid of 10 misclassified test images showing ground-truth class, predicted P(AI), original resolution, and file size.",
    )
    err_csv = fig_dir / "task2_6_error_hypotheses.csv"
    if err_csv.exists():
        pdf.add_dataframe_table(
            pd.read_csv(err_csv)[["Case #", "File Name", "Error Type", "P(AI)", "Orig Size", "Texture Var", "Diagnostic Hypothesis"]],
            "Table 2.6: Diagnostic Hypotheses for 10 Misclassified Test Images",
        )

    pdf.subtask_heading("Task 2.7: Tool Lab — Keras Tuner (Fashion-MNIST Toy & VisionGuard 8-Trial Search)")
    pdf.body_paragraph(
        "Which Setting Mattered Most & Why Validation Tuning Can Overfit",
        "In the toy lab on Fashion-MNIST, RandomSearch over units {32..128} and learning_rate {3e-4..1e-2} found 96 units with lr=0.01 achieved the highest validation accuracy (0.8170). On VisionGuard (8 trials across learning_rate, dropout_rate, and num_unfrozen_layers), num_unfrozen_layers had the largest impact range on mean validation AUC (0.0683), followed by dropout_rate (0.0473) and learning_rate (0.0451), because unfreezing 30 layers with a large learning rate (1e-3) destabilized pretrained weights (val_loss=1.7224). Choosing the single highest validation score across many trials can overfit the validation set (selection bias / multiple-comparisons problem) because random fluctuations on a 146-image validation split get rewarded; therefore a separate held-out test set must be evaluated only once.",
    )
    toy_kt_csv = fig_dir / "task2_7_toy_tuner_results.csv"
    if toy_kt_csv.exists():
        pdf.add_dataframe_table(pd.read_csv(toy_kt_csv), "Table 2.7a: Keras Tuner Toy Results on Fashion-MNIST")
    vg_kt_csv = fig_dir / "task2_7_vg_tuner_results.csv"
    if vg_kt_csv.exists():
        pdf.add_dataframe_table(pd.read_csv(vg_kt_csv), "Table 2.7b: Keras Tuner 8-Trial Results on VisionGuard (23L0713_A2_Ops)")

    pdf.subtask_heading("Task 2.8 (Bonus): Grad-CAM Visual Explanations")
    pdf.body_paragraph(
        "What the Model Looks At",
        "We computed Grad-CAM heatmaps with respect to the spatial feature activations of MobileNetV2 for 4 correct predictions (top row) and 4 misclassified predictions (bottom row). On correct detections, the model focuses on facial geometry, hand/limb contours, and complex foreground textures; on misclassified samples, activation diffuses across uniform sky/background gradients or stylized graphic borders.",
    )
    pdf.add_figure(
        fig_dir / "task2_8_gradcam_grid.png",
        "Figure 2.8 (Bonus): Grad-CAM heatmaps overlaid on 4 correctly classified (top) and 4 misclassified (bottom) test images.",
    )

    # CHAPTER 3: TASK 3
    pdf.add_page()
    pdf.chapter_title("Chapter 3 — Task 3: Experiment Tracking with MLflow and DagsHub (12 Marks)")

    pdf.subtask_heading("Task 3.1 & 3.2: MLflow Concepts, Local Toy Lab & DagsHub Remote Setup")
    pdf.body_paragraph(
        "Definitions in Our Own Words",
        "* Experiment: A named logical grouping of related training runs addressing the same ML problem (e.g., 23L0713_A2_Ops_VisionGuard).\n"
        "* Run: A single execution of training/evaluation code recorded inside an experiment with a unique run_id and timestamps.\n"
        "* Parameter: An immutable key-value input setting chosen before training starts (e.g., learning_rate=0.0001, img_size=160).\n"
        "* Metric: A numeric performance measurement logged either per epoch (step-indexed curves like loss, val_auc) or once at evaluation (test_f1).\n"
        "* Artifact: Any output file attached to a run, such as confusion matrix PNGs, split CSVs, requirements.txt, or serialized .keras models.\n"
        "* Tag: Mutable metadata labels used for filtering and provenance auditing (e.g., git_commit, dvc_data_version, roll_number=23L0713).",
    )
    pdf.body_paragraph(
        "DagsHub Setup & What Went Wrong",
        "We connected https://github.com/AbdullahChouhan/23L0713_A2_Ops to DagsHub and stored MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, and MLFLOW_TRACKING_PASSWORD strictly in .env (verified ignored via git check-ignore -v .env). During our first DagsHub run on Windows, MLflow threw UnicodeEncodeError when printing its completion emoji ('\\U0001f3c3 View run ...') to the cp1252 console; we fixed this by reconfiguring stdout/stderr to UTF-8.",
    )
    toy_ml_csv = fig_dir / "task3_1_3_2_toy_runs.csv"
    if toy_ml_csv.exists():
        pdf.add_dataframe_table(pd.read_csv(toy_ml_csv), "Table 3.1/3.2: Local and DagsHub Toy Fashion-MNIST Runs (LR=0.001 vs LR=0.01)")
    ig_txt = fig_dir / "task3_2_git_ignore_proof.txt"
    if ig_txt.exists():
        pdf.code_block("Terminal Proof (Task 3.2): .env is Git-Ignored", ig_txt.read_text(encoding="utf-8"))

    pdf.subtask_heading("Task 3.3 & 3.4: Manual Callback vs. TensorFlow Autologging & Run Artifacts")
    pdf.body_paragraph(
        "Manual Logging vs. mlflow.tensorflow.autolog()",
        "In src/tracking/mlflow_experiments.py, we instrumented training both ways. Manual logging (via ManualMLflowEpochLogger + mlflow.log_params + mlflow.set_tags) captures exact custom domain parameters (unfrozen_layers, augment, dvc_data_version) and held-out test metrics cleanly without noise. Conversely, mlflow.tensorflow.autolog() automatically captures optimizer settings (beta_1, beta_2, epsilon) and epoch metrics with one line of code, but misses custom dataset/DVC provenance tags, custom thresholded test metrics, and custom diagnostic plots unless logged manually.",
    )

    pdf.subtask_heading("Task 3.5: Comparison Across 6 Meaningful Experiment Runs")
    pdf.body_paragraph(
        "Experiment Comparison & Champion Selection",
        "We executed 6 systematic runs on DagsHub MLflow where each run varied one meaningful factor (Backbone, Augmentation, Frozen Layers, Learning Rate, Image Size). Run 4 (MobileNetV2 with 25 unfrozen layers, 160x160 resolution, data augmentation enabled, and fine-tuning LR=1e-4) achieved the highest Test ROC-AUC and F1-Score and was selected as our Champion model.",
    )
    comp_csv = fig_dir / "task3_5_experiment_comparison.csv"
    if comp_csv.exists():
        c_df = pd.read_csv(comp_csv)[
            ["Run Name", "Backbone", "Img Size", "LR", "Augment", "Unfrozen", "Logging", "Val AUC", "Test Acc", "Test F1", "Test ROC-AUC"]
        ]
        pdf.add_dataframe_table(c_df, "Table 3.5: Comparison of 6 VisionGuard Experiment Runs Logged to DagsHub MLflow")
    pdf.add_figure(
        fig_dir / "task3_5_comparison_chart.png",
        "Figure 3.5: Test F1-Score and Test ROC-AUC comparison across all 6 VisionGuard experiment runs.",
    )

    pdf.subtask_heading("Task 3.6: MLflow Model Registry & Loading by '@champion' Alias")
    pdf.body_paragraph(
        "Registry Versioning & Serving Delivery Plan",
        "We registered VisionGuard_23L0713_Classifier with two versions (Version 1: Baseline CNN; Version 2: Fine-Tuned MobileNetV2), assigned the alias 'champion' to Version 2, and verified alias resolution in src/tracking/load_champion_registry.py. For production serving (Task 4 & 6), the web server never calls DagsHub per request; instead, the champion artifact (models/champion_model.keras) is exported during CI/release and mounted/copied into the container so startup loads it once from local disk.",
    )
    reg_txt = fig_dir / "task3_6_registry_prediction.txt"
    if reg_txt.exists():
        pdf.code_block("Terminal Evidence (Task 3.6): python -m src.tracking.load_champion_registry", reg_txt.read_text(encoding="utf-8"))

    # CHAPTER 4 & 5: TASK 4 & TASK 5
    pdf.add_page()
    pdf.chapter_title("Chapter 4 & 5 — Flask Web Application (12 Marks) & Database Trail (8 Marks)")

    pdf.subtask_heading("Task 4.1: Singleton Inference Module & Preprocessing Parity")
    pdf.body_paragraph(
        "Why Loading Per Request Is Unacceptable & Training-Serving Skew Prevention",
        "Deserializing a Keras model and compiling its computation graph takes 1.5 to 4.0 seconds and allocates hundreds of megabytes of RAM; doing so per HTTP request would cause severe latency spikes, CPU thrashing, and Out-Of-Memory (OOM) crashes under concurrent load. Our VisionGuardInferenceService (app/inference.py) loads models/champion_model.keras once at startup and warms up the graph with a dummy tensor, achieving ~60-70 ms inference latency per image. It converts uploaded bytes to 3-channel RGB, resizes with bilinear interpolation to 160x160, and applies the model's built-in Rescaling(1/127.5, offset=-1.0), eliminating training-serving skew.",
    )
    cli_txt = fig_dir / "task4_1_inference_cli.txt"
    if cli_txt.exists():
        pdf.code_block("Console Output (Task 4.1): python -m app.inference on Two Known Test Images", cli_txt.read_text(encoding="utf-8"))

    pdf.subtask_heading("Task 4.2: Routes, Zero-Disk In-Memory Validation & Three Failure Cases")
    pdf.body_paragraph(
        "Validation Pipeline & Security",
        "Implemented in app/main.py: (1) filenames are sanitized via werkzeug.utils.secure_filename to block path traversal ('../../'); (2) disallowed extensions (.txt, .pdf, .exe) are rejected with HTTP 400; (3) payloads exceeding MAX_CONTENT_LENGTH_MB are rejected with HTTP 413; (4) files with valid extensions (.jpg) containing truncated or non-image bytes are caught by Pillow verify()/load() in io.BytesIO and rejected with HTTP 422. No uploaded image is ever written to disk; previews are rendered via in-memory base64 data URIs.",
    )
    pdf.add_figure(
        fig_dir / "task4_2_three_failure_cases.png",
        "Figure 4.2: Verification of the three validation failure cases (Disallowed extension HTTP 400, Oversized file HTTP 413, Corrupt image bytes HTTP 422).",
    )

    pdf.subtask_heading("Task 4.3, 4.4 & Task 5: Production Server, Health Check & Database Audit Trail")
    pdf.body_paragraph(
        "Environment Configuration, Debug Mode Off & Database Trail",
        "All parameters (MODEL_PATH, DECISION_THRESHOLD, DATABASE_URL, FLASK_DEBUG=0) are loaded from environment variables via app/config.py. Running with debug=False and a production WSGI server (gunicorn in Linux Docker, waitress on Windows) is essential because Werkzeug's debug reloader exposes an interactive remote-code-execution debugger on unhandled exceptions. Every prediction writes a permanent SQLAlchemy record (app/database.py -> prediction_logs table) storing timestamp, sanitized filename, dimensions, verdict, confidence, raw P(AI), threshold, model version, inference source, and latency.",
    )
    pdf.add_figure(
        fig_dir / "task4_4_health_check.png",
        "Figure 4.4: Production /health endpoint response confirming debug_mode=False, model_loaded=True, and database connectivity.",
        width=165.0,
    )
    db_csv = fig_dir / "task5_database_history_trail.csv"
    if db_csv.exists():
        h_df = pd.read_csv(db_csv)[["id", "created_at", "filename", "verdict", "confidence_pct", "prob_ai", "inference_source", "latency_ms"]]
        pdf.add_dataframe_table(h_df, "Table 5.1: Queryable Database Audit Trail (prediction_logs Table)")

    # CHAPTER 6: TASK 6
    pdf.add_page()
    pdf.chapter_title("Chapter 6 — Task 6: Containerization with Docker, Compose & CI (14 Marks)")

    pdf.subtask_heading("Task 6.1: Docker Warm-Up & Core Container Definitions")
    pdf.body_paragraph(
        "Core Definitions in Our Own Words",
        "* Image: An immutable, read-only template containing the OS filesystem layers, runtime libraries, and application code required to create a container.\n"
        "* Container: A live, isolated runtime instance of an image with its own writable overlay layer, process namespace, and network interface.\n"
        "* Layer: A cached filesystem diff produced by each Dockerfile instruction (FROM, RUN, COPY); unchanged upper layers are reused across builds.\n"
        "* Volume: A persistent storage mechanism managed by Docker outside the container's ephemeral union filesystem so database files survive container recreation.\n"
        "* Port Mapping: Forwarding traffic from a host port (e.g., -p 5000:5000) across the virtual bridge into a container's internal listening port.",
    )
    w_txt = fig_dir / "task6_1_docker_warmup.txt"
    if w_txt.exists():
        pdf.code_block("Terminal Evidence (Task 6.1): Docker Warm-Up (run, -p, -v, logs, exec, ps, rm, rmi)", w_txt.read_text(encoding="utf-8"), max_lines=28)

    pdf.subtask_heading("Task 6.2 & 6.4: Production Dockerfile, Non-Root Execution & Image Size Optimization")
    pdf.body_paragraph(
        "Instruction Ordering, CPU-Only TensorFlow & Size Optimization (4.24 GB -> 2.13 GB)",
        "In Dockerfile, we copy requirements-prod.txt and run pip install --no-cache-dir BEFORE copying app/ source code so code edits reuse the cached dependency layer. We chose python:3.11-slim and tensorflow-cpu==2.18.0 instead of the full python:3.11 base and cached wheels. As shown in docker images and docker history below, the unoptimized image (23l0713_a2_ops_web:unoptimized) was 4.24 GB (with a 1.88 GB pip layer and 696 MB + 202 MB Debian toolchain layers), whereas our optimized image (23l0713_a2_ops_web:latest) is 2.13 GB (largest layers: 1.52 GB pip install --no-cache-dir, 87.7 MB slim base, 22.5 MB champion_model.keras) — saving 2.11 GB (a 49.8% size reduction). Inside the container, whoami/id confirms execution as non-root uid=999(appuser) gid=999(appgroup).",
    )
    img_txt = fig_dir / "task6_2_6_4_image_analysis.txt"
    if img_txt.exists():
        img_lines = img_txt.read_text(encoding="utf-8").splitlines()
        # Extract the non-root check and the final docker images + docker history tables
        tail_excerpt = "\n".join(img_lines[430:481])
        pdf.code_block("Terminal Evidence (Task 6.2 & 6.4): Image Size Comparison (4.24GB vs 2.13GB) & Layer History", tail_excerpt, max_lines=45)

    pdf.subtask_heading("Task 6.3 & 6.5: Docker Compose Stack, Network Isolation & Failure Diagnosis")
    pdf.body_paragraph(
        "Compose Architecture & Intentional Failure Troubleshooting",
        "docker-compose.yml defines two services on internal bridge network 23l0713_a2_ops_net: db (postgres:16-alpine with named volume 23l0713_a2_ops_pgdata and pg_isready healthcheck) and web (depending on db with condition: service_healthy). The db service publishes NO host ports (docker compose port db 5432 returns :0 and host TCP socket check to 127.0.0.1:5432 returns False), making PostgreSQL unreachable from outside while reachable by service name 'db:5432' from web. For Task 6.5, we executed interactive shells in both services, demonstrated restart/stop/start, and simulated a credential mismatch by setting an invalid POSTGRES_PASSWORD; diagnosed the FATAL: password authentication failed for user 'visionguard_user' error strictly from docker compose logs; and restored .env to recover HTTP 200 healthy status.",
    )
    comp_txt = fig_dir / "task6_3_6_5_compose_and_troubleshooting.txt"
    if comp_txt.exists():
        c_lines = comp_txt.read_text(encoding="utf-8").splitlines()
        compose_excerpt = "\n".join(c_lines[311:338] + [""] + c_lines[430:546])
        pdf.code_block("Terminal Evidence (Task 6.3 & 6.5): Compose Health, Closed DB Port Proof, Shells & Failure Logs", compose_excerpt, max_lines=52)

    pdf.subtask_heading("Task 6.6: GitHub Actions Continuous Integration (CI)")
    pdf.body_paragraph(
        "Workflow Design & Secret Management",
        "We created .github/workflows/ci.yml which triggers on every push/PR to main, installs dependencies, runs ruff check linting, executes all 5 automated validation/database tests in tests/test_app.py via pytest, and builds the production Docker image. Workflow secrets (such as DAGSHUB_TOKEN or cloud deployment keys) are stored in GitHub Repository Settings -> Secrets and variables -> Actions and injected via ${{ secrets.NAME }} at runtime because committing secrets in YAML exposes them permanently in public Git history.",
    )

    # CHAPTER 7 & 8: TASK 7 & TASK 8
    pdf.add_page()
    pdf.chapter_title("Chapter 7 & 8 — Cloud API Deployment & Local Flask Integration (30 Marks)")

    pdf.subtask_heading("Task 7.1: Cloud Model API Deployment on Render & Step-by-Step Instructions")
    pdf.body_paragraph(
        "Cloud Deployment Pathway (Pathway C — Render Docker Web Service) & Architecture",
        f"Following Pathway C, the VisionGuard Docker container (23L0713_A2_Ops) is deployed as a live public HTTPS Web Service on Render ({cloud_url}, Service ID: srv-dbnno81srm7s739v7io0) exposing POST {cloud_url}/api/predict and GET {cloud_url}/health. Step-by-step deployment workflow:\n"
        "1. Configured Dockerfile with dynamic $PORT binding (gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 2) and TensorFlow CPU arena/thread tuning (OMP_NUM_THREADS=1, MALLOC_ARENA_MAX=2) to fit comfortably inside Render's 512 MB Free Tier RAM envelope.\n"
        "2. Pushed repository (Dockerfile, render.yaml, requirements-prod.txt, app/, models/champion_model.keras) to GitHub (https://github.com/AbdullahChouhan/23L0713_A2_Ops).\n"
        "3. Created a Render Docker Web Service named 23L0713_A2_Ops connected to the main branch.\n"
        f"4. Verified live container readiness at {cloud_url}/health (HTTP 200, model_loaded=true, debug_mode=false) and tested multipart image inference against POST {cloud_url}/api/predict.",
    )
    pdf.add_figure(
        fig_dir / "task7_api_test_evidence.png",
        "Figure 7.1: REST API POST /api/predict multipart request and logged JSON response payloads.",
    )

    pdf.subtask_heading("Task 8.1: Integrating Localhost Flask UI with Cloud API & Database History")
    pdf.body_paragraph(
        "How Local Flask Communicates with the Cloud Model API",
        "When USE_CLOUD_API=true and CLOUD_API_URL is set in .env (or selected via the radio toggle on the Flask upload form), the local Flask server (http://localhost:5000) validates the uploaded image in memory and dispatches a multipart POST request with header X-API-Key to {CLOUD_API_URL}/api/predict. It parses the returned JSON verdict, confidence, raw P(AI) score, and model version, records a permanent audit row in prediction_logs with inference_source='cloud_api', and renders both the result page and the /history view.",
    )
    pdf.add_figure(
        fig_dir / "task8_multiple_inputs_results.png",
        "Figure 8.1: Six test inputs (3 AI-Generated, 3 Real) evaluated through the VisionGuard API and logged in the database history trail.",
    )

    pdf.output(str(output_pdf))
    print(f"[INFO] Successfully generated PDF report: {output_pdf.resolve()} ({output_pdf.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    build_pdf_report()
