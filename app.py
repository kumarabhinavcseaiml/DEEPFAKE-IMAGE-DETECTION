"""Polished Streamlit prototype for the Deepfake Image Detection project."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from skimage.feature import hog


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
REPORTS_DIR = ROOT / "reports"
COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
SUMMARY_PATH = REPORTS_DIR / "dataset_summary.json"

st.set_page_config(
    page_title="Deepfake Lens | Image Intelligence",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --ink: #17212b;
        --muted: #64748b;
        --teal: #0d9488;
        --teal-dark: #0f766e;
        --line: #e6edf0;
        --surface: #ffffff;
    }
    html, body, [class*="css"]  {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background: linear-gradient(180deg, #f5f9f9 0%, #f7f9fc 48%, #f8fafc 100%);
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    [data-testid="stSidebar"] {
        background: #102a34;
        border-right: 1px solid rgba(255,255,255,.08);
    }
    [data-testid="stSidebar"] * { color: #e7f2f2; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #b9cccf; }
    [data-testid="stSidebar"] .stRadio label { padding: 7px 0; }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] { gap: 7px; }
    .brand-mark {
        width: 46px; height: 46px; border-radius: 15px;
        display: flex; align-items: center; justify-content: center;
        background: linear-gradient(135deg, #2dd4bf, #0f766e);
        color: #062c2b; font-size: 23px; font-weight: 800;
        box-shadow: 0 8px 20px rgba(13,148,136,.25);
    }
    .sidebar-brand { display:flex; gap:12px; align-items:center; margin: 5px 0 24px 0; }
    .sidebar-brand-title { font-family:'Manrope',sans-serif; font-weight:800; font-size:17px; color:#f0fdfa; line-height:1.15; }
    .sidebar-brand-sub { font-size:10px; color:#9bb8ba; letter-spacing:1.6px; text-transform:uppercase; margin-top:4px; }
    .hero {
        position: relative; overflow: hidden;
        border-radius: 25px; padding: 30px 34px 28px 34px;
        background: radial-gradient(circle at 82% 18%, rgba(94,234,212,.26), transparent 27%),
                    linear-gradient(118deg, #102f3a 0%, #164d53 52%, #0f766e 100%);
        box-shadow: 0 16px 40px rgba(15, 68, 75, .13);
        margin: 4px 0 25px 0;
    }
    .hero-kicker { color:#99f6e4; text-transform:uppercase; letter-spacing:2.1px; font-size:10px; font-weight:700; margin-bottom:9px; }
    .hero-title { color:#f0fdfa; font-family:'Manrope',sans-serif; font-size:clamp(26px,3vw,38px); line-height:1.12; font-weight:800; margin:0; }
    .hero-copy { color:#c5e4e1; font-size:14px; max-width:720px; margin-top:12px; line-height:1.65; }
    .hero-pill { display:inline-block; margin-top:18px; padding:6px 11px; border-radius:999px; color:#ccfbf1; background:rgba(153,246,228,.12); border:1px solid rgba(204,251,241,.2); font-size:11px; font-weight:600; }
    .section-heading { font-family:'Manrope',sans-serif; color:#142a35; font-size:21px; font-weight:800; margin: 12px 0 3px 0; }
    .section-caption { color:#718096; font-size:13px; margin:0 0 16px 0; }
    .metric-card {
        background: #ffffff; border: 1px solid #e7eef0; border-radius: 17px;
        padding: 17px 18px 15px 18px; min-height: 112px;
        box-shadow: 0 5px 18px rgba(20,54,64,.035);
    }
    .metric-label { color:#718096; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:1.1px; }
    .metric-value { color:#142a35; font-family:'Manrope',sans-serif; font-weight:800; font-size:27px; margin-top:7px; line-height:1.1; }
    .metric-note { color:#94a3b8; font-size:11px; margin-top:6px; }
    .panel {
        background:#fff; border:1px solid #e7eef0; border-radius:19px;
        padding:19px 21px; box-shadow:0 5px 18px rgba(20,54,64,.035);
    }
    .soft-note {
        padding:12px 15px; background:#effaf8; border:1px solid #cceee8;
        color:#28645e; border-radius:12px; font-size:12px; line-height:1.55;
    }
    .result-real {
        border-radius:18px; padding:19px; background:#ecfdf5;
        border:1px solid #b7eed1; color:#166534;
    }
    .result-fake {
        border-radius:18px; padding:19px; background:#fff1f2;
        border:1px solid #fecdd3; color:#9f1239;
    }
    .result-title { font-family:'Manrope',sans-serif; font-size:25px; font-weight:800; margin:3px 0; }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #0f766e, #0d9488); color:white;
        border:0; border-radius:11px; padding: .62rem 1.2rem; font-weight:700;
        box-shadow:0 7px 17px rgba(13,148,136,.18);
    }
    div.stButton > button[kind="primary"]:hover { background:linear-gradient(100deg,#115e59,#0f766e); color:white; }
    div[data-testid="stFileUploader"] {
        background:#f8fbfb; border:1px dashed #9ccac5; border-radius:15px; padding:8px;
    }
    .footer { color:#9aa9b4; font-size:11px; text-align:center; padding:25px 0 8px 0; }
    hr { border-color:#e6edf0; }
    </style>
    """,
    unsafe_allow_html=True,
)
def card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def hero(kicker: str, title: str, copy: str, pill: str = "ML MINI PROJECT  •  COMPUTER VISION") -> None:
    st.markdown(
        f"""<div class="hero">
          <div class="hero-kicker">{kicker}</div>
          <h1 class="hero-title">{title}</h1>
          <div class="hero-copy">{copy}</div>
          <span class="hero-pill">{pill}</span>
        </div>""",
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading trained model…")
def load_model_bundle(path_string: str):
    return joblib.load(path_string)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def extract_features(image: Image.Image, image_size: tuple[int, int], hog_settings: dict) -> np.ndarray:
    image_array = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, image_size, interpolation=cv2.INTER_AREA)
    features = hog(gray, **hog_settings)
    return np.asarray(features, dtype=np.float32).reshape(1, -1)


def section_title(title: str, caption: str = "") -> None:
    st.markdown(f'<div class="section-heading">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="section-caption">{caption}</div>', unsafe_allow_html=True)
with st.sidebar:
    st.markdown(
        """<div class="sidebar-brand">
            <div class="brand-mark">⌕</div>
            <div><div class="sidebar-brand-title">Deepfake Lens</div>
            <div class="sidebar-brand-sub">Image intelligence</div></div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("### WORKSPACE")
    page = st.radio(
        "Navigation",
        ["Overview", "Detect an image", "Model performance", "About the project"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**PROJECT STATUS**")
    if MODEL_PATH.exists():
        st.success("Trained model available")
    else:
        st.warning("Model not trained yet")
    st.markdown("---")
    st.caption("Educational prototype · HOG + classical ML")


if not MODEL_PATH.exists():
    hero(
        "GETTING STARTED",
        "Your detection workspace is almost ready.",
        "Train the three machine-learning models first. The training script will save the best model and generate the graphs used in this dashboard.",
    )
    st.error("Trained model not found: `models/best_model.joblib`.")
    st.markdown("**Run these commands in the project folder:**")
    st.code(
        r".\.venv313\Scripts\python.exe -m pip install -r requirements.txt" + "\n"
        + r".\.venv313\Scripts\python.exe -u train.py" + "\n"
        + r".\.venv313\Scripts\python.exe -m streamlit run app.py",
        language="powershell",
    )
    st.info("Keep your dataset in `data/raw/real/` and `data/raw/fake/`. The ZIP contains the code, not the large image dataset.")
    st.stop()

try:
    bundle = load_model_bundle(str(MODEL_PATH))
    model = bundle["model"]
    best_model_name = bundle.get("best_model_name", "Selected ML model")
    image_size = tuple(bundle.get("image_size", (64, 64)))
    hog_settings = bundle.get(
        "hog_settings",
        {
            "orientations": 9,
            "pixels_per_cell": (8, 8),
            "cells_per_block": (2, 2),
            "block_norm": "L2-Hys",
        },
    )
except Exception as exc:
    st.error(f"The model file could not be loaded: {exc}")
    st.info("Run `python train.py` again to create a fresh model file.")
    st.stop()

summary = read_json(SUMMARY_PATH)
if not summary:
    summary = read_json(REPORTS_DIR / "training_summary.json")

comparison = None
if COMPARISON_PATH.exists():
    try:
        comparison = pd.read_csv(COMPARISON_PATH)
    except (OSError, pd.errors.ParserError):
        comparison = None
if page == "Overview":
    hero(
        "WELCOME TO DEEPFAKE LENS",
        "See beyond what the eye can see.",
        "An educational machine-learning prototype that examines visual patterns in face images and predicts whether an image resembles the REAL or FAKE examples in its training dataset.",
        f"BEST VALIDATION MODEL  •  {best_model_name.upper()}",
    )

    section_title("Project at a glance", "Your dataset and trained model, in one place.")
    real_count = int(summary.get("real_images", 0))
    fake_count = int(summary.get("fake_images", 0))
    total_count = int(summary.get("total_images", real_count + fake_count))
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("TOTAL IMAGES", f"{total_count:,}", "Readable images used for training")
    with c2:
        card("REAL IMAGES", f"{real_count:,}", "Authentic face examples")
    with c3:
        card("FAKE IMAGES", f"{fake_count:,}", "AI-generated / manipulated examples")
    with c4:
        card("SELECTED MODEL", best_model_name, "Chosen using validation F1-score")

    st.write("")
    left, right = st.columns([1.08, 0.92], gap="large")
    with left:
        section_title("Dataset distribution", "Class counts after unreadable images were skipped.")
        chart_path = REPORTS_DIR / "class_distribution.png"
        if chart_path.exists():
            st.image(str(chart_path), use_container_width=True)
        else:
            st.info("Dataset graph will appear after training.")
    with right:
        section_title("How the detector works", "From uploaded image to model prediction.")
        st.markdown(
            """<div class="panel">
              <div style="display:flex;gap:13px;align-items:center;margin:5px 0 15px 0">
                <div class="brand-mark" style="width:36px;height:36px;font-size:16px">1</div>
                <div><b>Image preprocessing</b><br><span style="color:#718096;font-size:12px">Grayscale conversion and resizing</span></div>
              </div>
              <div style="height:1px;background:#edf2f4;margin:0 0 15px 0"></div>
              <div style="display:flex;gap:13px;align-items:center;margin-bottom:15px">
                <div class="brand-mark" style="width:36px;height:36px;font-size:16px">2</div>
                <div><b>HOG feature extraction</b><br><span style="color:#718096;font-size:12px">Shape and gradient information</span></div>
              </div>
              <div style="height:1px;background:#edf2f4;margin:0 0 15px 0"></div>
              <div style="display:flex;gap:13px;align-items:center;margin-bottom:15px">
                <div class="brand-mark" style="width:36px;height:36px;font-size:16px">3</div>
                <div><b>Machine-learning classifier</b><br><span style="color:#718096;font-size:12px">Logistic Regression, Random Forest and SVM</span></div>
              </div>
              <div style="height:1px;background:#edf2f4;margin:0 0 15px 0"></div>
              <div style="display:flex;gap:13px;align-items:center">
                <div class="brand-mark" style="width:36px;height:36px;font-size:16px">4</div>
                <div><b>Prediction</b><br><span style="color:#718096;font-size:12px">REAL or FAKE with model confidence</span></div>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown(
        '<div class="soft-note"><b>Important:</b> This prototype learns patterns from one training dataset. A REAL/FAKE label or confidence score is not forensic proof that an image is authentic or manipulated. Results may be unreliable for unseen generators, heavy editing, low-quality images or images that do not contain a clear face.</div>',
        unsafe_allow_html=True,
    )

elif page == "Detect an image":
    hero(
        "IMAGE ANALYSIS",
        "Check an image in a few seconds.",
        "Upload one clear face image. The selected trained model will extract HOG features and return a REAL or FAKE prediction.",
        f"ACTIVE MODEL  •  {best_model_name.upper()}",
    )

    left, right = st.columns([1, 1], gap="large")
    with left:
        section_title("01 · Upload image", "Supported formats: JPG, JPEG, PNG and WEBP.")
        uploaded_file = st.file_uploader(
            "Choose a face image",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            help="A single, front-facing face crop is recommended for this educational prototype.",
        )
        if uploaded_file is not None:
            try:
                preview = Image.open(uploaded_file).convert("RGB")
                st.image(preview, caption="Image ready for analysis", use_container_width=True)
            except Exception:
                preview = None
                st.error("This file could not be opened as an image.")
        else:
            preview = None
            st.markdown(
                '<div class="panel" style="text-align:center;padding:42px 15px;color:#718096">'
                '<div style="font-size:38px;margin-bottom:8px">▧</div>'
                '<b style="color:#28434b">Your image preview will appear here</b>'
                '<div style="font-size:12px;margin-top:6px">Upload a face image to get started</div></div>',
                unsafe_allow_html=True,
            )

    with right:
        section_title("02 · Detection result", "The prediction appears here after analysis.")
        if preview is None:
            st.markdown(
                '<div class="panel" style="min-height:230px;display:flex;align-items:center;justify-content:center;text-align:center;color:#94a3b8">'
                '<div><div style="font-size:34px">◉</div><b>Waiting for an image</b><br><span style="font-size:12px">Your result and confidence will be shown here.</span></div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="soft-note">Ready to analyse. Select the button below to run the trained model.</div>',
                unsafe_allow_html=True,
            )
            if st.button("Analyse image", type="primary", use_container_width=True):
                try:
                    with st.spinner("Extracting image features and running prediction…"):
                        feature_vector = extract_features(preview, image_size, hog_settings)
                        prediction = str(model.predict(feature_vector)[0]).lower()
                        probability_map = {}
                        if hasattr(model, "predict_proba"):
                            probabilities = model.predict_proba(feature_vector)[0]
                            for class_name, probability in zip(model.classes_, probabilities):
                                probability_map[str(class_name).lower()] = float(probability)
                        predicted_confidence = probability_map.get(prediction)

                    st.session_state["last_prediction"] = {
                        "prediction": prediction,
                        "probabilities": probability_map,
                        "confidence": predicted_confidence,
                    }
                except Exception as exc:
                    st.error(f"Image analysis failed: {exc}")

            result = st.session_state.get("last_prediction")
            if result:
                prediction = result["prediction"]
                confidence = result["confidence"]
                if prediction == "real":
                    st.markdown(
                        '<div class="result-real"><div style="font-size:11px;font-weight:700;letter-spacing:1.3px">MODEL PREDICTION</div><div class="result-title">✓ REAL</div><div style="font-size:12px">The image was classified as REAL by this model.</div></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="result-fake"><div style="font-size:11px;font-weight:700;letter-spacing:1.3px">MODEL PREDICTION</div><div class="result-title">! FAKE</div><div style="font-size:12px">The image was classified as FAKE by this model.</div></div>',
                        unsafe_allow_html=True,
                    )
                if confidence is not None:
                    st.metric("Model confidence", f"{confidence * 100:.1f}%")
                    st.progress(float(np.clip(confidence, 0, 1)))
                probabilities = result["probabilities"]
                if probabilities:
                    probability_df = pd.DataFrame(
                        {"Estimated model probability": [probabilities.get("real", 0), probabilities.get("fake", 0)]},
                        index=["REAL", "FAKE"],
                    )
                    st.bar_chart(probability_df, horizontal=True)
                st.caption("Confidence is the model's estimated probability for its chosen class; it is not a guarantee of correctness.")

    st.markdown(
        '<div class="soft-note" style="margin-top:18px"><b>Use responsibly:</b> Do not use this experimental result as the sole basis for accusations, identity verification, employment decisions or other high-impact judgments.</div>',
        unsafe_allow_html=True,
    )

elif page == "Model performance":
    hero(
        "EXPERIMENTAL RESULTS",
        "Compare the models. Understand the metrics.",
        "All test metrics below are generated by train.py using the held-out test split. The best model is selected separately using validation F1-score.",
        "HELD-OUT TEST SET  •  20% OF READABLE IMAGES",
    )

    if comparison is None or comparison.empty:
        st.warning("No model comparison is available yet. Run train.py to generate the reports.")
    else:
        required = {"Model", "Accuracy", "Precision", "Recall", "F1 Score"}
        if not required.issubset(set(comparison.columns)):
            st.error("The model comparison file is missing expected columns. Please retrain using the supplied train.py.")
        else:
            best_row = comparison.sort_values("F1 Score", ascending=False).iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                card("TOP TEST ACCURACY", f"{comparison['Accuracy'].max() * 100:.1f}%", "Highest accuracy on held-out test data")
            with c2:
                card("TOP F1 SCORE", f"{best_row['F1 Score'] * 100:.1f}%", str(best_row["Model"]))
            with c3:
                card("ALGORITHMS", str(len(comparison)), "Models trained and evaluated")
            with c4:
                card("FEATURE METHOD", "HOG", "Histogram of Oriented Gradients")

            st.write("")
            section_title("Performance comparison", "Higher values indicate stronger performance for each metric.")
            chart_columns = [column for column in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"] if column in comparison.columns]
            display_data = comparison.set_index("Model")[chart_columns] * 100
            st.bar_chart(display_data, y_label="Score (%)")

            st.write("")
            section_title("Exact test metrics", "Precision, recall, F1 and ROC-AUC focus on the FAKE class as positive.")
            formatted = comparison.copy()
            for column in chart_columns:
                formatted[column] = formatted[column].map(lambda value: f"{value * 100:.2f}%" if pd.notna(value) else "N/A")
            st.dataframe(formatted, use_container_width=True, hide_index=True)

    st.write("")
    section_title("Confusion matrices", "Rows are actual labels; columns are model predictions.")
    matrix_files = [
        ("Logistic Regression", "logistic_regression_confusion_matrix.png"),
        ("Random Forest", "random_forest_confusion_matrix.png"),
        ("SVM", "svm_confusion_matrix.png"),
    ]
    cols = st.columns(3)
    for col, (name, filename) in zip(cols, matrix_files):
        with col:
            st.markdown(f"**{name}**")
            path = REPORTS_DIR / filename
            if path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.caption("Available after training.")

    roc_path = REPORTS_DIR / "roc_curves.png"
    if roc_path.exists():
        st.write("")
        section_title("ROC curve", "Shows the trade-off between true-positive and false-positive rates.")
        st.image(str(roc_path), use_container_width=True)

elif page == "About the project":
    hero(
        "PROJECT DOCUMENTATION",
        "A small ML prototype with a clear workflow.",
        "This mini-project demonstrates a complete binary image-classification workflow: dataset preparation, preprocessing, feature extraction, model comparison, evaluation and an interactive demonstration.",
        "PYTHON  •  OPENCV  •  SCIKIT-LEARN  •  STREAMLIT",
    )

    section_title("Problem statement")
    st.markdown(
        """This project explores whether a face image resembles the REAL or FAKE examples in a labelled image dataset. It converts images into HOG features and compares three classical machine-learning algorithms. It is an educational classifier, not a forensic authentication system."""
    )

    section_title("Objectives")
    objectives = [
        "Prepare a labelled dataset of real and fake face images.",
        "Preprocess images and explore the class distribution.",
        "Extract Histogram of Oriented Gradients (HOG) features.",
        "Train Logistic Regression, Random Forest and SVM classifiers.",
        "Compare Accuracy, Precision, Recall, F1-score and ROC-AUC.",
        "Demonstrate predictions through an image-upload web application.",
    ]
    for index, objective in enumerate(objectives, start=1):
        st.markdown(f"**{index:02d}.** {objective}")

    section_title("Technology stack")
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        card("LANGUAGE", "Python", "Project implementation")
    with t2:
        card("IMAGE PROCESSING", "OpenCV", "Resize and grayscale")
    with t3:
        card("MACHINE LEARNING", "Scikit-learn", "Three classifiers")
    with t4:
        card("WEB DEMO", "Streamlit", "Interactive prototype")

    section_title("Limitations and future scope")
    st.markdown(
        """- HOG captures visual gradients and texture-like structure, but may miss subtle or modern deepfake artifacts.
        - The current data split is random and image-level. If related images or near-duplicates occur across splits, the test score may be optimistic.
        - Performance may change substantially on images from generators or manipulations not represented in the training dataset.
        - Future improvements could include face detection and alignment, duplicate-aware or identity-aware splitting, testing on an independent benchmark, and a CNN-based model."""
    )

st.markdown('<div class="footer">DEEPFAKE LENS · MACHINE LEARNING MINI PROJECT · EDUCATIONAL USE ONLY</div>', unsafe_allow_html=True)
