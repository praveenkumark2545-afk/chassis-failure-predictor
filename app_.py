import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "random_forest_chassis_model.joblib"
METADATA_PATH = BASE_DIR / "model_metadata.json"

st.set_page_config(
    page_title="Chassis Failure Predictor",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom dark engineering-dashboard styling
st.markdown(
    """
    <style>
    :root {
        --page: #0b1220;
        --panel: #111c2e;
        --panel-2: #17243a;
        --border: #2b3b52;
        --text: #eef4fb;
        --muted: #b5c3d6;
        --blue: #38bdf8;
    }
    .stApp {
        background: radial-gradient(ellipse at top, #14233a 0%, #0b1220 58%, #080e18 100%);
        color: var(--text);
    }
    [data-testid="stHeader"] { background: rgba(11,18,32,.92); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101a2a 0%, #0c1422 100%);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * { color: var(--text); }
    .block-container { padding-top: 2rem; padding-bottom: 2.5rem; max-width: 1250px; }
    .hero {
        padding: 1.65rem 1.8rem; border-radius: 20px;
        background: linear-gradient(120deg, #10243a 0%, #174a70 60%, #075985 100%);
        border: 1px solid #2d668b;
        color: #f8fbff; margin-bottom: 1.25rem;
        box-shadow: 0 14px 36px rgba(0,0,0,.24);
    }
    .hero h1 { color: #ffffff !important; margin: 0 0 .4rem 0; font-size: 2.2rem; }
    .hero p { color: #dbeafe !important; margin: 0; font-size: 1rem; }
    h1, h2, h3, h4, p, label, .stMarkdown, [data-testid="stMetricLabel"] { color: var(--text); }
    [data-testid="stCaptionContainer"] { color: var(--muted); }
    [data-testid="stMetric"] {
        background: var(--panel); border: 1px solid var(--border);
        padding: 1rem 1.1rem; border-radius: 14px;
    }
    [data-testid="stMetricValue"] { color: #f8fafc; }
    [data-testid="stNumberInput"] input {
        background: #111c2e !important; color: #f8fafc !important;
        border: 1px solid #40536d !important; border-radius: 10px !important;
    }
    [data-testid="stNumberInput"] button {
        background: #1b2b42 !important; color: #eaf2fb !important;
        border-color: #40536d !important;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #0284c7, #2563eb) !important;
        color: white !important; border: 1px solid #38bdf8 !important;
        border-radius: 11px !important; font-weight: 700 !important;
        min-height: 2.8rem; box-shadow: 0 6px 18px rgba(2,132,199,.2);
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(90deg, #0369a1, #1d4ed8) !important;
        border-color: #7dd3fc !important;
    }
    .result-card {
        padding: 1rem 1.15rem; border-radius: 13px; border: 1px solid;
        min-height: 120px; margin-bottom: .5rem;
    }
    .result-failure { background: #351a26; border-color: #be5366; }
    .result-safe { background: #12352d; border-color: #2f9e78; }
    .result-title { font-size: 1.2rem; font-weight: 800; margin-bottom: .35rem; }
    .result-copy { color: #e2e8f0; font-size: .94rem; }
    .notice-card {
        padding: .95rem 1.1rem; border-radius: 12px; margin: .6rem 0 1rem 0;
        background: #1b2b42; border: 1px solid #3b5877; color: #e6f2ff;
    }
    .warning-card {
        padding: .95rem 1.1rem; border-radius: 12px; margin: .6rem 0 1rem 0;
        background: #3a2d13; border: 1px solid #a87924; color: #fff0c2;
    }
    [data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
    [data-testid="stExpander"] { background: rgba(17,28,46,.72); border: 1px solid var(--border); border-radius: 10px; }
    hr { border-color: var(--border); }
    </style>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error("Model files are missing. Keep the model, metadata, and app.py in the same folder.")
    st.stop()

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata

model, metadata = load_artifacts()
ranges = metadata["feature_ranges"]

st.markdown(
    """
    <div class="hero">
      <h1>Chassis Failure Prediction</h1>
      <p>Random Forest model · ANSYS structural simulation data · Enter design parameters to estimate the failure class.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div class=\"notice-card\"><strong>Research prototype only</strong><br>"
    "This model estimates labels learned from the supplied dataset. It does not replace ANSYS verification, "
    "engineering review, or a certified safety assessment.</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Model information")
    st.write(f"**Algorithm:** {metadata['model_name']}")
    st.write(f"**Training simulations:** {metadata['training_rows']}")
    st.write("**Inputs:**")
    st.write("• Tube outer diameter (mm)")
    st.write("• Wall thickness (mm)")
    st.write("• Applied load (N)")
    st.divider()
    st.caption("Predictions are most defensible within the ranges represented in the training data.")
    with st.expander("Training data ranges"):
        for feature, label in [
            ("Tube_OD_mm", "Tube OD (mm)"),
            ("Thickness_mm", "Thickness (mm)"),
            ("Load_N", "Load (N)"),
        ]:
            st.write(f"**{label}:** {ranges[feature]['min']:,.3f} to {ranges[feature]['max']:,.3f}")

st.subheader("1. Enter chassis parameters")
st.caption("Use values within the dataset's range where possible. The model will warn you if inputs are outside it.")

col1, col2, col3 = st.columns(3)
with col1:
    od = st.number_input(
        "Tube outer diameter (mm)",
        min_value=0.1,
        value=float(round((ranges["Tube_OD_mm"]["min"] + ranges["Tube_OD_mm"]["max"]) / 2, 3)),
        step=0.1,
        format="%.3f",
        help="Outer diameter of the chassis tube."
    )
with col2:
    thickness = st.number_input(
        "Tube wall thickness (mm)",
        min_value=0.1,
        value=float(round((ranges["Thickness_mm"]["min"] + ranges["Thickness_mm"]["max"]) / 2, 3)),
        step=0.1,
        format="%.3f",
        help="Wall thickness of the chassis tube."
    )
with col3:
    load = st.number_input(
        "Applied load (N)",
        min_value=0.0,
        value=float(round((ranges["Load_N"]["min"] + ranges["Load_N"]["max"]) / 2, 1)),
        step=100.0,
        format="%.1f",
        help="Applied structural load used in the simulation."
    )

run_prediction = st.button("Predict chassis outcome", type="primary", use_container_width=True)

if run_prediction:
    input_df = pd.DataFrame([{
        "Tube_OD_mm": od,
        "Thickness_mm": thickness,
        "Load_N": load,
    }], columns=metadata["features"])

    out_of_range = []
    for feature, value in input_df.iloc[0].items():
        lo = ranges[feature]["min"]
        hi = ranges[feature]["max"]
        if value < lo or value > hi:
            out_of_range.append(f"{feature}: entered {value:g}, training range {lo:g}–{hi:g}")

    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    class_to_index = {int(c): i for i, c in enumerate(model.classes_)}
    failure_probability = float(probabilities[class_to_index[1]]) if 1 in class_to_index else 0.0
    safe_probability = float(probabilities[class_to_index[0]]) if 0 in class_to_index else 0.0

    st.subheader("2. Prediction")
    result_col, prob_col, safe_col = st.columns(3)

    with result_col:
        if prediction == 1:
            st.markdown(
                "<div class=\"result-card result-failure\">"
                "<div class=\"result-title\">PREDICTED: FAILURE</div>"
                "<div class=\"result-copy\">The model classifies this input as Failure (1).</div></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class=\"result-card result-safe\">"
                "<div class=\"result-title\">PREDICTED: SAFE</div>"
                "<div class=\"result-copy\">The model classifies this input as Safe (0).</div></div>",
                unsafe_allow_html=True,
            )

    with prob_col:
        st.metric("Model-estimated failure score", f"{failure_probability:.1%}")
        st.progress(min(max(failure_probability, 0.0), 1.0))
    with safe_col:
        st.metric("Model-estimated safe score", f"{safe_probability:.1%}")
        st.progress(min(max(safe_probability, 0.0), 1.0))

    if out_of_range:
        range_items = "<br>".join(f"• {item}" for item in out_of_range)
        st.markdown(
            "<div class=\"warning-card\"><strong>Input range warning</strong><br>"
            "One or more inputs are outside the training-data range. The prediction may be unreliable:<br>"
            f"{range_items}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Input summary")
    st.dataframe(input_df, use_container_width=True, hide_index=True)
    report = {
        "Tube_OD_mm": od,
        "Thickness_mm": thickness,
        "Load_N": load,
        "Predicted_Label": prediction,
        "Predicted_Class": "Failure" if prediction == 1 else "Safe",
        "Failure_Score": failure_probability,
        "Safe_Score": safe_probability,
        "Outside_Training_Range": bool(out_of_range),
    }
    st.download_button(
        "Download prediction as CSV",
        data=pd.DataFrame([report]).to_csv(index=False).encode("utf-8"),
        file_name="chassis_prediction.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()
st.subheader("Model validation reference")
validation = metadata.get("validation_80_20", {})
if validation:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Holdout accuracy", f"{validation['accuracy']:.1%}")
    m2.metric("Failure recall", f"{validation['recall_failure']:.1%}")
    m3.metric("Failure F1-score", f"{validation['f1_failure']:.3f}")
    m4.metric("ROC-AUC", f"{validation['roc_auc']:.3f}" if validation.get("roc_auc") is not None else "N/A")
    st.caption(
        f"Reference only: stratified 80/20 split with {validation['test_rows']} test simulations. "
        "These metrics are not a guarantee of performance on new designs."
    )

with st.expander("What do the scores mean?"):
    st.write(
        "**Predicted class** is the model's classification using its default decision threshold. "
        "**Failure score** is the Random Forest's estimated class probability for label 1. "
        "These probabilities are model outputs and are not automatically calibrated real-world failure probabilities."
    )

st.caption("Model inputs: Tube_OD_mm, Thickness_mm, Load_N. Target: Failure_Label (0 = Safe, 1 = Failure).")
