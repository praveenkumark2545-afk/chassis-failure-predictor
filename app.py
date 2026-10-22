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

# Lightweight custom styling
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #f7f9fc 0%, #eef3f9 100%); }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }
    .hero {
        padding: 1.5rem 1.7rem; border-radius: 18px;
        background: linear-gradient(120deg, #10243a 0%, #1f4e79 100%);
        color: white; margin-bottom: 1.2rem;
        box-shadow: 0 10px 28px rgba(16,36,58,0.14);
    }
    .hero h1 { color: white; margin-bottom: .35rem; }
    .hero p { color: #e4edf7; margin-bottom: 0; font-size: 1.02rem; }
    .metric-card {
        background: white; border: 1px solid #dce5ef; border-radius: 14px;
        padding: 1rem 1.1rem; min-height: 112px;
    }
    .muted { color: #526276; font-size: .92rem; }
    .small-note { color: #526276; font-size: .86rem; }
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

st.warning(
    "Research prototype only. This model estimates labels learned from the supplied dataset; "
    "it does not replace ANSYS verification, engineering review, or a certified safety assessment."
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
            st.error("Predicted class: FAILURE")
            st.markdown("The model classifies this input as **Failure (1)**.")
        else:
            st.success("Predicted class: SAFE")
            st.markdown("The model classifies this input as **Safe (0)**.")

    with prob_col:
        st.metric("Model-estimated failure score", f"{failure_probability:.1%}")
        st.progress(min(max(failure_probability, 0.0), 1.0))
    with safe_col:
        st.metric("Model-estimated safe score", f"{safe_probability:.1%}")
        st.progress(min(max(safe_probability, 0.0), 1.0))

    if out_of_range:
        st.warning(
            "One or more inputs are outside the training-data range. The prediction is an extrapolation "
            "and may be unreliable:\n\n" + "\n".join(f"- {item}" for item in out_of_range)
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
