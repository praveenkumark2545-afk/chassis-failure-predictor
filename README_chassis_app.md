# Chassis Failure Prediction Web App

A Streamlit web interface that uses a Random Forest classifier trained on the supplied ANSYS chassis simulation dataset.

## Files
- `app.py`: web interface
- `random_forest_chassis_model.joblib`: trained Random Forest model
- `model_metadata.json`: model information, feature ranges, and holdout metrics
- `requirements.txt`: Python dependencies

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
4. Start the website:
   ```bash
   streamlit run app.py
   ```
5. Open the local URL printed by Streamlit (usually `http://localhost:8501`).

## Model setup
- Inputs: `Tube_OD_mm`, `Thickness_mm`, `Load_N`
- Target: `Failure_Label` (`0 = Safe`, `1 = Failure`)
- Training rows: 200
- Algorithm: Random Forest, 300 trees, `class_weight="balanced"`, `random_state=42`

The model artifact was refit on all valid rows in the supplied dataset after computing a separate stratified 80/20 holdout reference. The holdout metrics are stored in `model_metadata.json`.

## Important limitation
This is a research prototype trained on a small simulation dataset. It is not a substitute for ANSYS analysis, engineering judgment, independent validation, or a certified structural safety assessment. A model score is not a guarantee of real-world safety.
