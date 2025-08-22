import logging
from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

BASE = Path(".")
# Load artifacts
with open(BASE / "./data/ml/model.pkl", "rb") as f:
    model = pickle.load(f)
with open(BASE / "./data/ml/encoder_role.pkl", "rb") as f:
    enc_role = pickle.load(f)
with open(BASE / "./data/ml/encoder_app.pkl", "rb") as f:
    enc_app = pickle.load(f)
with open(BASE / "./data/ml/columns_order.pkl", "rb") as f:
    columns_order = pickle.load(f)

NUMERIC_FEATS = ["gpu_request", "disk_request", "memory_request", "max_instance_per_node"]
CATEGORICAL_FEATS = ["role", "app_name"]

app = Flask(__name__)

def safe_number(v):
    try:
        if v is None:
            return 0.0
        if isinstance(v, (int, float, np.integer, np.floating)):
            if np.isnan(v) or np.isinf(v):
                return 0.0
            return float(v)
        return float(v)
    except Exception:
        return 0.0

def preprocess_input(payload):
    df = pd.DataFrame([payload]) if isinstance(payload, dict) else pd.DataFrame(payload)

    # Ensure numeric columns exist & sanitize
    for col in NUMERIC_FEATS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].apply(safe_number)

    # Ensure categorical columns exist & sanitize
    for c in CATEGORICAL_FEATS:
        if c not in df.columns:
            df[c] = "UNKNOWN"
        else:
            df[c] = df[c].fillna("UNKNOWN").astype(str)

    # One-hot encoding
    role_ohe = enc_role.transform(df[["role"]])
    app_ohe = enc_app.transform(df[["app_name"]])
    role_cols = enc_role.get_feature_names_out(["role"]).tolist()
    app_cols = enc_app.get_feature_names_out(["app_name"]).tolist()

    X_num = df[NUMERIC_FEATS].reset_index(drop=True)
    X_final = pd.concat(
        [X_num,
         pd.DataFrame(role_ohe, columns=role_cols),
         pd.DataFrame(app_ohe, columns=app_cols)],
        axis=1
    )

    # Align with training columns
    X_final = X_final.reindex(columns=columns_order, fill_value=0)
    return X_final

# ---- ROUTES ----

@app.before_request
def log_request_info():
    logger.info(f"API hit: {request.method} {request.path} | Remote: {request.remote_addr}")

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Welcome to the CPU request prediction API",
        "routes": {
            "health": "/health",
            "predict": "/predict (POST)"
        },
        "sample_payload": {
            "gpu_request": 0.0,
            "disk_request": 500.0,
            "memory_request": 64.0,
            "max_instance_per_node": 2,
            "role": "worker",
            "app_name": "app1"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        logger.info(f"/predict payload: {payload}")

        X = preprocess_input(payload)
        preds = model.predict(X)
        preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)

        if isinstance(payload, dict):
            return jsonify({"cpu_pred": float(preds[0])})
        else:
            return jsonify({"cpu_pred": [float(x) for x in preds]})
    except Exception as e:
        logger.error(f"Error in /predict: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
