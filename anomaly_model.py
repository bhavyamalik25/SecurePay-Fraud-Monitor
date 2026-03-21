"""
anomaly_model.py
Machine Learning anomaly detection for SecurePay Fraud Monitor.
Uses Isolation Forest to detect behavioral anomalies in transaction patterns.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os
from typing import Tuple

MODEL_PATH = "data/isolation_forest.pkl"
SCALER_PATH = "data/scaler.pkl"
ENCODERS_PATH = "data/encoders.pkl"

CATEGORICAL_FEATURES = ["location", "device", "transaction_type", "merchant_category"]
NUMERIC_FEATURES = ["amount", "hour", "day_of_week", "is_weekend"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time and behavioral features from raw transactions."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["log_amount"] = np.log1p(df["amount"])
    return df


def encode_categoricals(df: pd.DataFrame, fit: bool = True, encoders: dict = None) -> Tuple:
    """Label-encode categorical columns."""
    from typing import Tuple as Tuple_
    df = df.copy()
    if fit:
        encoders = {}
        for col in CATEGORICAL_FEATURES:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in CATEGORICAL_FEATURES:
            le = encoders[col]
            df[col + "_enc"] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    return df, encoders


def prepare_features(df: pd.DataFrame, encoders: dict = None, scaler=None, fit: bool = True):
    """Full feature preparation pipeline."""
    df = engineer_features(df)
    df, encoders = encode_categoricals(df, fit=fit, encoders=encoders)

    feature_cols = (
        NUMERIC_FEATURES +
        ["log_amount"] +
        [c + "_enc" for c in CATEGORICAL_FEATURES]
    )

    X = df[feature_cols].fillna(0)

    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, encoders, scaler, feature_cols


def train_model(df: pd.DataFrame, contamination: float = 0.12) -> dict:
    """
    Train Isolation Forest on transaction data.
    contamination ~ expected fraud rate.
    """
    # Use only 'normal' behavior rows to train if labels exist
    if "is_fraud" in df.columns:
        train_df = df[df["is_fraud"] == 0].copy()
    else:
        train_df = df.copy()

    X_train, encoders, scaler, feature_cols = prepare_features(train_df, fit=True)

    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train)

    # Save artifacts
    os.makedirs("data", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(encoders, f)

    print(f"[AnomalyModel] Trained on {len(X_train)} normal samples. Model saved.")
    return {"model": model, "scaler": scaler, "encoders": encoders}


def load_model() -> dict:
    """Load pre-trained model artifacts."""
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, ENCODERS_PATH]):
        return None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)
    return {"model": model, "scaler": scaler, "encoders": encoders}


def predict_anomalies(df: pd.DataFrame, artifacts: dict = None) -> pd.DataFrame:
    """
    Run anomaly detection on transactions.
    Returns df with ml_anomaly_score and ml_is_anomaly columns.
    """
    if artifacts is None:
        artifacts = load_model()
        if artifacts is None:
            from data_generator import load_or_generate
            artifacts = train_model(df)

    model = artifacts["model"]
    scaler = artifacts["scaler"]
    encoders = artifacts["encoders"]

    X, _, _, _ = prepare_features(df, encoders=encoders, scaler=scaler, fit=False)

    # Isolation Forest: -1 = anomaly, 1 = normal
    predictions = model.predict(X)
    raw_scores = model.decision_function(X)  # more negative = more anomalous

    df = df.copy()
    df["ml_is_anomaly"] = (predictions == -1).astype(int)

    # Normalize to 0–100 anomaly score (higher = more suspicious)
    score_min, score_max = raw_scores.min(), raw_scores.max()
    if score_max != score_min:
        normalized = 1 - (raw_scores - score_min) / (score_max - score_min)
    else:
        normalized = np.zeros(len(raw_scores))
    df["ml_anomaly_score"] = np.round(normalized * 100, 1)

    return df


def get_ml_summary(df: pd.DataFrame) -> dict:
    """Return ML detection summary stats."""
    total = len(df)
    anomalies = df["ml_is_anomaly"].sum()
    return {
        "total_analyzed": total,
        "anomalies_detected": int(anomalies),
        "anomaly_rate_pct": round(anomalies / total * 100, 1) if total > 0 else 0,
        "avg_anomaly_score": round(df["ml_anomaly_score"].mean(), 1),
    }


if __name__ == "__main__":
    from data_generator import load_or_generate
    df = load_or_generate()
    artifacts = train_model(df)
    result = predict_anomalies(df, artifacts)
    print(get_ml_summary(result))
    print(result[result["ml_is_anomaly"] == 1][["transaction_id", "amount", "ml_anomaly_score"]].head(10))