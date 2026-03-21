"""
fraud_detection.py
Rule-based fraud detection engine for SecurePay Fraud Monitor.
Flags transactions based on cybersecurity heuristics and assigns risk scores.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Tuple

# --- Thresholds ---
HIGH_VALUE_THRESHOLD = 50000          # ₹50,000+
RAPID_BURST_WINDOW_MINS = 30          # Time window for burst detection
RAPID_BURST_COUNT = 3                 # Transactions within window = suspicious
SUSPICIOUS_LOCATIONS = {"Unknown", "Foreign-IP", "VPN-Detected"}
SUSPICIOUS_DEVICES = {"Jailbroken Device", "Unknown Device", "Emulator"}
SUSPICIOUS_MERCHANTS = {"Crypto Exchange", "Unknown Merchant"}

# Risk score weights
WEIGHT_HIGH_VALUE = 35
WEIGHT_SUSPICIOUS_LOCATION = 25
WEIGHT_SUSPICIOUS_DEVICE = 20
WEIGHT_RAPID_BURST = 30
WEIGHT_SUSPICIOUS_MERCHANT = 15
WEIGHT_OFF_HOURS = 10


def score_to_risk(score: int) -> str:
    """Convert numeric risk score to categorical label."""
    if score >= 60:
        return "High"
    elif score >= 30:
        return "Medium"
    else:
        return "Low"


def detect_rapid_burst(df: pd.DataFrame, user_id: str, current_ts: pd.Timestamp) -> bool:
    """Check if a user has made rapid-burst transactions."""
    window_start = current_ts - timedelta(minutes=RAPID_BURST_WINDOW_MINS)
    user_txs = df[
        (df["user_id"] == user_id) &
        (df["timestamp"] >= window_start) &
        (df["timestamp"] <= current_ts)
    ]
    return len(user_txs) >= RAPID_BURST_COUNT


def analyze_transaction(row: pd.Series, df: pd.DataFrame) -> Tuple[int, list]:
    """
    Analyze a single transaction row against rule-based heuristics.
    Returns (risk_score, list_of_triggered_rules).
    """
    score = 0
    flags = []

    # Rule 1: High-value transaction
    if row["amount"] >= HIGH_VALUE_THRESHOLD:
        score += WEIGHT_HIGH_VALUE
        flags.append(f"High-value transaction (₹{row['amount']:,.0f})")

    # Rule 2: Suspicious location
    if row.get("location") in SUSPICIOUS_LOCATIONS:
        score += WEIGHT_SUSPICIOUS_LOCATION
        flags.append(f"Suspicious location: {row['location']}")

    # Rule 3: Suspicious device
    if row.get("device") in SUSPICIOUS_DEVICES:
        score += WEIGHT_SUSPICIOUS_DEVICE
        flags.append(f"Suspicious device: {row['device']}")

    # Rule 4: Suspicious merchant category
    if row.get("merchant_category") in SUSPICIOUS_MERCHANTS:
        score += WEIGHT_SUSPICIOUS_MERCHANT
        flags.append(f"Risky merchant category: {row['merchant_category']}")

    # Rule 5: Rapid burst detection
    if detect_rapid_burst(df, row["user_id"], row["timestamp"]):
        score += WEIGHT_RAPID_BURST
        flags.append("Rapid transaction burst detected")

    # Rule 6: Off-hours transaction (midnight–5am)
    hour = pd.to_datetime(row["timestamp"]).hour
    if 0 <= hour <= 5:
        score += WEIGHT_OFF_HOURS
        flags.append(f"Off-hours transaction ({hour:02d}:00)")

    return min(score, 100), flags  # Cap at 100


def run_rule_engine(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply rule-based detection to entire DataFrame.
    Adds: rule_score, rule_risk, rule_flags columns.
    """
    scores = []
    risks = []
    flags_list = []

    for _, row in df.iterrows():
        score, flags = analyze_transaction(row, df)
        scores.append(score)
        risks.append(score_to_risk(score))
        flags_list.append(" | ".join(flags) if flags else "No anomalies detected")

    df = df.copy()
    df["rule_score"] = scores
    df["rule_risk"] = risks
    df["rule_flags"] = flags_list

    return df


def get_rule_summary(df: pd.DataFrame) -> dict:
    """Return aggregate statistics from rule-based engine."""
    total = len(df)
    high_risk = (df["rule_risk"] == "High").sum()
    medium_risk = (df["rule_risk"] == "Medium").sum()
    low_risk = (df["rule_risk"] == "Low").sum()

    return {
        "total_transactions": total,
        "high_risk_count": int(high_risk),
        "medium_risk_count": int(medium_risk),
        "low_risk_count": int(low_risk),
        "high_risk_pct": round(high_risk / total * 100, 1) if total > 0 else 0,
        "avg_risk_score": round(df["rule_score"].mean(), 1),
    }


if __name__ == "__main__":
    from data_generator import load_or_generate
    df = load_or_generate()
    result = run_rule_engine(df)
    summary = get_rule_summary(result)
    print(summary)
    print(result[result["rule_risk"] == "High"][["transaction_id", "user_id", "amount", "rule_score", "rule_flags"]].head(10))