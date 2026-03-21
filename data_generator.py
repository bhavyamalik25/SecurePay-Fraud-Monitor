"""
data_generator.py
Simulates realistic financial transactions for SecurePay Fraud Monitor.
"""

import pandas as pd
import numpy as np
import random
import uuid
from datetime import datetime, timedelta

# --- Seed for reproducibility ---
random.seed(42)
np.random.seed(42)

LOCATIONS = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Surat",
    "Unknown", "Foreign-IP", "VPN-Detected"
]

DEVICES = [
    "iPhone 14", "Samsung Galaxy S23", "MacBook Pro", "Windows PC",
    "iPad Air", "OnePlus 11", "Pixel 7", "Unknown Device",
    "Jailbroken Device", "Emulator"
]

TRANSACTION_TYPES = ["UPI Transfer", "NEFT", "IMPS", "Card Payment", "Wallet Transfer", "RTGS"]

MERCHANT_CATEGORIES = [
    "Groceries", "Electronics", "Travel", "Dining", "Entertainment",
    "Medical", "Utilities", "Luxury Goods", "Crypto Exchange", "Unknown Merchant"
]

USER_IDS = [f"USR{str(i).zfill(4)}" for i in range(1, 51)]  # 50 simulated users


def generate_normal_transaction(user_id: str, timestamp: datetime) -> dict:
    """Generate a normal, low-risk transaction."""
    return {
        "transaction_id": str(uuid.uuid4())[:12].upper(),
        "user_id": user_id,
        "amount": round(random.uniform(100, 15000), 2),
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "location": random.choice(LOCATIONS[:10]),  # only real cities
        "device": random.choice(DEVICES[:7]),       # only normal devices
        "transaction_type": random.choice(TRANSACTION_TYPES),
        "merchant_category": random.choice(MERCHANT_CATEGORIES[:8]),
        "is_fraud": 0
    }


def generate_fraud_transaction(user_id: str, timestamp: datetime, fraud_type: str = "random") -> dict:
    """Generate a suspicious/fraudulent transaction."""
    fraud_type = fraud_type if fraud_type != "random" else random.choice(
        ["high_value", "suspicious_location", "suspicious_device", "unusual_merchant"]
    )

    tx = {
        "transaction_id": str(uuid.uuid4())[:12].upper(),
        "user_id": user_id,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "transaction_type": random.choice(TRANSACTION_TYPES),
        "is_fraud": 1
    }

    if fraud_type == "high_value":
        tx["amount"] = round(random.uniform(55000, 500000), 2)
        tx["location"] = random.choice(LOCATIONS[:10])
        tx["device"] = random.choice(DEVICES[:7])
        tx["merchant_category"] = random.choice(MERCHANT_CATEGORIES[:8])

    elif fraud_type == "suspicious_location":
        tx["amount"] = round(random.uniform(5000, 80000), 2)
        tx["location"] = random.choice(["Unknown", "Foreign-IP", "VPN-Detected"])
        tx["device"] = random.choice(DEVICES[:7])
        tx["merchant_category"] = random.choice(MERCHANT_CATEGORIES)

    elif fraud_type == "suspicious_device":
        tx["amount"] = round(random.uniform(2000, 60000), 2)
        tx["location"] = random.choice(LOCATIONS[:10])
        tx["device"] = random.choice(["Jailbroken Device", "Unknown Device", "Emulator"])
        tx["merchant_category"] = random.choice(MERCHANT_CATEGORIES)

    elif fraud_type == "unusual_merchant":
        tx["amount"] = round(random.uniform(10000, 200000), 2)
        tx["location"] = random.choice(LOCATIONS[:10])
        tx["device"] = random.choice(DEVICES[:8])
        tx["merchant_category"] = random.choice(["Crypto Exchange", "Unknown Merchant"])

    return tx


def generate_rapid_burst(user_id: str, base_time: datetime, n: int = 5) -> list:
    """Simulate rapid burst transactions from a single user."""
    txs = []
    for i in range(n):
        ts = base_time + timedelta(minutes=random.randint(0, 10))
        tx = generate_fraud_transaction(user_id, ts, "high_value")
        txs.append(tx)
    return txs


def generate_dataset(n_normal: int = 800, n_fraud: int = 200, save: bool = True) -> pd.DataFrame:
    """Generate a full mixed dataset of normal + fraudulent transactions."""
    records = []
    base_date = datetime.now() - timedelta(days=30)

    # Normal transactions
    for _ in range(n_normal):
        user = random.choice(USER_IDS)
        ts = base_date + timedelta(
            days=random.randint(0, 29),
            hours=random.randint(6, 23),
            minutes=random.randint(0, 59)
        )
        records.append(generate_normal_transaction(user, ts))

    # Fraud transactions
    fraud_users = random.sample(USER_IDS, min(20, len(USER_IDS)))
    for i in range(n_fraud):
        user = random.choice(fraud_users)
        ts = base_date + timedelta(
            days=random.randint(0, 29),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        records.append(generate_fraud_transaction(user, ts))

    # Add 3 rapid burst events
    for _ in range(3):
        burst_user = random.choice(fraud_users)
        burst_time = base_date + timedelta(days=random.randint(0, 29), hours=random.randint(1, 22))
        records.extend(generate_rapid_burst(burst_user, burst_time, n=random.randint(4, 7)))

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    if save:
        import os
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/transactions.csv", index=False)
        print(f"[DataGenerator] Generated {len(df)} transactions → data/transactions.csv")

    return df


def load_or_generate() -> pd.DataFrame:
    """Load existing dataset or generate new one."""
    import os
    path = "data/transactions.csv"
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, parse_dates=["timestamp"])
            if df.empty or len(df.columns) == 0:
                raise ValueError("Empty file")
            return df
        except Exception:
            os.remove(path)
    return generate_dataset()


if __name__ == "__main__":
    df = generate_dataset()
    print(df.head())
    print(f"\nFraud: {df['is_fraud'].sum()} | Safe: {(df['is_fraud']==0).sum()}")