# SecurePay Fraud Monitor

A real-time financial fraud detection system that analyzes transactions through a hybrid rule-based engine and machine learning model, blends both scores, and surfaces high-risk activity on a live dashboard.

---

## What It Does

- Analyzes every transaction through two engines simultaneously — a handcrafted rule engine and an Isolation Forest ML model
- Blends both scores (rule x 60% + ML x 40%) into a single Low / Medium / High risk rating
- Surfaces flagged transactions instantly on a live dashboard with full context per alert
- Exports audit-ready fraud reports with one click

---

## Features

**Detection**
- 6 rule-based fraud heuristics — high-value transactions (50,000+), VPN/foreign IP detection, suspicious device flagging (jailbroken, emulator), risky merchant categories (crypto exchanges), rapid burst detection (3+ transactions in 30 min), off-hours activity (midnight-5am)
- Isolation Forest ML model trained on normal user behavior — detects anomalies it was never explicitly told to look for
- Combined risk score mapped to Low / Medium / High

**Dashboard**
- 30-day transaction volume timeline by risk level
- Risk distribution chart
- Transaction amount distribution by risk
- Top flagged users leaderboard
- Hourly activity heatmap
- Rule score vs ML score scatter plot

**Alerts**
- Live High Risk and Medium Risk alert feed with amount, location, device, timestamp, triggered flags, and both scores per transaction

**Explorer**
- Searchable and sortable full transaction table with color-coded risk levels

**Reporting**
- Auto-exports fraud log to CSV on every run
- Downloadable audit report with aggregate statistics

**Controls**
- Filter by risk level and transaction amount
- Regenerate dataset with one click
- Retrain the ML model with one click

---

## Tech Stack

- Language — Python
- Dashboard — Streamlit
- ML Model — Scikit-learn (Isolation Forest)
- Data — Pandas, NumPy
- Visualization — Plotly
- Other — Pickle, UUID, Datetime

---

## Why This Project Matters

- Payment fraud costs the global financial industry over $40 billion annually — every bank, fintech, and payment gateway runs some version of what this system does
- Combines rule-based and ML detection the same way production fraud systems at companies like Stripe, Razorpay, and PayPal actually work
- Uses unsupervised ML — Isolation Forest learns what normal looks like and flags deviations without needing labeled fraud data, which is a more realistic approach than supervised classification
- End-to-end system — data generation, feature engineering, rule engine, ML model, blended scoring, dashboard, alerts, and export — every layer of a real product is present

---

## Who Uses Systems Like This

- Fintech startups monitoring payment transactions
- Banking fraud analyst teams
- Internal security teams at e-commerce companies
- Any business processing high transaction volumes that needs automated monitoring

---

## File Structure

```
SecurePay_Fraud_Monitor/
├── app.py                  # Streamlit dashboard
├── fraud_detection.py      # Rule-based detection engine
├── anomaly_model.py        # Isolation Forest ML model
├── data_generator.py       # Synthetic transaction generator
├── data/                   # Auto-generated dataset and model files
├── logs/                   # Exported fraud logs
├── requirements.txt
└── README.md
```

---

## Getting Started

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app generates a transaction dataset and trains the ML model automatically on first launch.

---

## Deployment

1. Push to GitHub
2. Go to share.streamlit.io
3. Connect your repository
4. Set main file path to app.py
5. Deploy

---

## Tech Concepts Demonstrated

- Cybersecurity fraud detection logic
- Unsupervised machine learning and anomaly detection
- Feature engineering from transactional data
- Real-time data processing and visualization
- Dashboard UI design for financial and security systems