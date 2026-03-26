PMSecurePay Fraud Monitor   

What It Does
A real-time financial fraud detection system that analyzes every transaction through two engines simultaneously — a handcrafted rule engine and a machine learning model — blends both scores, and surfaces high-risk activity on a live dashboard. Think of it as a lightweight SIEM built specifically for payment fraud.

Features
Detection

Rule-based engine with 6 fraud heuristics — high-value transactions (₹50K+), VPN/foreign IP detection, suspicious device flagging (jailbroken, emulator), risky merchant categories (crypto exchanges), rapid burst detection (3+ transactions in 30 min), off-hours activity (midnight–5am)
Isolation Forest ML model trained on normal user behavior — automatically detects anomalies it was never explicitly told to look for
Combined risk score (rule × 60% + ML × 40%) mapped to Low / Medium / High

Dashboard

30-day transaction volume timeline broken down by risk level
Risk distribution donut chart
Transaction amount distribution by risk
Top flagged users leaderboard
Hourly activity heatmap — shows when fraud spikes
Rule score vs ML score scatter plot — visualizes how both engines agree or disagree

Alerts

Live High Risk and Medium Risk alert feed with full context — amount, location, device, timestamp, exact flags triggered, and both scores per transaction

Explorer

Searchable, sortable full transaction table with color-coded risk levels

Reporting

Auto-exports fraud log to CSV on every run
One-click downloadable audit report with aggregate statistics

Controls

Filter by risk level and transaction amount
Regenerate dataset with one click
Retrain the ML model with one click


Tech Stack
LanguagePythonDashboardStreamlitML ModelScikit-learn — Isolation ForestDataPandas, NumPyVisualizationPlotlySerializationPickleOtherUUID, OS, Datetime

Why It's a Strong Real-World Project
It solves a real problem. Payment fraud costs the global financial industry over $40 billion annually. Every bank, fintech, and payment gateway runs some version of what this app does.
It uses two detection approaches together. Rule engines catch known fraud patterns. ML catches unknown ones. Using both and blending the scores is exactly how production fraud systems at companies like Razorpay, Stripe, and PayPal actually work.
It's not just a model. Most ML student projects stop at training a model and printing accuracy. This goes further — feature engineering, a full UI, live alerting, exportable reports, and user-facing controls. It looks and works like a real product.
The ML is unsupervised. Isolation Forest doesn't need labeled fraud data to train — it learns what "normal" looks like and flags deviations. This is a more realistic and harder approach than supervised classification, and it demonstrates genuine understanding of when to use which type of ML.
End-to-end. Data generation → feature engineering → rule engine → ML model → blended scoring → dashboard → alerts → export. Every layer of a real system is present.

Who Would Actually Use Something Like This

Fintech startups monitoring UPI/payment transactions
Banking fraud analyst teams
Internal security teams at e-commerce companies
Any business processing high transaction volumes that needs automated monitoring without a full enterprise solution