"""
app.py
SecurePay Fraud Monitor — Main Streamlit Dashboard
Combines rule-based detection + ML anomaly detection into a unified UI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- Internal Modules ---
from data_generator import load_or_generate, generate_dataset
from fraud_detection import run_rule_engine, get_rule_summary
from anomaly_model import train_model, predict_anomalies, get_ml_summary, load_model

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SecurePay Fraud Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# COLOR PALETTE — "Soft Corporate Luxe"
# ─────────────────────────────────────────────
PALETTE = {
    "bg": "#FAFAF7",
    "card": "#FFFFFF",
    "gray_light": "#EDEDED",
    "slate": "#6B7280",
    "navy": "#2F3E46",
    "dusty_blue": "#A7B0B8",
    "sage": "#B7C4B1",
    "red": "#D96C6C",
    "amber": "#E6A23C",
    "green": "#7FAF8E",
    "border": "#E2E4E7",
}

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    background-color: {PALETTE['bg']};
    color: {PALETTE['navy']};
  }}

  .stApp {{
    background-color: {PALETTE['bg']};
  }}

  /* Sidebar */
  section[data-testid="stSidebar"] {{
    background-color: #EDE8E3 !important;
    border-right: 1px solid #D9D2CB;
  }}
  section[data-testid="stSidebar"] * {{
    color: {PALETTE['navy']} !important;
  }}
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stSlider label {{
    color: {PALETTE['slate']} !important;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }}

  /* Sidebar buttons - visible text, no weird hover */
  section[data-testid="stSidebar"] .stButton > button {{
    background-color: #F5F0EB !important;
    color: {PALETTE['navy']} !important;
    border: 1px solid #C8BFB6 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    transition: background-color 0.15s ease !important;
  }}
  section[data-testid="stSidebar"] .stButton > button:hover {{
    background-color: #EAE3DC !important;
    color: {PALETTE['navy']} !important;
    border-color: #B8AFA6 !important;
  }}

  /* Cards */
  .metric-card {{
    background: {PALETTE['card']};
    border: 1px solid {PALETTE['border']};
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 1px 6px rgba(47,62,70,0.06);
    transition: box-shadow 0.2s ease;
  }}
  .metric-card:hover {{
    box-shadow: 0 4px 16px rgba(47,62,70,0.12);
  }}
  .metric-label {{
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {PALETTE['slate']};
    margin-bottom: 6px;
  }}
  .metric-value {{
    font-size: 2rem;
    font-weight: 700;
    color: {PALETTE['navy']};
    line-height: 1.1;
  }}
  .metric-sub {{
    font-size: 0.75rem;
    color: {PALETTE['dusty_blue']};
    margin-top: 4px;
  }}

  /* Risk badges */
  .badge-high {{
    background: rgba(217,108,108,0.12);
    color: {PALETTE['red']};
    border: 1px solid rgba(217,108,108,0.3);
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
  }}
  .badge-medium {{
    background: rgba(230,162,60,0.12);
    color: {PALETTE['amber']};
    border: 1px solid rgba(230,162,60,0.3);
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
  }}
  .badge-low {{
    background: rgba(127,175,142,0.12);
    color: {PALETTE['green']};
    border: 1px solid rgba(127,175,142,0.3);
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
  }}

  /* Alert banners */
  .alert-high {{
    background: rgba(217,108,108,0.08);
    border-left: 3px solid {PALETTE['red']};
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin-bottom: 8px;
    font-size: 0.85rem;
  }}
  .alert-medium {{
    background: rgba(230,162,60,0.08);
    border-left: 3px solid {PALETTE['amber']};
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin-bottom: 8px;
    font-size: 0.85rem;
  }}

  /* Section headers */
  .section-header {{
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {PALETTE['slate']};
    margin-bottom: 14px;
    padding-bottom: 6px;
    border-bottom: 1px solid {PALETTE['border']};
  }}

  /* Page title */
  .page-title {{
    font-size: 1.5rem;
    font-weight: 700;
    color: {PALETTE['navy']};
    letter-spacing: -0.02em;
  }}
  .page-subtitle {{
    font-size: 0.82rem;
    color: {PALETTE['slate']};
    margin-top: 2px;
  }}

  /* Mono font for IDs */
  .mono {{
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
  }}

  /* Divider */
  hr {{
    border: none;
    border-top: 1px solid {PALETTE['border']};
    margin: 18px 0;
  }}

  /* Hide streamlit default branding but keep sidebar toggle */
  #MainMenu {{ visibility: hidden; }}
  footer {{ visibility: hidden; }}
  header[data-testid="stHeader"] {{ background: transparent; }}
  [data-testid="stToolbar"] {{ display: none; }}
  [data-testid="stDecoration"] {{ display: none; }}

  /* Dataframe styling */
  .stDataFrame {{
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid {PALETTE['border']};
  }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    background: {PALETTE['gray_light']};
    padding: 4px;
    border-radius: 10px;
  }}
  .stTabs [data-baseweb="tab"] {{
    border-radius: 7px;
    padding: 6px 16px;
    font-size: 0.82rem;
    font-weight: 500;
    color: #1a1a1a !important;
    background: transparent !important;
  }}
  .stTabs [data-baseweb="tab"]:hover {{
    background: transparent !important;
    color: #1a1a1a !important;
  }}
  .stTabs [aria-selected="true"] {{
    background: white !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    color: #1a1a1a !important;
  }}
  .stTabs [data-baseweb="tab"] p,
  .stTabs [data-baseweb="tab"] span {{
    color: #1a1a1a !important;
  }}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def risk_badge(risk: str) -> str:
    cls = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}.get(risk, "badge-low")
    icon = {"High": "🔴", "Medium": "⚠️", "Low": "✅"}.get(risk, "")
    return f'<span class="{cls}">{icon} {risk}</span>'


def format_inr(amount: float) -> str:
    return f"₹{amount:,.0f}"


@st.cache_data(ttl=300, show_spinner=False)
def load_and_analyze():
    """Load data, run both engines, return enriched DataFrame."""
    df = load_or_generate()
    df = run_rule_engine(df)

    artifacts = load_model()
    if artifacts is None:
        artifacts = train_model(df)

    df = predict_anomalies(df, artifacts)

    # Combined final risk
    def final_risk(row):
        if row["rule_risk"] == "High" or row["ml_anomaly_score"] >= 75:
            return "High"
        elif row["rule_risk"] == "Medium" or row["ml_anomaly_score"] >= 45:
            return "Medium"
        else:
            return "Low"

    df["final_risk"] = df.apply(final_risk, axis=1)
    df["combined_score"] = np.round((df["rule_score"] * 0.6 + df["ml_anomaly_score"] * 0.4), 1)

    return df


def save_fraud_log(df: pd.DataFrame):
    """Export flagged transactions to CSV."""
    os.makedirs("logs", exist_ok=True)
    flagged = df[df["final_risk"].isin(["High", "Medium"])].copy()
    path = "logs/fraud_logs.csv"
    flagged.to_csv(path, index=False)
    return path, flagged


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 16px 0 24px 0;">
        <div style="font-size: 1.15rem; font-weight: 700; color: #2F3E46; letter-spacing: -0.01em;">
            🛡️ SecurePay
        </div>
        <div style="font-size: 0.72rem; color: #6B7280; margin-top: 2px; letter-spacing: 0.06em;">
            FRAUD MONITOR v1.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div style="font-size:0.7rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#6B7280;margin-bottom:10px;">Filters</div>', unsafe_allow_html=True)

    risk_filter = st.multiselect(
        "Risk Level",
        ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"]
    )

    amount_range = st.slider(
        "Transaction Amount (₹)",
        min_value=0,
        max_value=500000,
        value=(0, 500000),
        step=5000,
        format="₹%d"
    )

    st.markdown("---")
    st.markdown('<div style="font-size:0.7rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#6B7280;margin-bottom:10px;">Actions</div>', unsafe_allow_html=True)

    if st.button("Regenerate Dataset", use_container_width=True):
        st.cache_data.clear()
        generate_dataset(save=True)
        st.rerun()

    if st.button("Retrain ML Model", use_container_width=True):
        st.cache_data.clear()
        df_temp = load_or_generate()
        with st.spinner("Training Isolation Forest…"):
            train_model(df_temp)
        st.success("Model retrained!")
        st.rerun()

    st.markdown("---")
    st.markdown('<div style="font-size:0.68rem;color:#6B7280;line-height:1.6;">Rule Engine + Isolation Forest<br>Behavioral Anomaly Detection<br>Real-Time Risk Scoring</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
with st.spinner("Loading transaction data…"):
    df_full = load_and_analyze()

# Apply filters
df = df_full[
    (df_full["final_risk"].isin(risk_filter)) &
    (df_full["amount"] >= amount_range[0]) &
    (df_full["amount"] <= amount_range[1])
].copy()


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
col_title, col_time = st.columns([3, 1])
with col_title:
    st.markdown(f"""
    <div class="page-title">SecurePay Fraud Monitor</div>
    <div class="page-subtitle">Real-time transaction analysis · Rule Engine + ML Anomaly Detection</div>
    """, unsafe_allow_html=True)
with col_time:
    st.markdown(f"""
    <div style="text-align:right;padding-top:8px;">
        <div style="font-size:0.7rem;color:{PALETTE['slate']};letter-spacing:0.08em;text-transform:uppercase;">Last Updated</div>
        <div style="font-size:0.85rem;font-weight:600;color:{PALETTE['navy']};">{datetime.now().strftime('%d %b %Y · %H:%M')}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# KPI METRICS ROW
# ─────────────────────────────────────────────
total = len(df_full)
high_risk = (df_full["final_risk"] == "High").sum()
medium_risk = (df_full["final_risk"] == "Medium").sum()
low_risk = (df_full["final_risk"] == "Low").sum()
total_flagged = high_risk + medium_risk
total_amount = df_full["amount"].sum()
avg_score = df_full["combined_score"].mean()

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Transactions</div>
        <div class="metric-value">{total:,}</div>
        <div class="metric-sub">Across all users</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🔴 High Risk</div>
        <div class="metric-value" style="color:{PALETTE['red']};">{high_risk}</div>
        <div class="metric-sub">{round(high_risk/total*100,1)}% of total</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">⚠️ Medium Risk</div>
        <div class="metric-value" style="color:{PALETTE['amber']};">{medium_risk}</div>
        <div class="metric-sub">{round(medium_risk/total*100,1)}% of total</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">✅ Safe</div>
        <div class="metric-value" style="color:{PALETTE['green']};">{low_risk}</div>
        <div class="metric-sub">{round(low_risk/total*100,1)}% of total</div>
    </div>""", unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Volume</div>
        <div class="metric-value" style="font-size:1.5rem;">₹{total_amount/1e6:.1f}M</div>
        <div class="metric-sub">Avg score: {avg_score:.0f}/100</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Dashboard",
    "Live Alerts",
    "Transaction Explorer",
    "Reports & Export"
])


# ══════════════════════════════════════════════
# TAB 1: DASHBOARD
# ══════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        # Transaction timeline
        st.markdown('<div class="section-header">Transaction Volume · 30-Day Timeline</div>', unsafe_allow_html=True)

        daily = df_full.copy()
        daily["date"] = pd.to_datetime(daily["timestamp"]).dt.date
        daily_counts = daily.groupby(["date", "final_risk"]).size().unstack(fill_value=0)
        for col in ["High", "Medium", "Low"]:
            if col not in daily_counts.columns:
                daily_counts[col] = 0

        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Scatter(
            x=daily_counts.index, y=daily_counts["Low"],
            name="Safe", fill='tozeroy',
            line=dict(color=PALETTE["green"], width=2),
            fillcolor="rgba(127,175,142,0.15)"
        ))
        fig_timeline.add_trace(go.Scatter(
            x=daily_counts.index, y=daily_counts["Medium"],
            name="Medium Risk", fill='tozeroy',
            line=dict(color=PALETTE["amber"], width=2),
            fillcolor="rgba(230,162,60,0.2)"
        ))
        fig_timeline.add_trace(go.Scatter(
            x=daily_counts.index, y=daily_counts["High"],
            name="High Risk", fill='tozeroy',
            line=dict(color=PALETTE["red"], width=2),
            fillcolor="rgba(217,108,108,0.2)"
        ))
        fig_timeline.update_layout(
            height=260,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(orientation="h", y=-0.2, x=0, font=dict(size=11)),
            xaxis=dict(showgrid=False, tickfont=dict(size=10, color=PALETTE["slate"])),
            yaxis=dict(showgrid=True, gridcolor=PALETTE["border"], tickfont=dict(size=10, color=PALETTE["slate"])),
            hovermode="x unified"
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

        # Amount distribution by risk
        st.markdown('<div class="section-header">Transaction Amount Distribution</div>', unsafe_allow_html=True)
        fig_dist = go.Figure()
        for risk, color in [("Low", PALETTE["green"]), ("Medium", PALETTE["amber"]), ("High", PALETTE["red"])]:
            subset = df_full[df_full["final_risk"] == risk]["amount"]
            fig_dist.add_trace(go.Histogram(
                x=subset, name=risk,
                marker_color=color,
                opacity=0.75,
                nbinsx=40,
                histnorm="probability density"
            ))
        fig_dist.update_layout(
            height=230,
            barmode="overlay",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(orientation="h", y=-0.25, font=dict(size=11)),
            xaxis=dict(title="Amount (₹)", showgrid=False, tickfont=dict(size=10)),
            yaxis=dict(showgrid=True, gridcolor=PALETTE["border"], tickfont=dict(size=10)),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_right:
        # Donut chart
        st.markdown('<div class="section-header">Risk Distribution</div>', unsafe_allow_html=True)
        fig_donut = go.Figure(go.Pie(
            labels=["High Risk", "Medium Risk", "Safe"],
            values=[high_risk, medium_risk, low_risk],
            hole=0.65,
            marker=dict(colors=[PALETTE["red"], PALETTE["amber"], PALETTE["green"]]),
            textinfo="percent",
            textfont=dict(size=12),
            hovertemplate="<b>%{label}</b><br>%{value} transactions<br>%{percent}<extra></extra>"
        ))
        fig_donut.update_layout(
            height=240,
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(orientation="v", x=0.75, y=0.5, font=dict(size=11)),
            annotations=[dict(text=f"<b>{total}</b><br><span style='font-size:10px'>Total</span>",
                              x=0.5, y=0.5, font=dict(size=14, color=PALETTE["navy"]),
                              showarrow=False)]
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        # Top risky users
        st.markdown('<div class="section-header">Top Flagged Users</div>', unsafe_allow_html=True)
        top_users = (df_full[df_full["final_risk"].isin(["High", "Medium"])]
                     .groupby("user_id")
                     .agg(flags=("transaction_id", "count"), avg_score=("combined_score", "mean"))
                     .sort_values("flags", ascending=False)
                     .head(6)
                     .reset_index())

        fig_users = go.Figure(go.Bar(
            y=top_users["user_id"],
            x=top_users["flags"],
            orientation="h",
            marker=dict(
                color=top_users["avg_score"],
                colorscale=[[0, PALETTE["amber"]], [1, PALETTE["red"]]],
                showscale=False
            ),
            text=top_users["flags"].astype(str),
            textposition="outside",
            textfont=dict(size=11)
        ))
        fig_users.update_layout(
            height=230,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=20, t=0, b=0),
            xaxis=dict(showgrid=True, gridcolor=PALETTE["border"], tickfont=dict(size=10)),
            yaxis=dict(showgrid=False, tickfont=dict(size=10, family="DM Mono")),
        )
        st.plotly_chart(fig_users, use_container_width=True)

        # Hourly heatmap
        st.markdown('<div class="section-header">Activity by Hour</div>', unsafe_allow_html=True)
        df_full["hour"] = pd.to_datetime(df_full["timestamp"]).dt.hour
        hourly = df_full.groupby("hour")["final_risk"].value_counts().unstack(fill_value=0)

        fig_hour = go.Figure()
        for risk, color in [("High", PALETTE["red"]), ("Medium", PALETTE["amber"]), ("Low", PALETTE["green"])]:
            if risk in hourly.columns:
                fig_hour.add_trace(go.Bar(
                    x=hourly.index, y=hourly[risk],
                    name=risk, marker_color=color, opacity=0.85
                ))
        fig_hour.update_layout(
            height=200, barmode="stack",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(orientation="h", y=-0.3, font=dict(size=10)),
            xaxis=dict(title="Hour of Day", showgrid=False, tickfont=dict(size=9)),
            yaxis=dict(showgrid=True, gridcolor=PALETTE["border"], tickfont=dict(size=9)),
        )
        st.plotly_chart(fig_hour, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2: LIVE ALERTS
# ══════════════════════════════════════════════
with tab2:
    col_alerts, col_info = st.columns([2, 1])

    with col_alerts:
        st.markdown('<div class="section-header">🔴 High Risk Alerts</div>', unsafe_allow_html=True)
        high_df = df_full[df_full["final_risk"] == "High"].sort_values("combined_score", ascending=False).head(20)

        if len(high_df) == 0:
            st.info("No high-risk transactions detected.")
        else:
            for _, row in high_df.iterrows():
                ts = pd.to_datetime(row["timestamp"]).strftime("%d %b · %H:%M")
                st.markdown(f"""
                <div class="alert-high">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <span class="mono">{row['transaction_id']}</span>
                            <span style="margin:0 8px;color:{PALETTE['slate']};">·</span>
                            <span style="font-weight:600;">{row['user_id']}</span>
                        </div>
                        <div style="font-weight:700;color:{PALETTE['red']};">{format_inr(row['amount'])}</div>
                    </div>
                    <div style="margin-top:6px;font-size:0.78rem;color:{PALETTE['slate']};">
                        📍 {row['location']} &nbsp;·&nbsp; 📱 {row['device']} &nbsp;·&nbsp; 🕐 {ts}
                    </div>
                    <div style="margin-top:4px;font-size:0.75rem;color:{PALETTE['red']};">
                        ⚡ {row['rule_flags'][:120]}{'…' if len(str(row['rule_flags'])) > 120 else ''}
                    </div>
                    <div style="margin-top:4px;font-size:0.75rem;color:{PALETTE['slate']};">
                        Risk Score: <b style="color:{PALETTE['red']};">{int(row['combined_score'])}/100</b>
                        &nbsp;·&nbsp; ML Anomaly: <b>{int(row['ml_anomaly_score'])}/100</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">⚠️ Medium Risk Alerts</div>', unsafe_allow_html=True)
        med_df = df_full[df_full["final_risk"] == "Medium"].sort_values("combined_score", ascending=False).head(15)

        for _, row in med_df.iterrows():
            ts = pd.to_datetime(row["timestamp"]).strftime("%d %b · %H:%M")
            st.markdown(f"""
            <div class="alert-medium">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <span class="mono">{row['transaction_id']}</span>
                        <span style="margin:0 8px;color:{PALETTE['slate']};">·</span>
                        <span style="font-weight:600;">{row['user_id']}</span>
                    </div>
                    <div style="font-weight:700;color:{PALETTE['amber']};">{format_inr(row['amount'])}</div>
                </div>
                <div style="margin-top:6px;font-size:0.78rem;color:{PALETTE['slate']};">
                    📍 {row['location']} &nbsp;·&nbsp; 📱 {row['device']} &nbsp;·&nbsp; 🕐 {ts}
                </div>
                <div style="margin-top:4px;font-size:0.75rem;color:{PALETTE['amber']};">
                    ⚡ {str(row['rule_flags'])[:100]}{'…' if len(str(row['rule_flags'])) > 100 else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_info:
        st.markdown('<div class="section-header">Detection Summary</div>', unsafe_allow_html=True)

        rule_sum = get_rule_summary(df_full)
        ml_sum = get_ml_summary(df_full)

        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:12px;">
            <div class="metric-label">Rule Engine</div>
            <div style="margin-top:8px;">
                <div style="display:flex;justify-content:space-between;font-size:0.82rem;padding:5px 0;border-bottom:1px solid {PALETTE['border']};">
                    <span>High Risk</span><span style="font-weight:600;color:{PALETTE['red']};">{rule_sum['high_risk_count']}</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.82rem;padding:5px 0;border-bottom:1px solid {PALETTE['border']};">
                    <span>Medium Risk</span><span style="font-weight:600;color:{PALETTE['amber']};">{rule_sum['medium_risk_count']}</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.82rem;padding:5px 0;">
                    <span>Avg Score</span><span style="font-weight:600;">{rule_sum['avg_risk_score']}/100</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ML Anomaly Detection</div>
            <div style="margin-top:8px;">
                <div style="display:flex;justify-content:space-between;font-size:0.82rem;padding:5px 0;border-bottom:1px solid {PALETTE['border']};">
                    <span>Anomalies Found</span><span style="font-weight:600;color:{PALETTE['red']};">{ml_sum['anomalies_detected']}</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.82rem;padding:5px 0;border-bottom:1px solid {PALETTE['border']};">
                    <span>Anomaly Rate</span><span style="font-weight:600;">{ml_sum['anomaly_rate_pct']}%</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.82rem;padding:5px 0;">
                    <span>Avg Score</span><span style="font-weight:600;">{ml_sum['avg_anomaly_score']}/100</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ML Scatter
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Rule vs ML Score</div>', unsafe_allow_html=True)
        fig_scatter = px.scatter(
            df_full.sample(min(300, len(df_full))),
            x="rule_score", y="ml_anomaly_score",
            color="final_risk",
            color_discrete_map={"High": PALETTE["red"], "Medium": PALETTE["amber"], "Low": PALETTE["green"]},
            opacity=0.7,
            size_max=8
        )
        fig_scatter.update_layout(
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(title="", orientation="h", y=-0.2, font=dict(size=10)),
            xaxis=dict(title="Rule Score", showgrid=True, gridcolor=PALETTE["border"], tickfont=dict(size=9)),
            yaxis=dict(title="ML Score", showgrid=True, gridcolor=PALETTE["border"], tickfont=dict(size=9)),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3: TRANSACTION EXPLORER
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Transaction Table</div>', unsafe_allow_html=True)

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        search_user = st.text_input("Search User ID", placeholder="e.g. USR0012")
    with col_s2:
        sort_by = st.selectbox("Sort By", ["combined_score", "amount", "timestamp", "ml_anomaly_score"])
    with col_s3:
        sort_order = st.selectbox("Order", ["Descending", "Ascending"])

    display_df = df.copy()
    if search_user:
        display_df = display_df[display_df["user_id"].str.contains(search_user.upper(), na=False)]

    display_df = display_df.sort_values(sort_by, ascending=(sort_order == "Ascending"))

    show_cols = ["transaction_id", "user_id", "amount", "timestamp", "location",
                 "device", "transaction_type", "final_risk", "combined_score", "rule_flags"]
    display_df_show = display_df[show_cols].head(200).copy()
    display_df_show["amount"] = display_df_show["amount"].apply(lambda x: f"₹{x:,.0f}")
    display_df_show["timestamp"] = pd.to_datetime(display_df_show["timestamp"]).dt.strftime("%d %b %Y %H:%M")
    display_df_show.columns = ["Tx ID", "User", "Amount", "Timestamp", "Location",
                                "Device", "Type", "Risk", "Score", "Flags"]

    def color_risk(val):
        if val == "High":
            return f"color: {PALETTE['red']}; font-weight: 600"
        elif val == "Medium":
            return f"color: {PALETTE['amber']}; font-weight: 600"
        return f"color: {PALETTE['green']}"

    styled = display_df_show.style.applymap(color_risk, subset=["Risk"])
    st.dataframe(styled, use_container_width=True, height=500)

    st.caption(f"Showing {min(200, len(display_df))} of {len(display_df):,} transactions")


# ══════════════════════════════════════════════
# TAB 4: REPORTS & EXPORT
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Export Fraud Logs</div>', unsafe_allow_html=True)

    col_r1, col_r2 = st.columns([2, 1])

    with col_r1:
        log_path, flagged_df = save_fraud_log(df_full)
        st.success(f"✅ Fraud log saved → `{log_path}` ({len(flagged_df)} flagged transactions)")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Flagged Transaction Summary</div>', unsafe_allow_html=True)

        summary_data = {
            "Metric": [
                "Total Transactions Analyzed",
                "High Risk Flagged",
                "Medium Risk Flagged",
                "Total Flagged",
                "Flag Rate",
                "Avg Combined Score (Flagged)",
                "Total Amount at Risk",
                "ML Anomalies Detected",
            ],
            "Value": [
                f"{len(df_full):,}",
                f"{high_risk}",
                f"{medium_risk}",
                f"{total_flagged}",
                f"{total_flagged/len(df_full)*100:.1f}%",
                f"{df_full[df_full['final_risk'].isin(['High','Medium'])]['combined_score'].mean():.1f}/100",
                f"₹{df_full[df_full['final_risk'].isin(['High','Medium'])]['amount'].sum():,.0f}",
                f"{df_full['ml_is_anomaly'].sum()}",
            ]
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

        # Download button
        csv_data = flagged_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️  Download Fraud Report (CSV)",
            data=csv_data,
            file_name=f"securepay_fraud_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col_r2:
        st.markdown('<div class="section-header">Top Fraud Patterns</div>', unsafe_allow_html=True)

        # Breakdown by rule flags
        flag_counts = {}
        for flags in df_full[df_full["final_risk"] == "High"]["rule_flags"]:
            for flag in str(flags).split(" | "):
                f = flag.strip()
                if f and f != "No anomalies detected":
                    flag_counts[f[:40]] = flag_counts.get(f[:40], 0) + 1

        if flag_counts:
            fc_df = pd.DataFrame(list(flag_counts.items()), columns=["Pattern", "Count"]).sort_values("Count", ascending=True).tail(8)
            fig_flags = go.Figure(go.Bar(
                y=fc_df["Pattern"],
                x=fc_df["Count"],
                orientation="h",
                marker=dict(color=PALETTE["red"], opacity=0.8)
            ))
            fig_flags.update_layout(
                height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=20, t=0, b=0),
                xaxis=dict(showgrid=True, gridcolor=PALETTE["border"], tickfont=dict(size=10)),
                yaxis=dict(showgrid=False, tickfont=dict(size=9)),
            )
            st.plotly_chart(fig_flags, use_container_width=True)

        # By location
        st.markdown('<div class="section-header" style="margin-top:16px;">High Risk by Location</div>', unsafe_allow_html=True)
        loc_df = (df_full[df_full["final_risk"] == "High"]
                  .groupby("location").size()
                  .sort_values(ascending=False).head(8)
                  .reset_index(name="count"))
        fig_loc = go.Figure(go.Bar(
            x=loc_df["location"],
            y=loc_df["count"],
            marker=dict(color=PALETTE["amber"], opacity=0.85)
        ))
        fig_loc.update_layout(
            height=200,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, tickfont=dict(size=9), tickangle=30),
            yaxis=dict(showgrid=True, gridcolor=PALETTE["border"], tickfont=dict(size=9)),
        )
        st.plotly_chart(fig_loc, use_container_width=True)