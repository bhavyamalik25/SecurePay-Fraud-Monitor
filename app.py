"""
app.py — SecurePay Fraud Monitor
Real-time fraud detection dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

from data_generator import load_or_generate, generate_dataset
from fraud_detection import run_rule_engine, get_rule_summary
from anomaly_model import train_model, predict_anomalies, get_ml_summary, load_model

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SecurePay Fraud Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Palette — Security Dashboard ─────────────────────────────────────────────
C = {
    "bg":        "#0F1117",   # deep charcoal — main background
    "surface":   "#1A1D27",   # slightly lighter — cards
    "sidebar":   "#141720",   # sidebar
    "border":    "#2A2D3A",   # subtle borders
    "text":      "#E8EAF0",   # primary text
    "muted":     "#7B7F8E",   # secondary text
    "accent":    "#3B82F6",   # blue accent
    "red":       "#EF4444",   # high risk
    "amber":     "#F59E0B",   # medium risk
    "green":     "#22C55E",   # safe / low risk
}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {C['bg']};
    color: {C['text']};
  }}
  .stApp {{ background-color: {C['bg']}; }}

  section[data-testid="stSidebar"] {{
    background-color: {C['sidebar']} !important;
    border-right: 1px solid {C['border']};
  }}
  section[data-testid="stSidebar"] * {{ color: {C['text']} !important; }}
  section[data-testid="stSidebar"] .stButton > button {{
    background-color: {C['surface']} !important;
    color: {C['text']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 6px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
  }}
  section[data-testid="stSidebar"] .stButton > button:hover {{
    border-color: {C['accent']} !important;
    color: {C['text']} !important;
  }}

  .card {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 8px;
    padding: 20px 24px;
  }}
  .metric-label {{
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {C['muted']};
    margin-bottom: 6px;
  }}
  .metric-value {{
    font-size: 1.9rem;
    font-weight: 700;
    color: {C['text']};
    line-height: 1.1;
  }}
  .metric-sub {{
    font-size: 0.73rem;
    color: {C['muted']};
    margin-top: 4px;
  }}

  .section-label {{
    font-size: 0.67rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {C['muted']};
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid {C['border']};
  }}

  .alert-high {{
    background: rgba(239,68,68,0.07);
    border-left: 3px solid {C['red']};
    padding: 12px 16px;
    border-radius: 0 6px 6px 0;
    margin-bottom: 8px;
    font-size: 0.84rem;
  }}
  .alert-medium {{
    background: rgba(245,158,11,0.07);
    border-left: 3px solid {C['amber']};
    padding: 12px 16px;
    border-radius: 0 6px 6px 0;
    margin-bottom: 8px;
    font-size: 0.84rem;
  }}

  .mono {{ font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; }}

  .stTabs [data-baseweb="tab-list"] {{
    background: {C['surface']};
    border-radius: 8px;
    padding: 3px;
    gap: 2px;
    border: 1px solid {C['border']};
  }}
  .stTabs [data-baseweb="tab"] {{
    border-radius: 6px;
    padding: 6px 18px;
    font-size: 0.82rem;
    font-weight: 500;
    color: {C['muted']} !important;
    background: transparent !important;
  }}
  .stTabs [data-baseweb="tab"]:hover {{
    color: {C['text']} !important;
    background: transparent !important;
  }}
  .stTabs [aria-selected="true"] {{
    background: {C['bg']} !important;
    color: {C['text']} !important;
    box-shadow: none;
  }}
  .stTabs [data-baseweb="tab"] p,
  .stTabs [data-baseweb="tab"] span {{
    color: inherit !important;
  }}

  hr {{ border: none; border-top: 1px solid {C['border']}; margin: 16px 0; }}

  #MainMenu {{ visibility: hidden; }}
  footer {{ visibility: hidden; }}
  header[data-testid="stHeader"] {{ background: transparent; }}
  [data-testid="stToolbar"] {{ display: none; }}
  [data-testid="stDecoration"] {{ display: none; }}

  .stDataFrame {{ border-radius: 8px; overflow: hidden; border: 1px solid {C['border']}; }}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def inr(amount):
    return f"₹{amount:,.0f}"

def plot_layout(fig, height):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(color=C["muted"], size=10),
    )
    return fig


# ── Load & analyze data ───────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_data():
    df = load_or_generate()
    df = run_rule_engine(df)
    artifacts = load_model() or train_model(df)
    df = predict_anomalies(df, artifacts)

    def final_risk(row):
        if row["rule_risk"] == "High" or row["ml_anomaly_score"] >= 75:
            return "High"
        elif row["rule_risk"] == "Medium" or row["ml_anomaly_score"] >= 45:
            return "Medium"
        return "Low"

    df["final_risk"] = df.apply(final_risk, axis=1)
    df["combined_score"] = np.round(df["rule_score"] * 0.6 + df["ml_anomaly_score"] * 0.4, 1)
    return df


with st.spinner("Loading..."):
    df_full = load_data()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:12px 0 20px 0;">
      <div style="font-size:1.1rem;font-weight:700;color:{C['text']};">SecurePay</div>
      <div style="font-size:0.7rem;color:{C['muted']};letter-spacing:0.08em;margin-top:2px;">FRAUD MONITOR</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div style="font-size:0.68rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:{C["muted"]};margin-bottom:8px;">Filters</div>', unsafe_allow_html=True)

    risk_filter = st.multiselect("Risk Level", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    amount_range = st.slider("Amount (₹)", 0, 500000, (0, 500000), step=5000, format="₹%d")

    st.markdown("---")
    st.markdown(f'<div style="font-size:0.68rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:{C["muted"]};margin-bottom:8px;">Actions</div>', unsafe_allow_html=True)

    if st.button("Regenerate Dataset", use_container_width=True):
        st.cache_data.clear()
        generate_dataset(save=True)
        st.rerun()

    if st.button("Retrain ML Model", use_container_width=True):
        st.cache_data.clear()
        with st.spinner("Training..."):
            train_model(load_or_generate())
        st.success("Model retrained.")
        st.rerun()

    st.markdown("---")
    st.markdown(f'<div style="font-size:0.68rem;color:{C["muted"]};line-height:1.7;">Rule Engine + Isolation Forest<br>Behavioral Anomaly Detection<br>Real-Time Risk Scoring</div>', unsafe_allow_html=True)


# ── Apply filters ─────────────────────────────────────────────────────────────
df = df_full[
    df_full["final_risk"].isin(risk_filter) &
    df_full["amount"].between(amount_range[0], amount_range[1])
].copy()


# ── Header ────────────────────────────────────────────────────────────────────
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown(f'<div style="font-size:1.4rem;font-weight:700;color:{C["text"]};">SecurePay Fraud Monitor</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.8rem;color:{C["muted"]};">Real-time transaction analysis · Rule Engine + ML Anomaly Detection</div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div style="text-align:right;padding-top:6px;font-size:0.78rem;color:{C["muted"]};">{datetime.now().strftime("%d %b %Y · %H:%M")}</div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ── KPI Row ───────────────────────────────────────────────────────────────────
total      = len(df_full)
high_risk  = (df_full["final_risk"] == "High").sum()
medium_risk = (df_full["final_risk"] == "Medium").sum()
low_risk   = (df_full["final_risk"] == "Low").sum()
total_flagged = high_risk + medium_risk

k1, k2, k3, k4, k5 = st.columns(5)

for col, label, value, sub, color in [
    (k1, "Total Transactions", f"{total:,}",       "All users",                          C["text"]),
    (k2, "High Risk",          str(high_risk),      f"{round(high_risk/total*100,1)}%",   C["red"]),
    (k3, "Medium Risk",        str(medium_risk),    f"{round(medium_risk/total*100,1)}%", C["amber"]),
    (k4, "Safe",               str(low_risk),       f"{round(low_risk/total*100,1)}%",    C["green"]),
    (k5, "Total Volume",       f"₹{df_full['amount'].sum()/1e6:.1f}M", f"Avg score: {df_full['combined_score'].mean():.0f}/100", C["text"]),
]:
    col.markdown(f"""
    <div class="card">
      <div class="metric-label">{label}</div>
      <div class="metric-value" style="color:{color};">{value}</div>
      <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Live Alerts", "Transaction Explorer", "Reports & Export"])


# ════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ════════════════════════════════════════════════
with tab1:
    left, right = st.columns([3, 2])

    with left:
        # Timeline
        st.markdown('<div class="section-label">30-Day Transaction Volume</div>', unsafe_allow_html=True)
        daily = df_full.copy()
        daily["date"] = pd.to_datetime(daily["timestamp"]).dt.date
        dc = daily.groupby(["date", "final_risk"]).size().unstack(fill_value=0)
        for col in ["High", "Medium", "Low"]:
            if col not in dc.columns:
                dc[col] = 0

        fig = go.Figure()
        for risk, color, fill in [
            ("Low",    C["green"], "rgba(34,197,94,0.12)"),
            ("Medium", C["amber"], "rgba(245,158,11,0.15)"),
            ("High",   C["red"],   "rgba(239,68,68,0.18)"),
        ]:
            fig.add_trace(go.Scatter(x=dc.index, y=dc[risk], name=risk,
                fill="tozeroy", line=dict(color=color, width=1.5), fillcolor=fill))
        fig = plot_layout(fig, 250)
        fig.update_layout(
            legend=dict(orientation="h", y=-0.25, x=0),
            xaxis=dict(showgrid=False, tickfont=dict(size=10)),
            yaxis=dict(showgrid=True, gridcolor=C["border"]),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Amount distribution
        st.markdown('<div class="section-label">Amount Distribution by Risk</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        for risk, color in [("Low", C["green"]), ("Medium", C["amber"]), ("High", C["red"])]:
            fig2.add_trace(go.Histogram(
                x=df_full[df_full["final_risk"] == risk]["amount"],
                name=risk, marker_color=color, opacity=0.7, nbinsx=40
            ))
        fig2 = plot_layout(fig2, 220)
        fig2.update_layout(
            barmode="overlay",
            legend=dict(orientation="h", y=-0.28),
            xaxis=dict(title="Amount (₹)", showgrid=False, tickfont=dict(size=10)),
            yaxis=dict(showgrid=True, gridcolor=C["border"]),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with right:
        # Donut
        st.markdown('<div class="section-label">Risk Distribution</div>', unsafe_allow_html=True)
        fig3 = go.Figure(go.Pie(
            labels=["High", "Medium", "Safe"],
            values=[high_risk, medium_risk, low_risk],
            hole=0.65,
            marker=dict(colors=[C["red"], C["amber"], C["green"]]),
            textinfo="percent", textfont=dict(size=11),
        ))
        fig3 = plot_layout(fig3, 230)
        fig3.update_layout(
            legend=dict(orientation="v", x=0.72, y=0.5),
            annotations=[dict(text=f"<b>{total}</b>", x=0.5, y=0.5,
                              font=dict(size=16, color=C["text"]), showarrow=False)]
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Top flagged users
        st.markdown('<div class="section-label">Top Flagged Users</div>', unsafe_allow_html=True)
        top = (df_full[df_full["final_risk"].isin(["High", "Medium"])]
               .groupby("user_id")
               .agg(flags=("transaction_id", "count"), score=("combined_score", "mean"))
               .sort_values("flags", ascending=False).head(6).reset_index())
        fig4 = go.Figure(go.Bar(
            y=top["user_id"], x=top["flags"], orientation="h",
            marker=dict(color=top["score"],
                        colorscale=[[0, C["amber"]], [1, C["red"]]],
                        showscale=False),
            text=top["flags"].astype(str), textposition="outside",
        ))
        fig4 = plot_layout(fig4, 220)
        fig4.update_layout(
            xaxis=dict(showgrid=True, gridcolor=C["border"]),
            yaxis=dict(showgrid=False, tickfont=dict(family="JetBrains Mono", size=10)),
        )
        st.plotly_chart(fig4, use_container_width=True)

        # Hourly activity
        st.markdown('<div class="section-label">Activity by Hour</div>', unsafe_allow_html=True)
        df_full["hour"] = pd.to_datetime(df_full["timestamp"]).dt.hour
        hourly = df_full.groupby("hour")["final_risk"].value_counts().unstack(fill_value=0)
        fig5 = go.Figure()
        for risk, color in [("High", C["red"]), ("Medium", C["amber"]), ("Low", C["green"])]:
            if risk in hourly.columns:
                fig5.add_trace(go.Bar(x=hourly.index, y=hourly[risk], name=risk,
                                      marker_color=color, opacity=0.85))
        fig5 = plot_layout(fig5, 195)
        fig5.update_layout(
            barmode="stack",
            legend=dict(orientation="h", y=-0.3),
            xaxis=dict(title="Hour", showgrid=False, tickfont=dict(size=9)),
            yaxis=dict(showgrid=True, gridcolor=C["border"]),
        )
        st.plotly_chart(fig5, use_container_width=True)


# ════════════════════════════════════════════════
# TAB 2 — LIVE ALERTS
# ════════════════════════════════════════════════
with tab2:
    col_alerts, col_summary = st.columns([2, 1])

    with col_alerts:
        st.markdown('<div class="section-label">High Risk Alerts</div>', unsafe_allow_html=True)
        high_df = df_full[df_full["final_risk"] == "High"].sort_values("combined_score", ascending=False).head(20)

        if high_df.empty:
            st.info("No high-risk transactions detected.")
        else:
            for _, row in high_df.iterrows():
                ts = pd.to_datetime(row["timestamp"]).strftime("%d %b · %H:%M")
                st.markdown(f"""
                <div class="alert-high">
                  <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                      <span class="mono">{row['transaction_id']}</span>
                      <span style="margin:0 8px;color:{C['muted']};">·</span>
                      <span style="font-weight:600;">{row['user_id']}</span>
                    </div>
                    <div style="font-weight:700;color:{C['red']};">{inr(row['amount'])}</div>
                  </div>
                  <div style="margin-top:5px;font-size:0.77rem;color:{C['muted']};">
                    {row['location']} · {row['device']} · {ts}
                  </div>
                  <div style="margin-top:4px;font-size:0.75rem;color:{C['red']};">
                    {str(row['rule_flags'])[:120]}{'…' if len(str(row['rule_flags'])) > 120 else ''}
                  </div>
                  <div style="margin-top:3px;font-size:0.73rem;color:{C['muted']};">
                    Score: <b style="color:{C['text']};">{int(row['combined_score'])}/100</b>
                    · ML: <b style="color:{C['text']};">{int(row['ml_anomaly_score'])}/100</b>
                  </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Medium Risk Alerts</div>', unsafe_allow_html=True)
        med_df = df_full[df_full["final_risk"] == "Medium"].sort_values("combined_score", ascending=False).head(15)
        for _, row in med_df.iterrows():
            ts = pd.to_datetime(row["timestamp"]).strftime("%d %b · %H:%M")
            st.markdown(f"""
            <div class="alert-medium">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                  <span class="mono">{row['transaction_id']}</span>
                  <span style="margin:0 8px;color:{C['muted']};">·</span>
                  <span style="font-weight:600;">{row['user_id']}</span>
                </div>
                <div style="font-weight:700;color:{C['amber']};">{inr(row['amount'])}</div>
              </div>
              <div style="margin-top:5px;font-size:0.77rem;color:{C['muted']};">
                {row['location']} · {row['device']} · {ts}
              </div>
              <div style="margin-top:4px;font-size:0.75rem;color:{C['amber']};">
                {str(row['rule_flags'])[:100]}{'…' if len(str(row['rule_flags'])) > 100 else ''}
              </div>
            </div>""", unsafe_allow_html=True)

    with col_summary:
        rule_sum = get_rule_summary(df_full)
        ml_sum   = get_ml_summary(df_full)

        st.markdown('<div class="section-label">Rule Engine</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card" style="margin-bottom:12px;">
          {''.join([
            f'<div style="display:flex;justify-content:space-between;font-size:0.82rem;padding:6px 0;border-bottom:1px solid {C["border"]};">'
            f'<span style="color:{C["muted"]};">{k}</span><span style="font-weight:600;color:{v2};">{v}</span></div>'
            for k, v, v2 in [
                ("High Risk",    rule_sum['high_risk_count'],   C["red"]),
                ("Medium Risk",  rule_sum['medium_risk_count'], C["amber"]),
                ("Avg Score",    f"{rule_sum['avg_risk_score']}/100", C["text"]),
            ]
          ])}
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-label">ML Detection</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card" style="margin-bottom:16px;">
          {''.join([
            f'<div style="display:flex;justify-content:space-between;font-size:0.82rem;padding:6px 0;border-bottom:1px solid {C["border"]};">'
            f'<span style="color:{C["muted"]};">{k}</span><span style="font-weight:600;color:{v2};">{v}</span></div>'
            for k, v, v2 in [
                ("Anomalies",   ml_sum['anomalies_detected'],    C["red"]),
                ("Rate",        f"{ml_sum['anomaly_rate_pct']}%", C["text"]),
                ("Avg Score",   f"{ml_sum['avg_anomaly_score']}/100", C["text"]),
            ]
          ])}
        </div>""", unsafe_allow_html=True)

        # Scatter
        st.markdown('<div class="section-label">Rule vs ML Score</div>', unsafe_allow_html=True)
        fig6 = px.scatter(
            df_full.sample(min(300, len(df_full))),
            x="rule_score", y="ml_anomaly_score", color="final_risk",
            color_discrete_map={"High": C["red"], "Medium": C["amber"], "Low": C["green"]},
            opacity=0.65,
        )
        fig6 = plot_layout(fig6, 270)
        fig6.update_layout(
            legend=dict(title="", orientation="h", y=-0.22),
            xaxis=dict(title="Rule Score", showgrid=True, gridcolor=C["border"]),
            yaxis=dict(title="ML Score",   showgrid=True, gridcolor=C["border"]),
        )
        st.plotly_chart(fig6, use_container_width=True)


# ════════════════════════════════════════════════
# TAB 3 — TRANSACTION EXPLORER
# ════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-label">Transaction Table</div>', unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    search_user = s1.text_input("Search User ID", placeholder="e.g. USR0012")
    sort_by     = s2.selectbox("Sort By", ["combined_score", "amount", "timestamp", "ml_anomaly_score"])
    sort_asc    = s3.selectbox("Order", ["Descending", "Ascending"]) == "Ascending"

    view = df.copy()
    if search_user:
        view = view[view["user_id"].str.contains(search_user.upper(), na=False)]
    view = view.sort_values(sort_by, ascending=sort_asc)

    cols = ["transaction_id", "user_id", "amount", "timestamp", "location",
            "device", "transaction_type", "final_risk", "combined_score", "rule_flags"]
    view = view[cols].head(200).copy()
    view["amount"]    = view["amount"].apply(lambda x: f"₹{x:,.0f}")
    view["timestamp"] = pd.to_datetime(view["timestamp"]).dt.strftime("%d %b %Y %H:%M")
    view.columns = ["Tx ID", "User", "Amount", "Timestamp", "Location", "Device", "Type", "Risk", "Score", "Flags"]

    def color_risk(val):
        return {
            "High":   f"color:{C['red']};font-weight:600",
            "Medium": f"color:{C['amber']};font-weight:600",
            "Low":    f"color:{C['green']}",
        }.get(val, "")

    st.dataframe(view.style.applymap(color_risk, subset=["Risk"]), use_container_width=True, height=500)
    st.caption(f"Showing {min(200, len(view))} of {len(df):,} transactions")


# ════════════════════════════════════════════════
# TAB 4 — REPORTS & EXPORT
# ════════════════════════════════════════════════
with tab4:
    r1, r2 = st.columns([2, 1])

    with r1:
        os.makedirs("logs", exist_ok=True)
        flagged = df_full[df_full["final_risk"].isin(["High", "Medium"])].copy()
        flagged.to_csv("logs/fraud_logs.csv", index=False)
        st.success(f"Fraud log saved → logs/fraud_logs.csv ({len(flagged)} flagged transactions)")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Summary</div>', unsafe_allow_html=True)

        summary = pd.DataFrame({
            "Metric": [
                "Total Transactions Analyzed", "High Risk", "Medium Risk",
                "Total Flagged", "Flag Rate", "Avg Score (Flagged)",
                "Total Amount at Risk", "ML Anomalies",
            ],
            "Value": [
                f"{len(df_full):,}", str(high_risk), str(medium_risk),
                str(total_flagged), f"{total_flagged/len(df_full)*100:.1f}%",
                f"{df_full[df_full['final_risk'].isin(['High','Medium'])]['combined_score'].mean():.1f}/100",
                f"₹{flagged['amount'].sum():,.0f}",
                str(df_full['ml_is_anomaly'].sum()),
            ]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.download_button(
            label="Download Fraud Report (CSV)",
            data=flagged.to_csv(index=False).encode("utf-8"),
            file_name=f"securepay_fraud_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with r2:
        st.markdown('<div class="section-label">Top Fraud Patterns</div>', unsafe_allow_html=True)
        flag_counts = {}
        for flags in df_full[df_full["final_risk"] == "High"]["rule_flags"]:
            for f in str(flags).split(" | "):
                f = f.strip()
                if f and f != "No anomalies detected":
                    flag_counts[f[:40]] = flag_counts.get(f[:40], 0) + 1

        if flag_counts:
            fc = pd.DataFrame(list(flag_counts.items()), columns=["Pattern", "Count"]).sort_values("Count").tail(8)
            fig7 = go.Figure(go.Bar(y=fc["Pattern"], x=fc["Count"], orientation="h",
                                    marker=dict(color=C["red"], opacity=0.85)))
            fig7 = plot_layout(fig7, 300)
            fig7.update_layout(
                xaxis=dict(showgrid=True, gridcolor=C["border"]),
                yaxis=dict(showgrid=False, tickfont=dict(size=9)),
            )
            st.plotly_chart(fig7, use_container_width=True)

        st.markdown('<div class="section-label" style="margin-top:12px;">High Risk by Location</div>', unsafe_allow_html=True)
        loc = (df_full[df_full["final_risk"] == "High"].groupby("location").size()
               .sort_values(ascending=False).head(8).reset_index(name="count"))
        fig8 = go.Figure(go.Bar(x=loc["location"], y=loc["count"],
                                marker=dict(color=C["amber"], opacity=0.85)))
        fig8 = plot_layout(fig8, 195)
        fig8.update_layout(
            xaxis=dict(showgrid=False, tickangle=30, tickfont=dict(size=9)),
            yaxis=dict(showgrid=True, gridcolor=C["border"]),
        )
        st.plotly_chart(fig8, use_container_width=True)