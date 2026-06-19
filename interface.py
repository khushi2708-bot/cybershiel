import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import datetime
warnings.filterwarnings("ignore")

# ─── Alert Notification System ───────────────────────────────────────────────

def init_alert_state():
    """Initialize session state for alert tracking."""
    if "alert_log" not in st.session_state:
        st.session_state.alert_log = []
    if "dismissed_alerts" not in st.session_state:
        st.session_state.dismissed_alerts = set()
    if "alert_counter" not in st.session_state:
        st.session_state.alert_counter = 0

def add_alert(level: str, title: str, message: str):
    """
    Register an alert in the session log.
    level: 'critical' | 'warning' | 'info' | 'success'
    """
    st.session_state.alert_counter += 1
    alert_id = st.session_state.alert_counter
    st.session_state.alert_log.append({
        "id":        alert_id,
        "level":     level,
        "title":     title,
        "message":   message,
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "date":      datetime.datetime.now().strftime("%Y-%m-%d"),
    })
    return alert_id

def generate_alerts(total_records, high_count, medium_count, anomaly_count,
                    avg_risk, max_risk, risk_threshold, contamination):
    """
    Evaluate detection results and register appropriate alerts.
    Returns list of alert ids generated this run.
    """
    generated = []
    pct_high = high_count / total_records * 100 if total_records else 0
    pct_med  = medium_count / total_records * 100 if total_records else 0

    # ── CRITICAL alerts ──────────────────────────────────────────────────────
    if pct_high > 20:
        generated.append(add_alert(
            "critical",
            "🚨 CRITICAL — Mass Mule Account Outbreak",
            f"{high_count:,} accounts ({pct_high:.1f}%) exceed the high-risk threshold of {risk_threshold}. "
            "Immediate review and potential account freeze recommended.",
        ))
    elif pct_high > 10:
        generated.append(add_alert(
            "critical",
            "🚨 CRITICAL — High Volume of Suspicious Accounts",
            f"{high_count:,} accounts ({pct_high:.1f}%) flagged as High Risk. "
            "Escalate to fraud investigation team.",
        ))

    if max_risk >= 98:
        generated.append(add_alert(
            "critical",
            "🚨 CRITICAL — Extreme Risk Account Detected",
            f"One or more accounts scored {max_risk:.1f}/100 — near-perfect anomaly signature. "
            "Manual inspection required immediately.",
        ))

    # ── WARNING alerts ───────────────────────────────────────────────────────
    if 5 < pct_high <= 10:
        generated.append(add_alert(
            "warning",
            "⚠️ WARNING — Elevated High-Risk Account Rate",
            f"{high_count:,} accounts ({pct_high:.1f}%) flagged as High Risk. "
            "Monitor closely and consider lowering the risk threshold.",
        ))

    if pct_med > 30:
        generated.append(add_alert(
            "warning",
            "⚠️ WARNING — Large Medium-Risk Pool",
            f"{medium_count:,} accounts ({pct_med:.1f}%) sit in the Medium Risk band. "
            "A significant portion may be borderline mule accounts.",
        ))

    if avg_risk > 65:
        generated.append(add_alert(
            "warning",
            "⚠️ WARNING — High Average Risk Score Across Dataset",
            f"Dataset average risk score is {avg_risk:.1f}/100 — well above normal baseline. "
            "The overall account population shows elevated anomaly signals.",
        ))

    if contamination >= 0.15:
        generated.append(add_alert(
            "warning",
            "⚠️ WARNING — High Contamination Rate Setting",
            f"Contamination is set to {contamination*100:.0f}%. At this level the model expects "
            "many anomalies, which may inflate false positives. Consider tuning this value.",
        ))

    # ── INFO alerts ──────────────────────────────────────────────────────────
    if 1 < pct_high <= 5:
        generated.append(add_alert(
            "info",
            "ℹ️ INFO — Moderate High-Risk Accounts Detected",
            f"{high_count:,} accounts ({pct_high:.1f}%) flagged High Risk — within expected contamination range. "
            "Standard review process recommended.",
        ))

    if anomaly_count != high_count:
        diff = abs(anomaly_count - high_count)
        generated.append(add_alert(
            "info",
            "ℹ️ INFO — Model vs Threshold Count Divergence",
            f"Isolation Forest flagged {anomaly_count:,} anomalies, but the risk-score threshold "
            f"captured {high_count:,} high-risk records (Δ {diff:,}). "
            "Adjusting the contamination rate or threshold may align these.",
        ))

    # ── SUCCESS alert ────────────────────────────────────────────────────────
    if pct_high <= 1 and avg_risk < 40:
        generated.append(add_alert(
            "success",
            "✅ CLEAR — Dataset Appears Low Risk",
            f"Only {high_count:,} accounts ({pct_high:.1f}%) flagged High Risk and the average "
            f"risk score is {avg_risk:.1f}/100. No immediate action required.",
        ))

    return generated


def render_alert_banners(alert_ids: list):
    """Display dismissible banner notifications for newly generated alerts."""
    if not alert_ids:
        return

    st.markdown('<p class="sec-head">🔔 Active Alert Notifications</p>', unsafe_allow_html=True)

    LEVEL_CFG = {
        "critical": {"fn": st.error,   "icon": "🚨"},
        "warning":  {"fn": st.warning, "icon": "⚠️"},
        "info":     {"fn": st.info,    "icon": "ℹ️"},
        "success":  {"fn": st.success, "icon": "✅"},
    }

    for alert in st.session_state.alert_log:
        if alert["id"] not in alert_ids:
            continue
        if alert["id"] in st.session_state.dismissed_alerts:
            continue

        cfg = LEVEL_CFG.get(alert["level"], LEVEL_CFG["info"])
        col_msg, col_btn = st.columns([9, 1])
        with col_msg:
            cfg["fn"](f"**{alert['title']}**  \n{alert['message']}")
        with col_btn:
            if st.button("✖", key=f"dismiss_{alert['id']}", help="Dismiss alert"):
                st.session_state.dismissed_alerts.add(alert["id"])
                st.rerun()


def render_sidebar_alert_panel():
    """Show a compact alert summary badge panel in the sidebar."""
    if not st.session_state.alert_log:
        return

    counts = {"critical": 0, "warning": 0, "info": 0, "success": 0}
    for a in st.session_state.alert_log:
        counts[a["level"]] = counts.get(a["level"], 0) + 1

    active = sum(1 for a in st.session_state.alert_log
                 if a["id"] not in st.session_state.dismissed_alerts)

    badge_html = f"""
    <div style="margin-top:0.8rem;">
      <div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;
                  letter-spacing:0.15em;color:#00d4ff;margin-bottom:0.5rem;">
        🔔 ALERT SUMMARY ({active} active)
      </div>
      <div style="display:flex;gap:0.4rem;flex-wrap:wrap;">
        {"" if counts["critical"]==0 else
          f'<span style="background:#ff4d6a22;border:1px solid #ff4d6a;color:#ff4d6a;'
          f'border-radius:4px;padding:2px 8px;font-size:0.72rem;">🚨 {counts["critical"]} Critical</span>'}
        {"" if counts["warning"]==0 else
          f'<span style="background:#ffb34722;border:1px solid #ffb347;color:#ffb347;'
          f'border-radius:4px;padding:2px 8px;font-size:0.72rem;">⚠️ {counts["warning"]} Warning</span>'}
        {"" if counts["info"]==0 else
          f'<span style="background:#00d4ff22;border:1px solid #00d4ff;color:#00d4ff;'
          f'border-radius:4px;padding:2px 8px;font-size:0.72rem;">ℹ️ {counts["info"]} Info</span>'}
        {"" if counts["success"]==0 else
          f'<span style="background:#39d98a22;border:1px solid #39d98a;color:#39d98a;'
          f'border-radius:4px;padding:2px 8px;font-size:0.72rem;">✅ {counts["success"]} Clear</span>'}
      </div>
    </div>
    """
    st.markdown(badge_html, unsafe_allow_html=True)


def render_alert_log_table():
    """Render the full alert history as a styled dataframe."""
    if not st.session_state.alert_log:
        return

    st.markdown('<p class="sec-head">📋 Alert Log</p>', unsafe_allow_html=True)

    LEVEL_EMOJI = {"critical": "🚨 Critical", "warning": "⚠️ Warning",
                   "info": "ℹ️ Info", "success": "✅ Clear"}

    log_rows = []
    for a in st.session_state.alert_log:
        log_rows.append({
            "Time":    a["timestamp"],
            "Level":   LEVEL_EMOJI.get(a["level"], a["level"]),
            "Title":   a["title"],
            "Details": a["message"],
            "Status":  "Dismissed" if a["id"] in st.session_state.dismissed_alerts else "Active",
        })

    log_df = pd.DataFrame(log_rows)

    col_filter, col_clear = st.columns([3, 1])
    with col_filter:
        level_filter = st.selectbox(
            "Filter by Level",
            ["All", "🚨 Critical", "⚠️ Warning", "ℹ️ Info", "✅ Clear"],
            key="alert_log_filter",
        )
    with col_clear:
        st.write("")
        if st.button("🗑️ Clear Log", key="clear_alert_log"):
            st.session_state.alert_log = []
            st.session_state.dismissed_alerts = set()
            st.session_state.alert_counter = 0
            st.rerun()

    if level_filter != "All":
        log_df = log_df[log_df["Level"] == level_filter]

    if log_df.empty:
        st.info("No alerts match the selected filter.")
    else:
        st.dataframe(log_df, use_container_width=True, hide_index=True, height=220)
        st.caption(f"Total alerts logged this session: {len(st.session_state.alert_log)}")

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CyberShield — Mule Account Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global Style ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
      background-color: #070b12 !important;
      color: #c9d6e3 !important;
      font-family: 'Inter', sans-serif;
  }
  .stApp { background-color: #070b12; }

  /* Header */
  .cs-header {
      text-align: center;
      padding: 2rem 0 1.2rem;
      border-bottom: 1px solid #1a2a3a;
      margin-bottom: 1.8rem;
  }
  .cs-title {
      font-family: 'Share Tech Mono', monospace;
      font-size: 2rem;
      letter-spacing: 0.15em;
      color: #00d4ff;
      text-shadow: 0 0 18px rgba(0,212,255,0.45);
      margin: 0;
  }
  .cs-subtitle {
      font-size: 0.78rem;
      letter-spacing: 0.25em;
      text-transform: uppercase;
      color: #4a6a8a;
      margin-top: 0.3rem;
  }

  /* Metric cards */
  .metric-row { display: flex; gap: 1rem; margin-bottom: 1.4rem; flex-wrap: wrap; }
  .metric-card {
      flex: 1 1 160px;
      background: #0d1a27;
      border: 1px solid #1a2d42;
      border-radius: 6px;
      padding: 1rem 1.2rem;
      text-align: center;
  }
  .metric-card .val {
      font-family: 'Share Tech Mono', monospace;
      font-size: 1.8rem;
      color: #00d4ff;
      line-height: 1;
  }
  .metric-card.danger .val { color: #ff4d6a; }
  .metric-card.warn   .val { color: #ffb347; }
  .metric-card.ok     .val { color: #39d98a; }
  .metric-card .lbl {
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: #4a6a8a;
      margin-top: 0.35rem;
  }

  /* Section headers */
  .sec-head {
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.85rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: #00d4ff;
      border-left: 3px solid #00d4ff;
      padding-left: 0.7rem;
      margin: 1.6rem 0 0.9rem;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
      background-color: #0d1520 !important;
      border-right: 1px solid #1a2d42;
  }
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stSlider label,
  section[data-testid="stSidebar"] p { color: #8ab0cc !important; }

  /* Dataframe */
  .stDataFrame { border: 1px solid #1a2d42; border-radius: 4px; }

  /* Alerts */
  .stAlert { border-radius: 4px !important; }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cs-header">
  <p class="cs-title">⚡ CYBERSHIELD ⚡</p>
  <p class="cs-subtitle">AI-Based Mule Account Detection System</p>
</div>
""", unsafe_allow_html=True)

# ─── Initialize Alert State ───────────────────────────────────────────────────
init_alert_state()

# ─── Sidebar Controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Detection Settings")

    contamination = st.slider(
        "Contamination Rate",
        min_value=0.01, max_value=0.20, value=0.05, step=0.01,
        help="Fraction of records expected to be anomalous (mule accounts).",
    )
    risk_threshold = st.slider(
        "High-Risk Score Threshold",
        min_value=50, max_value=95, value=75, step=5,
        help="Records scoring above this value are flagged as High Risk.",
    )
    max_features = st.slider(
        "Max Features to Use",
        min_value=50, max_value=500, value=200, step=50,
        help="Number of best-filled feature columns fed to the model.",
    )
    show_pca = st.checkbox("Show PCA Scatter Plot", value=True)
    show_raw  = st.checkbox("Show Raw High-Risk Records", value=True)

    st.markdown("---")
    st.caption("Upload your dataset below ↓")

    # ── Alert summary badges in sidebar ──────────────────────────────────────
    render_sidebar_alert_panel()

# ─── File Upload ──────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload Dataset (CSV)",
    type=["csv"],
    help="Expects the CyberShield feature matrix (F1 … FN columns).",
)

# ─── Helper: pick best-filled numeric columns ─────────────────────────────────
def select_features(df: pd.DataFrame, max_cols: int) -> list:
    """Return up to max_cols numeric columns with the highest fill-rate."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Unnamed: 0" in num_cols:
        num_cols.remove("Unnamed: 0")
    fill_rates = df[num_cols].notna().mean().sort_values(ascending=False)
    return fill_rates.head(max_cols).index.tolist()


# ─── Processing ───────────────────────────────────────────────────────────────
if uploaded_file is not None:
    st.info("🔄 Loading and processing dataset…")

    # --- Load in chunks to stay memory-friendly ---
    chunks = []
    for chunk in pd.read_csv(uploaded_file, chunksize=5000):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)

    total_records = len(df)
    total_features = len([c for c in df.columns if c.startswith("F")])

    # --- Feature selection ---
    feature_cols = select_features(df, max_features)
    X_raw = df[feature_cols].copy()

    # Impute missing values with column median (fast, robust)
    X_filled = X_raw.fillna(X_raw.median(numeric_only=True))

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filled)

    # --- Isolation Forest ---
    model = IsolationForest(
        contamination=contamination,
        n_estimators=150,
        random_state=42,
        n_jobs=-1,
    )
    flags = model.fit_predict(X_scaled)          # -1 = anomaly, 1 = normal
    raw_scores = model.decision_function(X_scaled)  # lower = more anomalous

    # Map to 0-100 risk score (higher = riskier)
    score_min, score_max = raw_scores.min(), raw_scores.max()
    risk_scores = 100 * (1 - (raw_scores - score_min) / (score_max - score_min + 1e-9))
    risk_scores = np.clip(risk_scores, 0, 100)

    df["risk_score"]   = risk_scores.round(1)
    df["anomaly_flag"] = flags
    df["risk_level"]   = pd.cut(
        df["risk_score"],
        bins=[0, 40, risk_threshold, 100],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )

    high_risk_df  = df[df["risk_score"] > risk_threshold]
    medium_risk_df = df[(df["risk_score"] > 40) & (df["risk_score"] <= risk_threshold)]
    anomaly_count  = int((flags == -1).sum())

    # ── Generate & display alert notifications ────────────────────────────────
    new_alert_ids = generate_alerts(
        total_records  = total_records,
        high_count     = len(high_risk_df),
        medium_count   = len(medium_risk_df),
        anomaly_count  = anomaly_count,
        avg_risk       = df["risk_score"].mean(),
        max_risk       = df["risk_score"].max(),
        risk_threshold = risk_threshold,
        contamination  = contamination,
    )
    render_alert_banners(new_alert_ids)

    # ── KPI cards ────────────────────────────────────────────────────────────
    st.success(f"✅ Processed {total_records:,} records across {total_features:,} features ({len(feature_cols)} used for detection).")

    st.markdown('<p class="sec-head">📊 Detection Summary</p>', unsafe_allow_html=True)

    pct_high = len(high_risk_df) / total_records * 100
    pct_med  = len(medium_risk_df) / total_records * 100
    pct_low  = (total_records - len(high_risk_df) - len(medium_risk_df)) / total_records * 100

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card"><div class="val">{total_records:,}</div><div class="lbl">Total Records</div></div>
      <div class="metric-card danger"><div class="val">{len(high_risk_df):,}</div><div class="lbl">High Risk ({pct_high:.1f}%)</div></div>
      <div class="metric-card warn"><div class="val">{len(medium_risk_df):,}</div><div class="lbl">Medium Risk ({pct_med:.1f}%)</div></div>
      <div class="metric-card ok"><div class="val">{total_records - len(high_risk_df) - len(medium_risk_df):,}</div><div class="lbl">Low Risk ({pct_low:.1f}%)</div></div>
      <div class="metric-card"><div class="val">{anomaly_count:,}</div><div class="lbl">Model Anomalies</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Risk Distribution Histogram ──────────────────────────────────────────
    st.markdown('<p class="sec-head">📈 Risk Score Distribution</p>', unsafe_allow_html=True)

    col_chart, col_summary = st.columns([3, 1])
    with col_chart:
        bins = np.arange(0, 101, 5)
        hist_vals, bin_edges = np.histogram(df["risk_score"], bins=bins)

        fig, ax = plt.subplots(figsize=(10, 3.5))
        fig.patch.set_facecolor("#070b12")
        ax.set_facecolor("#0d1520")

        colors = []
        for edge in bin_edges[:-1]:
            if edge >= risk_threshold:
                colors.append("#ff4d6a")
            elif edge >= 40:
                colors.append("#ffb347")
            else:
                colors.append("#39d98a")

        ax.bar(bin_edges[:-1], hist_vals, width=4.5, color=colors, edgecolor="#0d1520", linewidth=0.4)
        ax.axvline(risk_threshold, color="#ff4d6a", linewidth=1.5, linestyle="--", label=f"High-Risk Threshold ({risk_threshold})")
        ax.set_xlabel("Risk Score", color="#8ab0cc", fontsize=9)
        ax.set_ylabel("Record Count", color="#8ab0cc", fontsize=9)
        ax.tick_params(colors="#8ab0cc", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1a2d42")
        ax.legend(fontsize=8, facecolor="#0d1520", edgecolor="#1a2d42", labelcolor="#c9d6e3")
        ax.grid(axis="y", color="#1a2d42", linewidth=0.5, linestyle=":")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_summary:
        st.metric("Avg Risk Score",   f"{df['risk_score'].mean():.1f}")
        st.metric("Median Risk Score", f"{df['risk_score'].median():.1f}")
        st.metric("Max Risk Score",    f"{df['risk_score'].max():.1f}")
        st.metric("Features Used",     len(feature_cols))

    # ── PCA Scatter ──────────────────────────────────────────────────────────
    if show_pca:
        st.markdown('<p class="sec-head">🔵 PCA Anomaly Scatter (2D Projection)</p>', unsafe_allow_html=True)

        with st.spinner("Running PCA…"):
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X_scaled)

        fig2, ax2 = plt.subplots(figsize=(10, 4.5))
        fig2.patch.set_facecolor("#070b12")
        ax2.set_facecolor("#0d1520")

        # Sample for speed if large
        sample_size = min(3000, total_records)
        idx = np.random.choice(total_records, sample_size, replace=False)
        scores_s = df["risk_score"].values[idx]
        coords_s = coords[idx]

        sc = ax2.scatter(
            coords_s[:, 0], coords_s[:, 1],
            c=scores_s, cmap="RdYlGn_r",
            s=8, alpha=0.65, linewidths=0,
        )
        cbar = plt.colorbar(sc, ax=ax2, pad=0.01)
        cbar.set_label("Risk Score", color="#8ab0cc", fontsize=8)
        cbar.ax.yaxis.set_tick_params(color="#8ab0cc", labelsize=7)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8ab0cc")

        ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", color="#8ab0cc", fontsize=9)
        ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", color="#8ab0cc", fontsize=9)
        ax2.tick_params(colors="#8ab0cc", labelsize=8)
        for spine in ax2.spines.values():
            spine.set_edgecolor("#1a2d42")
        ax2.set_title(f"Sampled {sample_size:,} of {total_records:,} records", color="#4a6a8a", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        expl = pca.explained_variance_ratio_
        st.caption(f"PCA explains {expl[0]*100:.1f}% + {expl[1]*100:.1f}% = {sum(expl)*100:.1f}% of total variance.")

    # ── Feature Variance Insight ──────────────────────────────────────────────
    st.markdown('<p class="sec-head">🔍 Top Discriminating Features</p>', unsafe_allow_html=True)

    # Compare mean feature values between high-risk and low-risk groups
    high_mask = df["risk_score"] > risk_threshold
    low_mask  = df["risk_score"] < 40

    if high_mask.sum() > 0 and low_mask.sum() > 0:
        hi_means  = X_filled[high_mask].mean()
        lo_means  = X_filled[low_mask].mean()
        diff      = (hi_means - lo_means).abs()
        top_feats = diff.nlargest(15).index.tolist()

        fig3, ax3 = plt.subplots(figsize=(10, 3.5))
        fig3.patch.set_facecolor("#070b12")
        ax3.set_facecolor("#0d1520")

        x = np.arange(len(top_feats))
        w = 0.38
        ax3.bar(x - w/2, hi_means[top_feats], w, label="High Risk",  color="#ff4d6a", alpha=0.85)
        ax3.bar(x + w/2, lo_means[top_feats], w, label="Low Risk",   color="#39d98a", alpha=0.85)
        ax3.set_xticks(x)
        ax3.set_xticklabels(top_feats, rotation=45, ha="right", fontsize=7, color="#8ab0cc")
        ax3.tick_params(axis="y", colors="#8ab0cc", labelsize=8)
        for spine in ax3.spines.values():
            spine.set_edgecolor("#1a2d42")
        ax3.legend(fontsize=8, facecolor="#0d1520", edgecolor="#1a2d42", labelcolor="#c9d6e3")
        ax3.grid(axis="y", color="#1a2d42", linewidth=0.5, linestyle=":")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)
    else:
        st.info("Not enough records in both risk tiers to compare features.")

    # ── High-Risk Records Table ───────────────────────────────────────────────
    if show_raw:
        st.markdown('<p class="sec-head">⚠️ High-Risk Account Records</p>', unsafe_allow_html=True)

        if not high_risk_df.empty:
            display_cols = ["Unnamed: 0", "risk_score", "risk_level"] + feature_cols[:10]
            display_cols = [c for c in display_cols if c in high_risk_df.columns]

            st.dataframe(
                high_risk_df[display_cols]
                    .sort_values("risk_score", ascending=False)
                    .head(200)
                    .reset_index(drop=True)
                    .rename(columns={"Unnamed: 0": "record_id"}),
                use_container_width=True,
                height=340,
            )
            st.caption(f"Showing top 200 of {len(high_risk_df):,} high-risk records (risk score > {risk_threshold}). First 10 features shown.")

            # Download button
            csv_out = (
                high_risk_df[display_cols]
                .sort_values("risk_score", ascending=False)
                .rename(columns={"Unnamed: 0": "record_id"})
                .to_csv(index=False)
            )
            st.download_button(
                label="⬇️ Download High-Risk Records (CSV)",
                data=csv_out,
                file_name="cybershield_high_risk_accounts.csv",
                mime="text/csv",
            )
        else:
            st.success("✅ No records exceeded the high-risk threshold. Try lowering the threshold in the sidebar.")

    # ── Risk Level Breakdown ─────────────────────────────────────────────────
    st.markdown('<p class="sec-head">📋 Risk Level Breakdown</p>', unsafe_allow_html=True)
    breakdown = df["risk_level"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
    breakdown_df = pd.DataFrame({
        "Risk Level": breakdown.index,
        "Count":      breakdown.values,
        "Percentage": (breakdown.values / total_records * 100).round(2),
    })
    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

    # ── Alert Log ────────────────────────────────────────────────────────────
    render_alert_log_table()

else:
    # ── Landing screen ────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 3rem 1rem; color: #4a6a8a;">
      <p style="font-size:3rem; margin:0;">🛡️</p>
      <p style="font-family:'Share Tech Mono',monospace; font-size:1.1rem; color:#00d4ff; margin:0.5rem 0;">
        Upload your dataset to begin threat analysis
      </p>
      <p style="font-size:0.85rem; margin-top:0.8rem;">
        Accepts any CSV with numerical feature columns (F1 … FN format).<br>
        The model auto-selects the best-filled features and scores every account.
      </p>
    </div>
    """, unsafe_allow_html=True)
