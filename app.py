import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import io
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plant Health Analyzer",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme constants ───────────────────────────────────────────────────────────
BG    = "#0d1117"
PANEL = "#161b22"
GRID  = "#21262d"
TEXT  = "#e6edf3"
MUTED = "#8b949e"
HC    = {"Healthy": "#2ecc71", "Unhealthy": "#f39c12", "Dry": "#e74c3c"}
HM    = {"Healthy": "o",       "Unhealthy": "s",        "Dry": "^"}

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Lora:ital,wght@0,500;1,400&display=swap');

html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
h1, h2, h3 { font-family: 'Lora', serif !important; }

.stApp { background-color: #0d1117; color: #e6edf3; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #21262d;
}

.metric-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.metric-card .label {
    font-size: 10px;
    letter-spacing: .08em;
    color: #8b949e;
    margin-bottom: 4px;
    text-transform: uppercase;
}
.metric-card .value {
    font-size: 28px;
    font-weight: 500;
}

.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
}
.badge-strong   { background: #1a4a2a; color: #2ecc71; }
.badge-moderate { background: #4a3a0a; color: #f39c12; }
.badge-weak     { background: #4a1a1a; color: #e74c3c; }

div[data-testid="stDataFrame"] { border-radius: 8px; }

.stButton > button {
    background: #161b22;
    color: #e6edf3;
    border: 1px solid #30363d;
    border-radius: 6px;
    font-family: 'DM Mono', monospace;
    font-size: 13px;
}
.stButton > button:hover { background: #21262d; border-color: #58a6ff; }

.flag-match    { color: #2ecc71; font-weight: 500; }
.flag-mismatch { color: #e74c3c; font-weight: 500; }

.section-divider {
    border: none;
    border-top: 1px solid #21262d;
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

def set_dark_axes(ax):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    ax.grid(True, color=GRID, linewidth=0.4, alpha=0.5)


def lin_reg(x, y):
    if len(x) < 2:
        return 0, 0, 1, 0, 0
    slope, intercept, r, p, se = stats.linregress(x, y)
    return r**2, r, p, slope, intercept


def r2_badge(r2):
    if r2 >= 0.6:
        return f'<span class="badge badge-strong">R²={r2:.3f} Strong</span>'
    elif r2 >= 0.3:
        return f'<span class="badge badge-moderate">R²={r2:.3f} Moderate</span>'
    else:
        return f'<span class="badge badge-weak">R²={r2:.3f} Weak</span>'


def scatter_ax(ax, df, x_col, y_col, xlabel, ylabel, title):
    x_all = df[x_col].values
    y_all = df[y_col].values
    r2, r, p, slope, intercept = lin_reg(x_all, y_all)

    xfit = np.linspace(x_all.min() - x_all.max() - x_all.min() * 0.1,
                       x_all.max() + x_all.max() - x_all.min() * 0.1, 200)
    yfit = slope * xfit + intercept
    if len(x_all) >= 2:
        n = len(x_all)
        t_crit = stats.t.ppf(0.975, df=n - 2)
        ci = t_crit * stats.linregress(x_all, y_all)[4] * np.sqrt(
            1/n + (xfit - x_all.mean())**2 / np.sum((x_all - x_all.mean())**2))
        ax.fill_between(xfit, yfit - ci, yfit + ci, color="#58a6ff", alpha=0.10, zorder=1)
    ax.plot(xfit, yfit, color="#58a6ff", linewidth=1.4, linestyle="--", alpha=0.75, zorder=2)

    for h in ["Healthy", "Unhealthy", "Dry"]:
        mask = df["Health"] == h
        sub = df[mask]
        ax.scatter(sub[x_col], sub[y_col],
                   color=HC[h], marker=HM[h], s=90,
                   alpha=0.95, zorder=4, edgecolors="white", linewidths=0.5, label=h)
        for _, row in sub.iterrows():
            ax.annotate(row["Plant"].replace("Plant ", "P"),
                        (row[x_col], row[y_col]),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=6.5, color=HC[h], alpha=0.9)

    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    badge_c = "#2ecc71" if r2 >= 0.5 else ("#f39c12" if r2 >= 0.25 else "#e74c3c")
    ax.text(0.03, 0.97,
            f"R² = {r2:.3f}\nr  = {r:.3f}\np  = {p:.3f} {sig}",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
            color=badge_c, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=BG,
                      edgecolor=badge_c, alpha=0.9))

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, pad=8, color=TEXT)
    set_dark_axes(ax)
    return r2


def classify_plant(ndvi, gndvi, ndre, spad, temp):
    def vote_ndvi(v): return "Healthy" if v >= 0.60 else ("Unhealthy" if v >= 0.40 else "Dry")
    def vote_gndvi(v): return "Healthy" if v >= 0.55 else ("Unhealthy" if v >= 0.35 else "Dry")
    def vote_ndre(v): return "Healthy" if v >= 0.15 else ("Unhealthy" if v >= 0.10 else "Dry")
    def vote_spad(v): return "Healthy" if v >= 45 else ("Unhealthy" if v >= 30 else "Dry")
    def vote_temp(v): return "Healthy" if v <= 32 else ("Unhealthy" if v <= 35 else "Dry")
    votes = [vote_ndvi(ndvi), vote_gndvi(gndvi), vote_ndre(ndre),
             vote_spad(spad), vote_temp(temp)]
    count = Counter(votes)
    winner = count.most_common(1)[0][0]
    conf = round(count.most_common(1)[0][1] / 5 * 100, 1)
    breakdown = f"H:{count.get('Healthy',0)} U:{count.get('Unhealthy',0)} D:{count.get('Dry',0)}"
    return winner, conf, breakdown, votes


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🌿 Plant Health Analyzer")
    st.markdown(
        "<p style='color:#8b949e;font-size:12px;margin-bottom:1.5rem'>"
        "Multi-sensor correlation dashboard<br>NDVI · GNDVI · NDRE · SPAD · Thermal</p>",
        unsafe_allow_html=True
    )

    st.markdown("### Navigation")
    page = st.radio("", [
        "📥  Data Entry",
        "🔬  Formula Classifier",
        "📊  Correlation Analysis",
        "🗺️  Spatial Map",
        "📈  R² Rankings",
    ], label_visibility="collapsed")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("### Upload Excel")
    uploaded = st.file_uploader("Upload your Data.xlsx", type=["xlsx", "xls"],
                                 label_visibility="collapsed")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#8b949e;font-size:10px;line-height:1.6'>"
        "Threshold references<br>"
        "NDVI: Tucker (1979)<br>"
        "SPAD: Peng et al. (1993)<br>"
        "GNDVI: Gitelson et al. (1996)<br>"
        "NDRE: Barnes et al. (2000)</p>",
        unsafe_allow_html=True
    )

# ── Default sample data ───────────────────────────────────────────────────────
SAMPLE = pd.DataFrame([
    {"Plant": "Plant A", "Latitude": 14.083005, "Longitude": 100.611228,
     "NDVI": 0.61, "GNDVI": 0.61, "NDRE": 0.12, "SPAD": 44.0,
     "Temperature": 33.1, "Health": "Healthy"},
    {"Plant": "Plant B", "Latitude": 14.082983, "Longitude": 100.611267,
     "NDVI": 0.83, "GNDVI": 0.83, "NDRE": 0.18, "SPAD": 51.5,
     "Temperature": 31.6, "Health": "Healthy"},
    {"Plant": "Plant C", "Latitude": 14.082987, "Longitude": 100.611282,
     "NDVI": 0.39, "GNDVI": 0.28, "NDRE": 0.07, "SPAD": 54.3,
     "Temperature": 29.8, "Health": "Healthy"},
    {"Plant": "Plant D", "Latitude": 14.082955, "Longitude": 100.611412,
     "NDVI": 0.62, "GNDVI": 0.58, "NDRE": 0.13, "SPAD": 47.3,
     "Temperature": 34.0, "Health": "Unhealthy"},
    {"Plant": "Plant E", "Latitude": 14.082980, "Longitude": 100.611422,
     "NDVI": 0.56, "GNDVI": 0.53, "NDRE": 0.13, "SPAD": 45.1,
     "Temperature": 33.3, "Health": "Unhealthy"},
    {"Plant": "Plant F", "Latitude": 14.083002, "Longitude": 100.611427,
     "NDVI": 0.58, "GNDVI": 0.55, "NDRE": 0.15, "SPAD": 44.9,
     "Temperature": 34.5, "Health": "Dry"},
    {"Plant": "Plant G", "Latitude": 14.083045, "Longitude": 100.611417,
     "NDVI": 0.34, "GNDVI": 0.48, "NDRE": 0.12, "SPAD": 26.0,
     "Temperature": 34.8, "Health": "Dry"},
    {"Plant": "Plant H", "Latitude": 14.082963, "Longitude": 100.611237,
     "NDVI": 0.67, "GNDVI": 0.56, "NDRE": 0.09, "SPAD": 16.1,
     "Temperature": 36.1, "Health": "Dry"},
    {"Plant": "Plant I", "Latitude": 14.082958, "Longitude": 100.611205,
     "NDVI": 0.59, "GNDVI": 0.56, "NDRE": 0.14, "SPAD": 30.9,
     "Temperature": 31.9, "Health": "Unhealthy"},
    {"Plant": "Plant J", "Latitude": 14.082947, "Longitude": 100.611317,
     "NDVI": 0.78, "GNDVI": 0.68, "NDRE": 0.14, "SPAD": 40.3,
     "Temperature": 29.6, "Health": "Healthy"},
])

# ── Load data ─────────────────────────────────────────────────────────────────
if uploaded is not None:
    try:
        df_raw = pd.read_excel(uploaded)
        df_raw.columns = df_raw.columns.str.strip()
        df_raw = df_raw.rename(columns={
            "PLANTS": "Plant", "SPAD Meter": "SPAD",
            "Personal_assumption_classification": "Health",
            "Longtitude": "Longitude", "Longitude": "Longitude",
            "Latitude": "Latitude",
        })
        df_raw["Health"] = df_raw["Health"].str.strip()
        df = df_raw[["Plant","Latitude","Longitude","NDVI","GNDVI","NDRE","SPAD","Temperature","Health"]].copy()
        st.sidebar.success(f"Loaded {len(df)} plants from file")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        df = SAMPLE.copy()
else:
    df = SAMPLE.copy()

# Compute composite health score
scaler = MinMaxScaler()
temp_norm = 1 - scaler.fit_transform(df[["Temperature"]].values)
veg_norm  = scaler.fit_transform(df[["NDVI", "GNDVI", "NDRE", "SPAD"]].values)
df["Health_Score"] = (veg_norm.mean(axis=1) * 0.7 + temp_norm.flatten() * 0.3).round(3)

features     = ["NDVI", "GNDVI", "NDRE", "SPAD", "Temperature"]
feat_display = ["NDVI", "GNDVI", "NDRE", "SPAD", "Temp (°C)"]

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Data Entry
# ═══════════════════════════════════════════════════════════════════════════════
if "Data Entry" in page:
    st.markdown("## 📥 Data Entry")
    st.markdown(
        "<p style='color:#8b949e;font-size:13px;margin-bottom:1.5rem'>"
        "Edit the table below or upload an Excel file from the sidebar. "
        "Add rows for each plant with all sensor readings.</p>",
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="label">Total Plants</div><div class="value">{len(df)}</div></div>', unsafe_allow_html=True)
    with col2:
        h_count = (df["Health"] == "Healthy").sum()
        st.markdown(f'<div class="metric-card"><div class="label">Healthy</div><div class="value" style="color:#2ecc71">{h_count}</div></div>', unsafe_allow_html=True)
    with col3:
        u_count = (df["Health"] == "Unhealthy").sum()
        st.markdown(f'<div class="metric-card"><div class="label">Unhealthy</div><div class="value" style="color:#f39c12">{u_count}</div></div>', unsafe_allow_html=True)
    with col4:
        d_count = (df["Health"] == "Dry").sum()
        st.markdown(f'<div class="metric-card"><div class="label">Dry</div><div class="value" style="color:#e74c3c">{d_count}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    edited = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Plant":        st.column_config.TextColumn("Plant", width="small"),
            "Latitude":     st.column_config.NumberColumn("Latitude",  format="%.6f"),
            "Longitude":    st.column_config.NumberColumn("Longitude", format="%.6f"),
            "NDVI":         st.column_config.NumberColumn("NDVI",  min_value=0.0, max_value=1.0, format="%.3f"),
            "GNDVI":        st.column_config.NumberColumn("GNDVI", min_value=0.0, max_value=1.0, format="%.3f"),
            "NDRE":         st.column_config.NumberColumn("NDRE",  min_value=0.0, max_value=1.0, format="%.3f"),
            "SPAD":         st.column_config.NumberColumn("SPAD",  min_value=0.0, max_value=100.0, format="%.1f"),
            "Temperature":  st.column_config.NumberColumn("Temp (°C)", format="%.1f"),
            "Health":       st.column_config.SelectboxColumn("Classification",
                                options=["Healthy", "Unhealthy", "Dry"]),
            "Health_Score": st.column_config.NumberColumn("Health Score", format="%.3f", disabled=True),
        },
        hide_index=True,
        key="data_editor"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    col_dl1, col_dl2 = st.columns([1, 5])
    with col_dl1:
        buf = io.BytesIO()
        edited.to_excel(buf, index=False)
        st.download_button("⬇ Download Excel", buf.getvalue(),
                           file_name="plant_data.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("### Summary Statistics")
    st.dataframe(
        df[["NDVI","GNDVI","NDRE","SPAD","Temperature","Health_Score"]].describe().round(3),
        use_container_width=True
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Formula Classifier
# ═══════════════════════════════════════════════════════════════════════════════
elif "Formula Classifier" in page:
    st.markdown("## 🔬 Formula-Based Classifier")
    st.markdown(
        "<p style='color:#8b949e;font-size:13px;margin-bottom:1rem'>"
        "Each index independently votes using scientific thresholds. "
        "Majority vote determines the formula class, then compared against your human label.</p>",
        unsafe_allow_html=True
    )

    with st.expander("📋 Threshold Reference Table", expanded=False):
        thresh_df = pd.DataFrame({
            "Index":     ["NDVI",  "GNDVI", "NDRE",         "SPAD",    "Temperature"],
            "Healthy":   ["≥ 0.60","≥ 0.55","≥ 0.15",       "≥ 45",    "≤ 32°C"],
            "Unhealthy": ["0.40–0.59","0.35–0.54","0.10–0.14","30–44", "32–35°C"],
            "Dry":       ["< 0.40","< 0.35","< 0.10",        "< 30",    "> 35°C"],
            "Reference": ["Tucker (1979)","Gitelson et al. (1996)",
                          "Barnes et al. (2000)","Peng et al. (1993)","CWSI Idso (1981)"],
        })
        st.dataframe(thresh_df, use_container_width=True, hide_index=True)

    results = []
    for _, row in df.iterrows():
        formula, conf, breakdown, votes = classify_plant(
            row["NDVI"], row["GNDVI"], row["NDRE"], row["SPAD"], row["Temperature"])
        match = formula.strip().lower() == str(row["Health"]).strip().lower()
        results.append({
            "Plant":         row["Plant"],
            "Human Label":   row["Health"],
            "Formula Class": formula,
            "Confidence %":  conf,
            "Vote Breakdown":breakdown,
            "NDVI Vote":     votes[0],
            "GNDVI Vote":    votes[1],
            "NDRE Vote":     votes[2],
            "SPAD Vote":     votes[3],
            "Temp Vote":     votes[4],
            "Match":         match,
        })

    res_df = pd.DataFrame(results)
    matches   = res_df["Match"].sum()
    mismatches = (~res_df["Match"]).sum()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="label">Total Classified</div><div class="value">{len(res_df)}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="label">Matches</div><div class="value" style="color:#2ecc71">{matches}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="label">Flagged for Review</div><div class="value" style="color:#e74c3c">{mismatches}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    def color_match(val):
        if val is True:  return "color: #2ecc71; font-weight: 600"
        if val is False: return "color: #e74c3c; font-weight: 600"
        return ""

    styled = res_df[["Plant","Human Label","Formula Class","Confidence %",
                      "Vote Breakdown","Match"]].style.map(
        color_match, subset=["Match"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    flagged = res_df[~res_df["Match"]]
    if len(flagged):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### ⚠️ Flagged Plants — Detailed Review")
        for _, row in flagged.iterrows():
            with st.expander(f"🔴 {row['Plant']}  ·  Human: {row['Human Label']}  →  Formula: {row['Formula Class']} ({row['Confidence %']}%)"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Index votes:**")
                    vote_data = {
                        "Index": ["NDVI","GNDVI","NDRE","SPAD","Temp"],
                        "Vote":  [row["NDVI Vote"],row["GNDVI Vote"],row["NDRE Vote"],row["SPAD Vote"],row["Temp Vote"]]
                    }
                    st.dataframe(pd.DataFrame(vote_data), hide_index=True)
                with c2:
                    plant_row = df[df["Plant"]==row["Plant"]].iloc[0]
                    st.markdown("**Raw sensor values:**")
                    vals = pd.DataFrame({
                        "Sensor": ["NDVI","GNDVI","NDRE","SPAD","Temp"],
                        "Value":  [plant_row["NDVI"],plant_row["GNDVI"],plant_row["NDRE"],
                                   plant_row["SPAD"],plant_row["Temperature"]]
                    })
                    st.dataframe(vals, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    buf2 = io.BytesIO()
    res_df.to_excel(buf2, index=False)
    st.download_button("⬇ Download Classification Results",
                       buf2.getvalue(), file_name="classification_results.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Correlation Analysis
# ═══════════════════════════════════════════════════════════════════════════════
elif "Correlation Analysis" in page:
    st.markdown("## 📊 Correlation Analysis")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Correlation Heatmap", "Temp vs Indices", "Index Cross-Corr", "vs SPAD Validation"
    ])

    # ── Tab 1: Heatmap ────────────────────────────────────────────────────────
    with tab1:
        st.markdown("#### R² Correlation Heatmap — All Index Pairs")
        n = len(features)
        r2_mat = np.zeros((n, n))
        for i, fi in enumerate(features):
            for j, fj in enumerate(features):
                if i == j:
                    r2_mat[i, j] = 1.0
                else:
                    r2_mat[i, j], *_ = lin_reg(df[fi].values, df[fj].values)

        fig, ax = plt.subplots(figsize=(7, 5.5))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(PANEL)
        cmap = LinearSegmentedColormap.from_list(
            "r2", ["#161b22", "#1a3a5c", "#2563eb", "#2ecc71"])
        im = ax.imshow(r2_mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(n):
            for j in range(n):
                v = r2_mat[i, j]
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if v < 0.7 else "#0d1117")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(feat_display, color=TEXT, fontsize=10)
        ax.set_yticklabels(feat_display, color=TEXT, fontsize=10)
        ax.tick_params(colors=MUTED, length=0)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("R² Value", color=MUTED, fontsize=9)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=MUTED)
        cbar.outline.set_edgecolor(GRID)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown(
            "🟢 **Green** = Strong (R² ≥ 0.60) &nbsp;·&nbsp; "
            "🟡 **Amber** = Moderate (0.30–0.59) &nbsp;·&nbsp; "
            "🔴 **Red** = Weak (< 0.30)"
        )

        # Print table too
        r2_df = pd.DataFrame(r2_mat.round(3), index=feat_display, columns=feat_display)
        st.dataframe(r2_df, use_container_width=True)

    # ── Tab 2: Temp vs Indices ────────────────────────────────────────────────
    with tab2:
        st.markdown("#### Canopy Temperature vs Vegetation Indices")
        pairs = [
            ("Temperature","NDVI","Temp vs NDVI","Canopy Temp (°C)","NDVI"),
            ("Temperature","GNDVI","Temp vs GNDVI","Canopy Temp (°C)","GNDVI"),
            ("Temperature","NDRE","Temp vs NDRE","Canopy Temp (°C)","NDRE"),
            ("Temperature","SPAD","Temp vs SPAD","Canopy Temp (°C)","SPAD"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.patch.set_facecolor(BG)
        for ax, (xc, yc, t, xl, yl) in zip(axes.flat, pairs):
            scatter_ax(ax, df, xc, yc, xl, yl, t)
        handles = [mpatches.Patch(color=c, label=h) for h, c in HC.items()]
        fig.legend(handles=handles, loc="lower center", ncol=3,
                   framealpha=0.3, facecolor=PANEL, labelcolor=TEXT,
                   fontsize=9, bbox_to_anchor=(0.5, -0.02))
        plt.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Tab 3: Index Cross-Corr ───────────────────────────────────────────────
    with tab3:
        st.markdown("#### Vegetation Index Cross-Correlations")
        cross_pairs = [
            ("NDVI","GNDVI","NDVI vs GNDVI","NDVI","GNDVI"),
            ("NDVI","NDRE","NDVI vs NDRE","NDVI","NDRE"),
            ("GNDVI","NDRE","GNDVI vs NDRE","GNDVI","NDRE"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.patch.set_facecolor(BG)
        for ax, (xc, yc, t, xl, yl) in zip(axes, cross_pairs):
            r2 = scatter_ax(ax, df, xc, yc, xl, yl, t)
        handles = [mpatches.Patch(color=c, label=h) for h, c in HC.items()]
        fig.legend(handles=handles, loc="lower center", ncol=3,
                   framealpha=0.3, facecolor=PANEL, labelcolor=TEXT,
                   fontsize=9, bbox_to_anchor=(0.5, -0.04))
        plt.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Tab 4: SPAD Validation ────────────────────────────────────────────────
    with tab4:
        st.markdown("#### Remote Sensor Validation — Plant-o-Meter vs SPAD")
        st.markdown(
            "<p style='color:#8b949e;font-size:12px'>A strong R² here means your "
            "Plant-o-Meter spectral reading can <b>substitute the contact SPAD meter</b> — "
            "a publishable sensor validation finding.</p>",
            unsafe_allow_html=True
        )
        spad_pairs = [
            ("NDVI","SPAD","NDVI vs SPAD","NDVI","SPAD"),
            ("GNDVI","SPAD","GNDVI vs SPAD","GNDVI","SPAD"),
            ("NDRE","SPAD","NDRE vs SPAD","NDRE","SPAD"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.patch.set_facecolor(BG)
        for ax, (xc, yc, t, xl, yl) in zip(axes, spad_pairs):
            scatter_ax(ax, df, xc, yc, xl, yl, t)
        handles = [mpatches.Patch(color=c, label=h) for h, c in HC.items()]
        fig.legend(handles=handles, loc="lower center", ncol=3,
                   framealpha=0.3, facecolor=PANEL, labelcolor=TEXT,
                   fontsize=9, bbox_to_anchor=(0.5, -0.04))
        plt.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Spatial Map
# ═══════════════════════════════════════════════════════════════════════════════
elif "Spatial Map" in page:
    st.markdown("## 🗺️ Spatial Map of Plant Distribution")
    st.markdown(
        "<p style='color:#8b949e;font-size:13px;margin-bottom:1rem'>"
        "Three spatial panels: NDVI distribution, SPAD chlorophyll, and "
        "composite health score. Coordinates displayed as relative offset (meters) "
        "from center.</p>",
        unsafe_allow_html=True
    )

    lat  = df["Latitude"].values
    lon  = df["Longitude"].values
    lat_plot = (lat - lat.mean()) * 1000
    lon_plot = (lon - lon.mean()) * 1000

    panel_configs = [
        {"title": "NDVI Distribution", "col": "NDVI",
         "cmap": LinearSegmentedColormap.from_list("ndvi",["#c0392b","#f39c12","#2ecc71"]),
         "size_scale": 600, "label": "NDVI Value"},
        {"title": "SPAD (Chlorophyll)", "col": "SPAD",
         "cmap": LinearSegmentedColormap.from_list("spad",["#e74c3c","#e67e22","#27ae60"]),
         "size_scale": 12, "label": "SPAD Value"},
        {"title": "Composite Health Score", "col": "Health_Score",
         "cmap": LinearSegmentedColormap.from_list("health",["#c0392b","#f39c12","#27ae60"]),
         "size_scale": 800, "label": "Score (0–1)"},
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Spatial Plant Distribution — Greenhouse Area",
                 color=TEXT, fontsize=13, fontweight="bold", y=1.01)

    for ax, cfg in zip(axes, panel_configs):
        ax.set_facecolor("#0a0f1a")
        col_vals = df[cfg["col"]].values
        mn, mx = col_vals.min(), col_vals.max()
        sizes = ((col_vals - mn) / (mx - mn + 1e-9)) * cfg["size_scale"] + 80

        sc = ax.scatter(lon_plot, lat_plot, c=col_vals, cmap=cfg["cmap"],
                        s=sizes, alpha=0.9, edgecolors="white", linewidths=0.6, zorder=3)
        for _, row in df.iterrows():
            lx = (row["Longitude"] - lon.mean()) * 1000
            ly = (row["Latitude"]  - lat.mean()) * 1000
            ax.annotate(row["Plant"].replace("Plant ", "P"),
                        (lx, ly), textcoords="offset points", xytext=(5, 5),
                        fontsize=7, color="white", fontweight="bold", alpha=0.9)
        for h in ["Healthy","Unhealthy","Dry"]:
            mask = df["Health"] == h
            lx = (df.loc[mask,"Longitude"] - lon.mean()) * 1000
            ly = (df.loc[mask,"Latitude"]  - lat.mean()) * 1000
            ax.scatter(lx, ly, s=sizes[mask.values]+120, facecolors="none",
                       edgecolors=HC[h], linewidths=1.5, zorder=2, alpha=0.8)

        cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label(cfg["label"], color=MUTED, fontsize=8)
        plt.setp(plt.getp(cbar.ax.axes,"yticklabels"), color=MUTED)
        cbar.outline.set_edgecolor(GRID)
        ax.set_title(cfg["title"], color=TEXT, fontsize=10, pad=10)
        ax.set_xlabel("Relative Easting (m)", color=MUTED, fontsize=8)
        ax.set_ylabel("Relative Northing (m)", color=MUTED, fontsize=8)
        ax.tick_params(colors=MUTED, labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)

    handles = [mpatches.Patch(color=c, label=f"{h} (ring)") for h, c in HC.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               framealpha=0.3, facecolor=PANEL, labelcolor=TEXT,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(pad=1.5)
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: R² Rankings
# ═══════════════════════════════════════════════════════════════════════════════
elif "R² Rankings" in page:
    st.markdown("## 📈 R² Rankings — All Pairwise Correlations")
    st.markdown(
        "<p style='color:#8b949e;font-size:13px;margin-bottom:1rem'>"
        "Every possible index pair ranked from strongest to weakest. "
        "Use this to identify which sensors carry the most predictive power.</p>",
        unsafe_allow_html=True
    )

    all_cols   = ["NDVI","GNDVI","NDRE","SPAD","Temperature","Health_Score"]
    all_labels = ["NDVI","GNDVI","NDRE","SPAD","Temp","Health Score"]
    all_pairs, all_r2 = [], []
    for i, a in enumerate(all_cols):
        for b in all_cols[i+1:]:
            r2, r, p, *_ = lin_reg(df[a].values, df[b].values)
            la = all_labels[i]; lb = all_labels[all_cols.index(b)]
            all_pairs.append({"Pair": f"{la}  ↔  {lb}", "R²": round(r2, 3),
                               "r": round(r, 3), "p-value": round(p, 4),
                               "Strength": "Strong" if r2>=0.6 else ("Moderate" if r2>=0.3 else "Weak")})

    rank_df = pd.DataFrame(all_pairs).sort_values("R²", ascending=False).reset_index(drop=True)
    rank_df.index += 1

    fig, ax = plt.subplots(figsize=(10, max(6, len(rank_df)*0.42)))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    colors = ["#27ae60" if v>=0.6 else ("#e8a020" if v>=0.3 else "#e74c3c")
              for v in rank_df["R²"]]
    bars = ax.barh(rank_df["Pair"][::-1], rank_df["R²"][::-1],
                   color=colors[::-1], edgecolor=BG, height=0.65)
    for bar, val in zip(bars, rank_df["R²"][::-1]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", color=TEXT, fontsize=9, fontweight="bold")
    ax.axvline(0.6, color="#2ecc71", linewidth=1, linestyle=":", alpha=0.6, label="Strong (0.60)")
    ax.axvline(0.3, color="#e8a020", linewidth=1, linestyle=":", alpha=0.6, label="Moderate (0.30)")
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("R² Value", color=MUTED, fontsize=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.grid(True, axis="x", color=GRID, linewidth=0.4, alpha=0.5)
    ax.legend(framealpha=0.3, facecolor=PANEL, labelcolor=TEXT, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(rank_df, use_container_width=True)

    buf3 = io.BytesIO()
    rank_df.to_excel(buf3, index=False)
    st.download_button("⬇ Download R² Rankings",
                       buf3.getvalue(), file_name="r2_rankings.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
