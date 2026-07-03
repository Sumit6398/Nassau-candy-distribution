"""
streamlit_app.py
-----------------
Nassau Candy — Factory Reallocation & Shipping Optimization dashboard.

Run with:  streamlit run app/streamlit_app.py
On first run it will automatically build the processed dataset and train
the lead-time model if those artifacts don't exist yet (so a fresh
`git clone` works out of the box with no manual pipeline step).
"""

import os
import sys
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------
# Path setup -- make src/ importable regardless of cwd, and resolve all
# data/model/output paths relative to the project root.
# ----------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)

RAW_PATH = os.path.join(ROOT_DIR, "data", "Nassau_Candy_Distributor.csv")
PROCESSED_PATH = os.path.join(ROOT_DIR, "outputs", "processed_data.csv")
CLUSTERS_PATH = os.path.join(ROOT_DIR, "outputs", "route_clusters.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "lead_time_model.joblib")
META_PATH = os.path.join(ROOT_DIR, "models", "model_metadata.json")

st.set_page_config(
    page_title="Nassau Candy | Factory Optimization",
    page_icon="\U0001F36C",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Design system -- "shipping manifest" visual language: deep cocoa
# background, amber + teal control-panel accents, monospace data type.
# ----------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #1B1410;
    --panel: #251A14;
    --panel-2: #2F2018;
    --amber: #F2A93B;
    --teal: #4FBDBA;
    --red: #E0533D;
    --ink: #F5EFE6;
    --muted: #B8A99A;
    --hairline: #3C2C22;
}

html, body, [class*="css"]  { color: var(--ink); }
.stApp { background: var(--bg); }

h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }
p, li, span, div, label { font-family: 'Inter', sans-serif; }

.eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.72rem;
    color: var(--amber);
    border-bottom: 1px dashed var(--hairline);
    padding-bottom: 6px;
    margin-bottom: 10px;
}

.hero-title {
    font-size: 2.1rem;
    font-weight: 700;
    margin-bottom: 0.1rem;
    color: var(--ink);
}
.hero-sub {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--muted);
    font-size: 0.88rem;
    margin-bottom: 1.2rem;
}

/* Manifest-tag KPI cards */
.tag-card {
    background: var(--panel);
    border: 1px solid var(--hairline);
    border-left: 3px solid var(--amber);
    border-radius: 4px;
    padding: 14px 16px;
    position: relative;
}
.tag-card::after {
    content: "";
    position: absolute;
    top: 12px; right: 10px;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--bg);
    border: 1px solid var(--hairline);
}
.tag-label {
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    color: var(--muted);
}
.tag-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.7rem;
    font-weight: 600;
    color: var(--ink);
    margin-top: 4px;
}
.tag-value.pos { color: var(--teal); }
.tag-value.neg { color: var(--red); }

/* Badges for risk flags / factory tags */
.badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 2px 8px;
    border-radius: 3px;
    border: 1px solid currentColor;
}
.badge-low { color: var(--teal); }
.badge-caution { color: var(--amber); }
.badge-high { color: var(--red); }
.badge-current { color: var(--muted); }

section[data-testid="stSidebar"] {
    background: var(--panel);
    border-right: 1px solid var(--hairline);
}

.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--hairline); }
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    font-size: 0.74rem;
    letter-spacing: 0.06em;
    color: var(--muted);
    padding: 10px 14px;
}
.stTabs [aria-selected="true"] { color: var(--amber) !important; border-bottom: 2px solid var(--amber) !important; }

div[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--hairline);
    border-left: 3px solid var(--teal);
    border-radius: 4px;
    padding: 10px 14px;
}

footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = go.layout.Template(
    layout=dict(
        paper_bgcolor="#1B1410",
        plot_bgcolor="#1B1410",
        font=dict(family="Inter, sans-serif", color="#F5EFE6"),
        colorway=["#F2A93B", "#4FBDBA", "#E0533D", "#8C7A6B", "#7FB3D5", "#C97B4A"],
        xaxis=dict(gridcolor="#3C2C22", zerolinecolor="#3C2C22"),
        yaxis=dict(gridcolor="#3C2C22", zerolinecolor="#3C2C22"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
)

# ----------------------------------------------------------------------
# Data / model bootstrap (idempotent -- builds artifacts if missing)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Preparing data and training models (first run only)...")
def bootstrap():
    import data_prep
    import modeling
    import clustering

    if not os.path.exists(PROCESSED_PATH):
        data_prep.run_pipeline(raw_path=RAW_PATH, out_path=PROCESSED_PATH)
    if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
        df = modeling.load_processed(PROCESSED_PATH)
        run = modeling.train_all_models(df)
        modeling.save_artifacts(run, out_dir=os.path.join(ROOT_DIR, "models"))
    if not os.path.exists(CLUSTERS_PATH):
        clustering.run(processed_path=PROCESSED_PATH, out_path=CLUSTERS_PATH)
    return True


@st.cache_resource(show_spinner=False)
def get_engine():
    from scenario_engine import RecommendationEngine
    return RecommendationEngine(
        processed_data_path=PROCESSED_PATH,
        model_path=MODEL_PATH,
        metadata_path=META_PATH,
    )


@st.cache_data(show_spinner=False)
def load_processed():
    return pd.read_csv(PROCESSED_PATH)


@st.cache_data(show_spinner=False)
def load_clusters():
    return pd.read_csv(CLUSTERS_PATH)


@st.cache_data(show_spinner=False)
def all_recommendations(priority):
    eng = get_engine()
    return eng.recommend_all_products(priority=priority)


bootstrap()
engine = get_engine()
df = load_processed()
clusters = load_clusters()

from geocoding import FACTORY_COORDINATES, STATE_CENTROIDS
from data_prep import PRODUCT_FACTORY_MAP

PRODUCTS = sorted(PRODUCT_FACTORY_MAP.keys())
REGIONS = sorted(df["Region"].unique())
SHIP_MODES = sorted(df["Ship Mode"].unique())

# ----------------------------------------------------------------------
# Sidebar -- "control panel"
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="eyebrow">Route Parameters</div>', unsafe_allow_html=True)
    st.markdown("### Control Panel")

    selected_product = st.selectbox("Product", PRODUCTS, index=0)
    selected_regions = st.multiselect("Region filter (EDA views)", REGIONS, default=REGIONS)
    selected_modes = st.multiselect("Ship mode filter (EDA views)", SHIP_MODES, default=SHIP_MODES)

    st.markdown("---")
    st.markdown('<div class="eyebrow">Optimization Priority</div>', unsafe_allow_html=True)
    priority = st.slider(
        "Profit \u2190\u2192 Speed", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="0.0 = optimize purely for profit impact. 1.0 = optimize purely for lead-time reduction.",
    )
    st.caption(f"Weighting: **{round((1-priority)*100)}% profit** / **{round(priority*100)}% speed**")

    st.markdown("---")
    meta = json.load(open(META_PATH))
    st.markdown('<div class="eyebrow">Model</div>', unsafe_allow_html=True)
    st.caption(f"Best model: **{meta['best_model']}**")
    m = meta["metrics"][meta["best_model"]]
    st.caption(f"RMSE {m['RMSE']} \u00b7 MAE {m['MAE']} \u00b7 R\u00b2 {m['R2']}")

# ----------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------
st.markdown('<div class="eyebrow">Nassau Candy Distributor \u00b7 Decision Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Factory Reallocation &amp; Shipping Optimization</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">PREDICT \u2192 SIMULATE \u2192 RECOMMEND \u2014 lead-time modeling and factory '
    'reassignment scenarios across the 5-factory network</div>',
    unsafe_allow_html=True,
)

# Top-line KPI row
recs_all = all_recommendations(priority=priority)
avg_lt_reduction = recs_all.loc[recs_all["Action"] == "Reassign", "Lead Time Change %"].mean()
avg_profit_impact = recs_all["Profit Change %"].mean()
avg_confidence = recs_all["Confidence Score"].mean()
coverage = (recs_all["Action"] == "Reassign").mean() * 100

k1, k2, k3, k4 = st.columns(4)
def kpi_card(col, label, value, suffix="", cls=""):
    col.markdown(
        f"""<div class="tag-card"><div class="tag-label">{label}</div>
        <div class="tag-value {cls}">{value}{suffix}</div></div>""",
        unsafe_allow_html=True,
    )

kpi_card(k1, "Avg Lead Time Reduction", f"{avg_lt_reduction:.1f}", "%", "pos")
kpi_card(k2, "Avg Profit Impact", f"{avg_profit_impact:+.1f}", "% of sales", "pos" if avg_profit_impact >= 0 else "neg")
kpi_card(k3, "Avg Scenario Confidence", f"{avg_confidence:.2f}", "", "")
kpi_card(k4, "Recommendation Coverage", f"{coverage:.0f}", "%", "")

st.write("")

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab_overview, tab_sim, tab_whatif, tab_recs, tab_risk, tab_clusters = st.tabs(
    ["Overview", "Factory Simulator", "What-If Analysis", "Recommendations", "Risk & Impact", "Route Clusters"]
)

# ===== OVERVIEW =========================================================
with tab_overview:
    filtered = df[df["Region"].isin(selected_regions) & df["Ship Mode"].isin(selected_modes)]

    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown('<div class="eyebrow">Network Map</div>', unsafe_allow_html=True)
        state_vol = (
            filtered.groupby("State/Province")
            .agg(Orders=("Row ID", "count"), Lead_Time=("Lead Time Days", "mean"))
            .reset_index()
        )
        state_vol["lat"] = state_vol["State/Province"].map(lambda s: STATE_CENTROIDS[s][0])
        state_vol["lon"] = state_vol["State/Province"].map(lambda s: STATE_CENTROIDS[s][1])

        fig = go.Figure()
        fig.add_trace(go.Scattergeo(
            lat=state_vol["lat"], lon=state_vol["lon"],
            text=state_vol["State/Province"] + "<br>Orders: " + state_vol["Orders"].astype(str),
            marker=dict(size=(state_vol["Orders"] / state_vol["Orders"].max() * 28 + 4),
                        color="#4FBDBA", opacity=0.6, line=dict(width=0)),
            mode="markers", name="Customer demand", hoverinfo="text",
        ))
        fac_df = pd.DataFrame(
            [{"name": k, "lat": v[0], "lon": v[1]} for k, v in FACTORY_COORDINATES.items()]
        )
        fig.add_trace(go.Scattergeo(
            lat=fac_df["lat"], lon=fac_df["lon"], text=fac_df["name"],
            marker=dict(size=16, color="#F2A93B", symbol="diamond", line=dict(width=1, color="#1B1410")),
            mode="markers+text", textposition="top center", textfont=dict(color="#F5EFE6", size=10),
            name="Factories", hoverinfo="text",
        ))
        fig.update_geos(
            scope="north america", showland=True, landcolor="#251A14",
            showcountries=True, countrycolor="#3C2C22", showocean=True, oceancolor="#1B1410",
            showlakes=False, bgcolor="#1B1410",
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, height=420, margin=dict(l=0, r=0, t=10, b=0),
                           legend=dict(orientation="h", y=-0.05))
        st.plotly_chart(fig, width='stretch')

    with c2:
        st.markdown('<div class="eyebrow">Lead Time Distribution</div>', unsafe_allow_html=True)
        fig2 = px.histogram(filtered, x="Lead Time Days", nbins=30, template=PLOTLY_TEMPLATE)
        fig2.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        fig2.update_traces(marker_color="#F2A93B")
        st.plotly_chart(fig2, width='stretch')

        st.markdown('<div class="eyebrow">Avg Lead Time by Ship Mode</div>', unsafe_allow_html=True)
        sm = filtered.groupby("Ship Mode")["Lead Time Days"].mean().sort_values().reset_index()
        fig3 = px.bar(sm, x="Lead Time Days", y="Ship Mode", orientation="h", template=PLOTLY_TEMPLATE)
        fig3.update_traces(marker_color="#4FBDBA")
        fig3.update_layout(height=190, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig3, width='stretch')

    st.markdown('<div class="eyebrow">Division Performance</div>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    div_summary = filtered.groupby("Division").agg(
        Sales=("Sales", "sum"), Profit=("Adjusted Gross Profit", "sum"),
        Lead_Time=("Lead Time Days", "mean"), Orders=("Row ID", "count"),
    ).round(2).reset_index()
    with d1:
        fig4 = px.bar(div_summary, x="Division", y="Sales", template=PLOTLY_TEMPLATE, title="Total Sales")
        fig4.update_traces(marker_color="#F2A93B")
        fig4.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig4, width='stretch')
    with d2:
        fig5 = px.bar(div_summary, x="Division", y="Profit", template=PLOTLY_TEMPLATE, title="Adjusted Gross Profit")
        fig5.update_traces(marker_color="#4FBDBA")
        fig5.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig5, width='stretch')
    with d3:
        fig6 = px.bar(div_summary, x="Division", y="Lead_Time", template=PLOTLY_TEMPLATE, title="Avg Lead Time (days)")
        fig6.update_traces(marker_color="#E0533D")
        fig6.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig6, width='stretch')

# ===== FACTORY OPTIMIZATION SIMULATOR ==================================
with tab_sim:
    st.markdown(f'<div class="eyebrow">Selected Product</div><div class="hero-title" style="font-size:1.4rem;">{selected_product}</div>', unsafe_allow_html=True)
    comparison = engine.compare_all_factories(selected_product)
    current_factory = PRODUCT_FACTORY_MAP[selected_product]
    st.caption(f"Current factory: **{current_factory}**  \u00b7  Historical orders: **{int(comparison['Order Volume'].iloc[0])}**")

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown('<div class="eyebrow">Predicted Lead Time by Factory</div>', unsafe_allow_html=True)
        comp_sorted = comparison.sort_values("Avg Predicted Lead Time")
        colors = ["#F2A93B" if cur else "#4FBDBA" for cur in comp_sorted["Is Current"]]
        figA = go.Figure(go.Bar(
            x=comp_sorted["Avg Predicted Lead Time"], y=comp_sorted["Factory"], orientation="h",
            marker_color=colors, text=comp_sorted["Avg Predicted Lead Time"].round(2), textposition="outside",
        ))
        figA.update_layout(template=PLOTLY_TEMPLATE, height=320, margin=dict(l=10, r=10, t=10, b=10),
                            xaxis_title="Days")
        st.plotly_chart(figA, width='stretch')
        st.caption("\U0001F538 Amber = current factory \u00b7 \U0001F539 Teal = alternative")

    with cc2:
        st.markdown('<div class="eyebrow">Total Adjusted Profit by Factory</div>', unsafe_allow_html=True)
        colors2 = ["#F2A93B" if cur else "#4FBDBA" for cur in comp_sorted["Is Current"]]
        figB = go.Figure(go.Bar(
            x=comp_sorted["Total Adjusted Profit"], y=comp_sorted["Factory"], orientation="h",
            marker_color=colors2, text=comp_sorted["Total Adjusted Profit"].round(0), textposition="outside",
        ))
        figB.update_layout(template=PLOTLY_TEMPLATE, height=320, margin=dict(l=10, r=10, t=10, b=10),
                            xaxis_title="$ (historical orders)")
        st.plotly_chart(figB, width='stretch')

    st.markdown('<div class="eyebrow">Full Comparison</div>', unsafe_allow_html=True)
    show_cols = ["Factory", "Is Current", "Avg Predicted Lead Time", "Avg Distance Miles",
                 "Total Adjusted Profit", "Avg Profit Margin %", "Lead Time Change %", "Profit Change %"]
    st.dataframe(comparison[show_cols], width='stretch', hide_index=True)

# ===== WHAT-IF SCENARIO ANALYSIS =======================================
with tab_whatif:
    st.markdown('<div class="eyebrow">Current vs Recommended Assignment</div>', unsafe_allow_html=True)
    _, ranked_alt, ranked_all = engine.recommend(selected_product, priority=priority, top_n=3)
    keep = ranked_all.iloc[0]["Is Current"]
    current_row = ranked_all[ranked_all["Is Current"]].iloc[0]

    if keep:
        st.info(f"**{selected_product}** is already optimally assigned to **{current_factory}** "
                f"under the current profit/speed priority. No reassignment recommended.")
        target_row = current_row
    else:
        target_row = ranked_alt.iloc[0]
        st.success(f"Recommended: move **{selected_product}** from **{current_factory}** "
                    f"to **{target_row['Factory']}**")

    w1, w2, w3 = st.columns(3)
    w1.metric("Lead Time (days)", f"{target_row['Avg Predicted Lead Time']:.2f}",
               delta=f"{target_row['Lead Time Change %']:.1f}% vs current" if not keep else None,
               delta_color="inverse")
    w2.metric("Distance (miles)", f"{target_row['Avg Distance Miles']:.0f}")
    w3.metric("Profit Impact", f"{target_row['Profit Change %']:+.1f}% of sales" if not keep else "0.0%",
               delta=None)

    st.markdown('<div class="eyebrow">Before / After</div>', unsafe_allow_html=True)
    baf = pd.DataFrame({
        "Scenario": ["Current", "Recommended"],
        "Lead Time": [current_row["Avg Predicted Lead Time"], target_row["Avg Predicted Lead Time"]],
        "Profit": [current_row["Total Adjusted Profit"], target_row["Total Adjusted Profit"]],
    })
    bc1, bc2 = st.columns(2)
    with bc1:
        figC = px.bar(baf, x="Scenario", y="Lead Time", template=PLOTLY_TEMPLATE, text_auto=".2f")
        figC.update_traces(marker_color=["#8C7A6B", "#4FBDBA"])
        figC.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(figC, width='stretch')
    with bc2:
        figD = px.bar(baf, x="Scenario", y="Profit", template=PLOTLY_TEMPLATE, text_auto=".0f")
        figD.update_traces(marker_color=["#8C7A6B", "#F2A93B"])
        figD.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(figD, width='stretch')

    st.markdown('<div class="eyebrow">Top 3 Alternatives (ranked by current priority weighting)</div>', unsafe_allow_html=True)
    if not ranked_alt.empty:
        st.dataframe(
            ranked_alt[["Factory", "Avg Predicted Lead Time", "Lead Time Change %",
                        "Profit Change %", "Confidence Score", "Risk Flag", "Score"]],
            width='stretch', hide_index=True,
        )

# ===== RECOMMENDATION DASHBOARD ========================================
with tab_recs:
    st.markdown('<div class="eyebrow">Ranked Reassignment Suggestions \u2014 All Products</div>', unsafe_allow_html=True)
    st.caption("Priority weighting applied from the sidebar slider.")

    def badge(flag):
        cls = {"Low Risk": "badge-low", "Caution": "badge-caution",
               "High Risk": "badge-high", "N/A (Current)": "badge-current"}.get(flag, "badge-current")
        return f'<span class="badge {cls}">{flag}</span>'

    display = recs_all.copy()
    display_html = display.copy()
    display_html["Risk Flag"] = display_html["Risk Flag"].apply(badge)
    display_html["Lead Time Change %"] = display_html["Lead Time Change %"].map(lambda v: f"{v:+.1f}%")
    display_html["Profit Change %"] = display_html["Profit Change %"].map(lambda v: f"{v:+.1f}%")

    st.write(
        display_html.to_html(escape=False, index=False, classes="rec-table"),
        unsafe_allow_html=True,
    )
    st.markdown(
        "<style>.rec-table{width:100%;border-collapse:collapse;font-family:'IBM Plex Mono',monospace;"
        "font-size:0.82rem;} .rec-table th{text-align:left;color:#B8A99A;text-transform:uppercase;"
        "font-size:0.68rem;letter-spacing:.06em;border-bottom:1px solid #3C2C22;padding:8px 10px;}"
        ".rec-table td{padding:8px 10px;border-bottom:1px solid #2F2018;}</style>",
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown('<div class="eyebrow">Expected Lead Time Improvement by Product</div>', unsafe_allow_html=True)
    reassign_only = recs_all[recs_all["Action"] == "Reassign"].sort_values("Lead Time Change %")
    if not reassign_only.empty:
        figE = px.bar(reassign_only, x="Lead Time Change %", y="Product Name", orientation="h",
                      template=PLOTLY_TEMPLATE, color="Risk Flag",
                      color_discrete_map={"Low Risk": "#4FBDBA", "Caution": "#F2A93B", "High Risk": "#E0533D"})
        figE.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(figE, width='stretch')

    csv = recs_all.to_csv(index=False).encode("utf-8")
    st.download_button("Download recommendations (CSV)", csv, "recommendations.csv", "text/csv")

# ===== RISK & IMPACT PANEL =============================================
with tab_risk:
    st.markdown('<div class="eyebrow">Profit Impact Alerts</div>', unsafe_allow_html=True)
    risky = recs_all[recs_all["Risk Flag"].isin(["High Risk", "Caution"])]
    if risky.empty:
        st.success("No high-risk or caution-flagged reassignments under the current priority weighting.")
    else:
        for _, row in risky.iterrows():
            color = "#E0533D" if row["Risk Flag"] == "High Risk" else "#F2A93B"
            st.markdown(
                f"""<div class="tag-card" style="border-left-color:{color}; margin-bottom:8px;">
                <span class="badge {'badge-high' if row['Risk Flag']=='High Risk' else 'badge-caution'}">{row['Risk Flag']}</span>
                &nbsp; <b>{row['Product Name']}</b> \u2192 {row['Recommended Factory']}
                &nbsp;|&nbsp; Lead time {row['Lead Time Change %']:+.1f}% &nbsp;|&nbsp;
                Profit impact {row['Profit Change %']:+.1f}% of sales
                </div>""",
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown('<div class="eyebrow">Profit Impact Distribution</div>', unsafe_allow_html=True)
    figF = px.bar(
        recs_all.sort_values("Profit Change %"), x="Profit Change %", y="Product Name", orientation="h",
        template=PLOTLY_TEMPLATE, color="Risk Flag",
        color_discrete_map={"Low Risk": "#4FBDBA", "Caution": "#F2A93B", "High Risk": "#E0533D", "N/A (Current)": "#8C7A6B"},
    )
    figF.add_vline(x=0, line_color="#F5EFE6", line_width=1)
    figF.update_layout(height=440, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(figF, width='stretch')

    st.markdown('<div class="eyebrow">Scenario Confidence vs Order Volume</div>', unsafe_allow_html=True)
    st.caption("Low order-volume products yield lower-confidence simulations \u2014 treat their recommendations as directional, not final.")
    vol_lookup = df.groupby("Product Name")["Row ID"].count().rename("Order Volume")
    conf_df = recs_all.merge(vol_lookup, on="Product Name")
    figG = px.scatter(conf_df, x="Order Volume", y="Confidence Score", color="Risk Flag",
                      hover_name="Product Name", template=PLOTLY_TEMPLATE, size_max=14,
                      color_discrete_map={"Low Risk": "#4FBDBA", "Caution": "#F2A93B", "High Risk": "#E0533D", "N/A (Current)": "#8C7A6B"})
    figG.update_traces(marker=dict(size=12))
    figG.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(figG, width='stretch')

# ===== ROUTE CLUSTERS ====================================================
with tab_clusters:
    st.markdown('<div class="eyebrow">Route Performance Clusters</div>', unsafe_allow_html=True)
    st.caption("Routes (Factory \u00d7 Region \u00d7 Ship Mode) grouped by lead time, distance, margin and volume similarity.")

    label_counts = clusters["Cluster Label"].value_counts().reset_index()
    label_counts.columns = ["Cluster Label", "Routes"]
    figH = px.bar(label_counts, x="Routes", y="Cluster Label", orientation="h", template=PLOTLY_TEMPLATE)
    figH.update_traces(marker_color="#F2A93B")
    figH.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(figH, width='stretch')

    figI = px.scatter(
        clusters, x="Avg_Distance", y="Avg_Lead_Time", size="Order_Volume", color="Cluster Label",
        hover_data=["Current Factory", "Region", "Ship Mode", "Avg_Profit_Margin"],
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=["#4FBDBA", "#F2A93B", "#E0533D", "#8C7A6B"],
    )
    figI.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title="Avg Distance (miles)", yaxis_title="Avg Lead Time (days)")
    st.plotly_chart(figI, width='stretch')

    st.markdown('<div class="eyebrow">Consistently Slow / High-Risk Routes</div>', unsafe_allow_html=True)
    slow = clusters[clusters["Cluster Label"].str.contains("Slow|Risk", regex=True)].sort_values("Avg_Lead_Time", ascending=False)
    st.dataframe(slow, width='stretch', hide_index=True)
