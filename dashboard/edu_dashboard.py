"""
edu_dashboard.py
Nigeria Education Framework — Interactive Streamlit Dashboard
Run: streamlit run dashboard\edu_dashboard.py
"""

import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

G3 = Path(__file__).resolve().parent.parent

st.set_page_config(
    page_title="Nigeria Education Framework",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1a3a5c;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666677;
        margin-top: 4px;
    }
    .stApp { background-color: #f5f7fa; }
    div[data-testid="stSidebarContent"] {
        background-color: #1a3a5c;
    }
    div[data-testid="stSidebarContent"] * {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────
@st.cache_data
def load_data():
    schools = gpd.read_file(
        G3/"gis/layers/schools_predicted.gpkg")
    states  = gpd.read_file(
        G3/"data/raw/boundaries/gadm41_NGA_1.shp")
    summary = pd.read_csv(
        G3/"outputs/reports/state_education_summary.csv")
    return schools, states, summary

with st.spinner("Loading data ..."):
    schools, states, summary = load_data()

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.title("🏫 Nigeria Education")
st.sidebar.markdown("**ANN + GIS Framework**")
st.sidebar.divider()

page = st.sidebar.radio("Navigation", [
    "📊 Overview",
    "🗺️ School Explorer",
    "📈 Performance Analysis",
    "⚠️ Dropout Risk",
    "♿ Accessibility",
    "🤖 ANN Model",
    "🗾 Maps Gallery",
])

st.sidebar.divider()
state_filter = st.sidebar.selectbox(
    "Filter by State",
    ["All States"] +
    sorted(schools["state"].dropna().unique().tolist())
)
type_filter = st.sidebar.multiselect(
    "School Type",
    ["Primary","Junior Secondary",
     "Senior Secondary","Tertiary"],
    default=["Primary","Junior Secondary",
             "Senior Secondary","Tertiary"]
)

# Apply filters
s = schools.copy()
if state_filter != "All States":
    s = s[s["state"]==state_filter]
if type_filter:
    s = s[s["school_type"].isin(type_filter)]

PERF_COLORS = {
    "High":   "#1a9850",
    "Medium": "#fdae61",
    "Low":    "#d73027"
}
DROP_COLORS = {
    "Low Risk":    "#1a9850",
    "Medium Risk": "#fdae61",
    "High Risk":   "#d73027"
}

# ══════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("🏫 Nigeria National Education Framework")
    st.markdown(
        "**ANN + GIS Integrated Analysis**  ·  "
        "OSM 2026  ·  WorldPop 2020  ·  TensorFlow  ·  GeoPandas"
    )

    c1,c2,c3,c4,c5 = st.columns(5)
    for col, val, label, unit in [
        (c1, f"{len(s):,}",
         "Schools", ""),
        (c2, f"{s['enrolment'].sum():,.0f}",
         "Total Enrolment", "students"),
        (c3, f"{s['pred_score'].mean():.1f}",
         "Avg Performance", "/100"),
        (c4, f"{(s['pred_dropout']=='High Risk').mean()*100:.1f}%",
         "High Dropout Risk", "schools"),
        (c5, f"{s['facilities_score'].mean():.1f}",
         "Avg Facilities", "/10"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}<br>
            <small style="color:#999">{unit}</small>
            </div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Schools by Type")
        tc = s["school_type"].value_counts().reset_index()
        fig = px.bar(tc, x="school_type", y="count",
                     color="school_type",
                     color_discrete_map={
                         "Primary":"#2166ac",
                         "Junior Secondary":"#74add1",
                         "Senior Secondary":"#f4a441",
                         "Tertiary":"#d73027"},
                     labels={"school_type":"Type",
                             "count":"Schools"},
                     template="simple_white")
        fig.update_layout(showlegend=False,
                           plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Performance Distribution")
        pc = s["pred_perf_cls"].value_counts().reset_index()
        fig2 = px.pie(pc, names="pred_perf_cls",
                      values="count",
                      color="pred_perf_cls",
                      color_discrete_map=PERF_COLORS,
                      hole=0.42,
                      template="simple_white")
        fig2.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Top & Bottom 10 States by Performance")
    col1, col2 = st.columns(2)
    with col1:
        top10 = summary.nlargest(10,"avg_score")
        fig_t = px.bar(top10, x="avg_score", y="state",
                       orientation="h",
                       color="avg_score",
                       color_continuous_scale="Greens",
                       template="simple_white",
                       labels={"avg_score":"Score","state":"State"})
        fig_t.update_layout(
            plot_bgcolor="white",
            yaxis={"categoryorder":"total ascending"},
            title="Top 10 — Highest Performing")
        st.plotly_chart(fig_t, use_container_width=True)
    with col2:
        bot10 = summary.nsmallest(10,"avg_score")
        fig_b = px.bar(bot10, x="avg_score", y="state",
                       orientation="h",
                       color="avg_score",
                       color_continuous_scale="Reds_r",
                       template="simple_white",
                       labels={"avg_score":"Score","state":"State"})
        fig_b.update_layout(
            plot_bgcolor="white",
            yaxis={"categoryorder":"total ascending"},
            title="Bottom 10 — Lowest Performing")
        st.plotly_chart(fig_b, use_container_width=True)

# ══════════════════════════════════════════════════════════
elif page == "🗺️ School Explorer":
    st.title("🗺️ School Location Explorer")

    col1, col2 = st.columns([2,1])
    with col1:
        color_by = st.selectbox(
            "Colour schools by:",
            ["Performance Class","Dropout Risk",
             "School Type","Ownership"])

        sample = s.sample(min(6000,len(s)),
                           random_state=42)

        if color_by == "Performance Class":
            color_col = "pred_perf_cls"
            cmap = PERF_COLORS
        elif color_by == "Dropout Risk":
            color_col = "pred_dropout"
            cmap = DROP_COLORS
        elif color_by == "School Type":
            color_col = "school_type"
            cmap = {"Primary":"#2166ac",
                    "Junior Secondary":"#74add1",
                    "Senior Secondary":"#f4a441",
                    "Tertiary":"#d73027"}
        else:
            color_col = "ownership"
            cmap = {"Public":"#1a9850",
                    "Private":"#d73027",
                    "Mission/Faith-based":"#4575b4"}

        fig_map = px.scatter_geo(
            sample,
            lat="lat", lon="lon",
            color=color_col,
            color_discrete_map=cmap,
            hover_name="name",
            hover_data={"state":True,
                        "school_type":True,
                        "pred_score":True,
                        "enrolment":True,
                        "lat":False,"lon":False},
            template="simple_white",
        )
        fig_map.update_geos(
            scope="africa",
            center={"lat":9.5,"lon":8.5},
            projection_scale=5.5,
            showland=True,
            landcolor="#f5f5f5",
            showocean=True,
            oceancolor="#dce8f0",
            showcountries=True,
            countrycolor="#aaaaaa",
            showframe=False,
        )
        fig_map.update_traces(marker_size=4)
        fig_map.update_layout(height=500,
                               margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_map, use_container_width=True)

    with col2:
        st.subheader("Statistics")
        st.dataframe(
            s.groupby("school_type").agg(
                Count=("school_id","count"),
                Avg_Score=("pred_score","mean"),
                Avg_Enrolment=("enrolment","mean"),
            ).round(1),
            use_container_width=True
        )
        st.subheader("Ownership Split")
        oc = s["ownership"].value_counts()
        fig_o = px.pie(values=oc.values,
                       names=oc.index,
                       color=oc.index,
                       color_discrete_map={
                           "Public":"#1a9850",
                           "Private":"#d73027",
                           "Mission/Faith-based":"#4575b4"},
                       hole=0.4,
                       template="simple_white")
        st.plotly_chart(fig_o, use_container_width=True)

# ══════════════════════════════════════════════════════════
elif page == "📈 Performance Analysis":
    st.title("📈 Education Performance Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Performance Score by State")
        fig_s = px.bar(
            summary.sort_values("avg_score"),
            x="avg_score", y="state",
            orientation="h",
            color="avg_score",
            color_continuous_scale=[
                "#d73027","#fc8d59","#fee090",
                "#91cf60","#1a9850"],
            range_color=[35,75],
            template="simple_white",
            labels={"avg_score":"Avg Score",
                    "state":"State"})
        fig_s.update_layout(
            plot_bgcolor="white", height=700,
            yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig_s, use_container_width=True)

    with col2:
        st.subheader("Score vs Facilities")
        sample2 = s.sample(min(3000,len(s)),
                            random_state=1)
        fig_sc = px.scatter(
            sample2,
            x="facilities_score",
            y="pred_score",
            color="pred_perf_cls",
            color_discrete_map=PERF_COLORS,
            opacity=0.55,
            template="simple_white",
            labels={"facilities_score":"Facilities (0-10)",
                    "pred_score":"Performance Score",
                    "pred_perf_cls":"Class"},
            hover_data=["state","school_type"])
        fig_sc.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_sc, use_container_width=True)

        st.subheader("Performance by School Type")
        fig_box = px.box(
            s, x="school_type", y="pred_score",
            color="school_type",
            color_discrete_map={
                "Primary":"#2166ac",
                "Junior Secondary":"#74add1",
                "Senior Secondary":"#f4a441",
                "Tertiary":"#d73027"},
            template="simple_white",
            labels={"pred_score":"Performance Score",
                    "school_type":"School Type"})
        fig_box.update_layout(plot_bgcolor="white",
                               showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

# ══════════════════════════════════════════════════════════
elif page == "⚠️ Dropout Risk":
    st.title("⚠️ Dropout Risk Analysis")

    c1,c2,c3 = st.columns(3)
    for col,risk,clr in [
        (c1,"Low Risk","#1a9850"),
        (c2,"Medium Risk","#fdae61"),
        (c3,"High Risk","#d73027"),
    ]:
        n   = (s["pred_dropout"]==risk).sum()
        pct = n/len(s)*100
        col.markdown(f"""
        <div class="metric-card"
             style="border-left: 4px solid {clr}">
            <div class="metric-value"
                 style="color:{clr}">{pct:.1f}%</div>
            <div class="metric-label">{risk}<br>
            <small>{n:,} schools</small></div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("High Dropout Risk by State")
        dr = s.groupby("state").apply(
            lambda x: round(
                (x["pred_dropout"]=="High Risk")\
                .mean()*100, 1)
        ).reset_index()
        dr.columns = ["state","pct_high_risk"]
        fig_dr = px.bar(
            dr.sort_values("pct_high_risk",
                           ascending=False).head(20),
            x="state", y="pct_high_risk",
            color="pct_high_risk",
            color_continuous_scale="Reds",
            template="simple_white",
            labels={"pct_high_risk":"% High Risk",
                    "state":"State"})
        fig_dr.update_layout(
            plot_bgcolor="white",
            xaxis_tickangle=-45)
        st.plotly_chart(fig_dr, use_container_width=True)

    with col2:
        st.subheader("Dropout Risk vs Poverty Index")
        sample3 = s.sample(min(3000,len(s)),
                            random_state=2)
        fig_pov = px.scatter(
            sample3,
            x="poverty_index",
            y="dropout_rate",
            color="pred_dropout",
            color_discrete_map=DROP_COLORS,
            opacity=0.5,
            template="simple_white",
            labels={
                "poverty_index":"Poverty Index (0-1)",
                "dropout_rate":"Dropout Rate (%)",
                "pred_dropout":"Risk Class"})
        fig_pov.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_pov, use_container_width=True)

    st.subheader("Hourly Dropout Rate Distribution")
    fig_hist = px.histogram(
        s, x="dropout_rate",
        color="pred_dropout",
        color_discrete_map=DROP_COLORS,
        nbins=40,
        template="simple_white",
        barmode="overlay",
        opacity=0.7,
        labels={"dropout_rate":"Dropout Rate (%)"})
    fig_hist.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig_hist, use_container_width=True)

# ══════════════════════════════════════════════════════════
elif page == "♿ Accessibility":
    st.title("♿ School Accessibility Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Population Density vs Performance")
        sample4 = s.sample(min(3000,len(s)),
                            random_state=3)
        fig_pd = px.scatter(
            sample4,
            x="socioeconomic_idx",
            y="pred_score",
            color="pred_perf_cls",
            color_discrete_map=PERF_COLORS,
            opacity=0.55,
            template="simple_white",
            labels={
                "socioeconomic_idx":"Socioeconomic Index",
                "pred_score":"Performance Score",
                "pred_perf_cls":"Performance"})
        fig_pd.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_pd, use_container_width=True)

    with col2:
        st.subheader("Facilities Score by State")
        fac = s.groupby("state")["facilities_score"]\
               .mean().sort_values(ascending=False)\
               .reset_index()
        fig_fac = px.bar(
            fac,
            x="facilities_score", y="state",
            orientation="h",
            color="facilities_score",
            color_continuous_scale="Blues",
            template="simple_white",
            labels={
                "facilities_score":"Facilities Score",
                "state":"State"})
        fig_fac.update_layout(
            plot_bgcolor="white", height=600,
            yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig_fac, use_container_width=True)

    st.subheader("Distance to Town vs Performance")
    fig_dist = px.scatter(
        s.sample(min(2000,len(s)), random_state=4),
        x="dist_to_town_km",
        y="pred_score",
        color="pred_perf_cls",
        color_discrete_map=PERF_COLORS,
        opacity=0.5,
        template="simple_white",
        labels={
            "dist_to_town_km":
                "Distance to Nearest Town (km)",
            "pred_score":"Performance Score",
            "pred_perf_cls":"Performance Class"})
    fig_dist.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig_dist, use_container_width=True)

# ══════════════════════════════════════════════════════════
elif page == "🤖 ANN Model":
    st.title("🤖 ANN Model Performance")

    c1,c2,c3 = st.columns(3)
    c1.markdown("""
    <div class="metric-card">
        <div class="metric-value">0.3844</div>
        <div class="metric-label">R² Score<br>
        <small>Performance Score</small></div>
    </div>""", unsafe_allow_html=True)
    c2.markdown("""
    <div class="metric-card">
        <div class="metric-value">52%</div>
        <div class="metric-label">Accuracy<br>
        <small>Performance Class</small></div>
    </div>""", unsafe_allow_html=True)
    c3.markdown("""
    <div class="metric-card">
        <div class="metric-value">76%</div>
        <div class="metric-label">Accuracy<br>
        <small>Dropout Risk</small></div>
    </div>""", unsafe_allow_html=True)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Feature Correlations with Performance")
        features = pd.DataFrame({
            "Feature": [
                "urban_weight","literacy_rate",
                "facilities_score","socioeconomic_idx",
                "internet_access","electricity",
                "toilets_per_100","teacher_student_ratio",
                "poverty_index","dist_to_town_km",
            ],
            "Correlation": [
                0.61, 0.58, 0.54, 0.52,
                0.44, 0.38, 0.31, 0.28,
               -0.55,-0.32,
            ]
        }).sort_values("Correlation")
        fig_feat = px.bar(
            features,
            x="Correlation", y="Feature",
            orientation="h",
            color="Correlation",
            color_continuous_scale="RdYlGn",
            range_color=[-0.7, 0.7],
            template="simple_white")
        fig_feat.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_feat,
                         use_container_width=True)

    with col2:
        st.subheader("ANN Architecture")
        st.code("""
Input (12 features)
  ├─ enrolment, teachers, teacher_student_ratio
  ├─ facilities_score, internet_access, electricity
  ├─ toilets_per_100, dist_to_town_km
  ├─ socioeconomic_idx, urban_weight
  ├─ poverty_index, literacy_rate

Dense(128) + BatchNorm + Dropout(0.25)
Dense(64)  + BatchNorm + Dropout(0.20)
Dense(32)  + BatchNorm

Output A: Dense(1)  → Performance Score (regression)
Output B: Dense(3)  → Performance Class (softmax)
Output C: Dense(3)  → Dropout Risk (softmax)

Optimizer : Adam (lr=0.001)
Loss A    : Huber
Loss B/C  : Sparse Categorical Cross-entropy
Epochs    : 80 (early stopping)
        """, language="text")

        st.subheader("Model Assessment")
        metrics_df = pd.DataFrame({
            "Model": ["Performance Regressor",
                      "Performance Classifier",
                      "Dropout Classifier"],
            "Metric": ["R² = 0.38, MAE = 9.89",
                       "Accuracy = 52%",
                       "Accuracy = 76%"],
            "Status": ["Acceptable","Moderate","Good"],
        })
        st.dataframe(metrics_df,
                      use_container_width=True)

# ══════════════════════════════════════════════════════════
elif page == "🗾 Maps Gallery":
    st.title("🗾 Thematic Maps Gallery")
    st.markdown("All maps exported at **300 DPI**  ·  PNG format")

    maps = [
        ("edu_map1_school_distribution",
         "Map 1 — School Distribution",
         "All 11,355 schools by type"),
        ("edu_map2_performance_choropleth",
         "Map 2 — Performance Choropleth",
         "State-level ANN performance scores"),
        ("edu_map3_ann_performance",
         "Map 3 — ANN Performance Class",
         "School-level High/Medium/Low classification"),
        ("edu_map4_dropout_risk",
         "Map 4 — Dropout Risk",
         "ANN-predicted dropout risk by school"),
        ("edu_map5_accessibility",
         "Map 5 — Accessibility",
         "Distance to nearest school analysis"),
        ("edu_map6_hotspots",
         "Map 6 — Hotspot Map",
         "Priority intervention zones"),
    ]

    for i in range(0, len(maps), 2):
        c1, c2 = st.columns(2)
        for col,(fname,title,cap) in zip(
                [c1,c2], maps[i:i+2]):
            img = G3/f"outputs/maps/{fname}.png"
            if img.exists():
                col.image(str(img),
                          caption=f"**{title}** — {cap}",
                          use_container_width=True)
            else:
                col.warning(f"Not found: {fname}.png")
