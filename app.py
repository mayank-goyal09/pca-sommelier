import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------------------------------------------
# 1. Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="CellarScope PCA Studio",
    page_icon="🍷",
    layout="wide"
)

# ---------------------------------------------------
# 2. Dark Grey / Dark Red / Dark Pink Glass Theme
# ---------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Background: dark grey with deep red glow on RIGHT */
.stApp {
    background:
        radial-gradient(circle at 90% 0%, rgba(185, 28, 28, 0.65) 0%, rgba(15, 23, 42, 0.0) 55%),
        radial-gradient(circle at 15% 100%, rgba(190, 24, 93, 0.55) 0%, rgba(15, 23, 42, 0.0) 55%),
        linear-gradient(135deg, #020617 0%, #020617 40%, #030712 100%);
    color: #e5e5e5;
}

.block-container {
    padding-top: 1.5rem;
    max-width: 1400px;
}

/* Hide default header */
[data-testid="stHeader"] {background: transparent;}
header {background: transparent;}

/* GLASS CARD */
.glass-card {
    background: rgba(15, 23, 42, 0.78);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(148, 163, 184, 0.35);
    border-radius: 20px;
    padding: 22px 24px;
    box-shadow: 0 18px 45px rgba(0, 0, 0, 0.85);
    margin-bottom: 22px;
}

/* BAR TITLE with red accent on RIGHT */
.bar-title-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background:
        linear-gradient(90deg, #020617 0%, #020617 55%, #7f1d1d 100%);
    padding: 18px 24px;
    border-radius: 18px;
    box-shadow: 0 16px 40px rgba(127, 29, 29, 0.7);
    border: 1px solid rgba(248, 113, 113, 0.45);
    margin-bottom: 26px;
}
.bar-left {
    display: flex;
    align-items: center;
    gap: 14px;
}
.bar-icon {
    font-size: 2.1rem;
    padding: 8px 10px;
    border-radius: 14px;
    background: radial-gradient(circle at 30% 20%, #f472b6 0%, #be123c 45%, #020617 90%);
    box-shadow: 0 0 30px rgba(248, 113, 113, 0.9);
}
.bar-text-main {
    font-size: 1.7rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    color: #fee2e2;
}
.bar-text-sub {
    font-size: 0.88rem;
    color: #fecaca;
    margin-top: 3px;
}
.bar-pill {
    font-size: 0.78rem;
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(15, 23, 42, 0.92);
    border: 1px solid rgba(248, 113, 113, 0.6);
    color: #fecaca;
    font-weight: 600;
    text-align: center;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: radial-gradient(circle at 10% 0%, rgba(248, 113, 113, 0.25) 0%, rgba(15, 23, 42, 1) 40%);
    border-right: 1px solid rgba(148, 163, 184, 0.3);
}
[data-testid="stSidebar"] * {
    color: #e5e7eb;
}
.sidebar-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #fb7185;
}
.sidebar-text {
    font-size: 0.9rem;
    color: #9ca3af;
}

/* UPLOADER */
[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.9);
    border: 1px dashed rgba(248, 113, 113, 0.7);
    border-radius: 14px;
    padding: 16px;
}

/* BUTTONS */
.stButton > button {
    background: linear-gradient(135deg, #b91c1c 0%, #be123c 50%, #db2777 100%);
    color: #fef2f2;
    border: none;
    border-radius: 999px;
    padding: 0.6rem 1.7rem;
    font-weight: 600;
    font-size: 0.92rem;
    box-shadow: 0 10px 30px rgba(248, 113, 113, 0.6);
    transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    background: linear-gradient(135deg, #ef4444 0%, #e11d48 50%, #db2777 100%);
}
.stDownloadButton > button {
    background: rgba(15, 23, 42, 0.9);
    border-radius: 999px;
    border: 1px solid rgba(248, 113, 113, 0.7);
    color: #fecaca;
    font-weight: 500;
}
.stDownloadButton > button:hover {
    background: rgba(248, 113, 113, 0.1);
}

/* DATAFRAME */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    border: 1px solid rgba(148, 163, 184, 0.4);
}

/* HEADINGS & TEXT */
h1, h2, h3 { color: #ffe4e6; }
p, span, div { color: #e5e5e5; }
.small-muted { color: #9ca3af; font-size: 0.85rem; }

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# 3. Sidebar
# ---------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">🍷 CellarScope Controls</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="sidebar-text">
<strong>How to use:</strong><br>
1️⃣ Download the sample wine CSV or use your own.<br>
2️⃣ Upload a CSV with numeric chemical features.<br>
3️⃣ Choose features and number of components.<br>
4️⃣ Read variance, 2D projection, and loadings.
</div>
""", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("PCA · StandardScaler · Plotly")

# ---------------------------------------------------
# 4. Bar-style Title
# ---------------------------------------------------
st.markdown("""
<div class="bar-title-container">
  <div class="bar-left">
    <div class="bar-icon">🍷</div>
    <div>
      <div class="bar-text-main">CELLARSCOPE PCA STUDIO</div>
      <div class="bar-text-sub">Wine Chemistry Dimensionality Reduction & Profile Mapping</div>
    </div>
  </div>
  <div class="bar-pill">
    Principal Component Analysis · Multivariate Projection
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# 5. Sample & Upload (Glass Card)
# ---------------------------------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

col1, col2 = st.columns([1.1, 1.3])

with col1:
    st.subheader("📂 Sample Dataset")
    st.markdown("""
Use this synthetic wine-like dataset to explore the PCA workflow quickly.  
Includes acidity, sulphates, alcohol, density, and quality.
""")

    # Generate sample data in-memory
    np.random.seed(42)
    dummy_data = {
        'fixed_acidity': np.random.normal(7, 1, 180),
        'volatile_acidity': np.random.normal(0.3, 0.1, 180),
        'citric_acid': np.random.normal(0.3, 0.1, 180),
        'residual_sugar': np.random.normal(5, 3, 180),
        'chlorides': np.random.normal(0.05, 0.02, 180),
        'free_sulfur_dioxide': np.random.normal(30, 10, 180),
        'total_sulfur_dioxide': np.random.normal(100, 30, 180),
        'density': np.random.normal(0.99, 0.005, 180),
        'pH': np.random.normal(3.2, 0.15, 180),
        'sulphates': np.random.normal(0.5, 0.1, 180),
        'alcohol': np.random.normal(10.5, 1.1, 180),
        'quality': np.random.randint(3, 9, 180),
        'type': np.random.choice(['Red', 'White'], 180)
    }
    sample_df = pd.DataFrame(dummy_data)
    csv_bytes = sample_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Sample Wine CSV",
        data=csv_bytes,
        file_name="cellarscope_sample_wine.csv",
        mime="text/csv",
    )
    st.caption(f"Rows: {sample_df.shape[0]} · Columns: {sample_df.shape[1]}")

with col2:
    st.subheader("📤 Upload Your Wine CSV")
    uploaded_file = st.file_uploader(
        "Upload a CSV with numeric wine features (acidity, alcohol, etc.).",
        type=["csv"],
        label_visibility="collapsed"
    )

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# 6. Load Data
# ---------------------------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Using uploaded file: {uploaded_file.name} ({len(df):,} rows)")
else:
    df = sample_df.copy()
    st.info("ℹ️ No file uploaded. Using the built-in synthetic wine dataset.")

# Preview
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("👀 Data Preview")
st.dataframe(df.head(), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# 7. PCA Settings
# ---------------------------------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("🔧 PCA Configuration")

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
default_features = [c for c in numeric_cols if c not in ["quality"]]
if not default_features:
    default_features = numeric_cols

col_conf1, col_conf2 = st.columns([3, 1])

with col_conf1:
    selected_features = st.multiselect(
        "Select numeric features to include in PCA:",
        options=numeric_cols,
        default=default_features,
        help="Choose at least two numeric columns representing wine chemistry."
    )

with col_conf2:
    max_components = min(len(selected_features), 10) if len(selected_features) >= 2 else 2
    n_components = st.slider(
        "Number of components",
        min_value=2,
        max_value=max_components,
        value=2,
        step=1
    )

if len(selected_features) < 2:
    st.error("⚠️ Please select at least 2 numeric features for PCA.")
    st.stop()

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# 8. Run PCA
# ---------------------------------------------------
X = df[selected_features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

explained_var = pca.explained_variance_ratio_
cum_var = np.cumsum(explained_var)

# ---------------------------------------------------
# 9. Variance + 2D Plot
# ---------------------------------------------------
col_v1, col_v2 = st.columns([1, 1.4])

# 9.1 Scree / Variance
with col_v1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📈 Variance Explained")

    scree_df = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(n_components)],
        "Variance": explained_var,
        "Cumulative": cum_var
    })

    fig_scree = go.Figure()
    fig_scree.add_trace(go.Bar(
        x=scree_df["Component"],
        y=scree_df["Variance"],
        name="Individual",
        marker_color="#fb7185"
    ))
    fig_scree.add_trace(go.Scatter(
        x=scree_df["Component"],
        y=scree_df["Cumulative"],
        name="Cumulative",
        mode="lines+markers",
        marker_color="#f97373"
    ))

    fig_scree.update_layout(
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.85)",
        font=dict(color="#e5e7eb"),
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    fig_scree.update_xaxes(gridcolor="rgba(148,163,184,0.2)")
    fig_scree.update_yaxes(gridcolor="rgba(148,163,184,0.2)")
    st.plotly_chart(fig_scree, use_container_width=True)

    st.markdown(
        f"<p style='text-align:center;' class='small-muted'>Total variance explained by first {n_components} PCs: "
        f"<span style='color:#fb7185;font-weight:600;'>{cum_var[-1]*100:.1f}%</span></p>",
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

# 9.2 2D PCA scatter
with col_v2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📊 PCA Projection (PC1 vs PC2)")

    pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])

    color_col = None
    for cand in ["type", "quality", "target", "class"]:
        if cand in df.columns:
            color_col = cand
            pca_df[cand] = df.loc[X.index, cand].values
            break

    fig_pca = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color=color_col,
        color_discrete_sequence=px.colors.qualitative.Bold,
        title=None,
        opacity=0.85
    )
    fig_pca.update_traces(
        marker=dict(
            size=9,
            line=dict(width=1, color="rgba(15,23,42,0.9)")
        )
    )
    fig_pca.update_layout(
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.85)",
        font=dict(color="#e5e7eb"),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    fig_pca.update_xaxes(gridcolor="rgba(148,163,184,0.2)")
    fig_pca.update_yaxes(gridcolor="rgba(148,163,184,0.2)")
    st.plotly_chart(fig_pca, use_container_width=True)

    st.markdown("""
<p class="small-muted">
PC1 and PC2 are linear combinations of your original features. Points that lie close together share similar
chemical profiles in the reduced PCA space.
</p>
""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# 10. Component Loadings
# ---------------------------------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("🧠 PCA Component Loadings")

loadings = pd.DataFrame(
    pca.components_.T,
    index=selected_features,
    columns=[f"PC{i+1}" for i in range(n_components)]
)

fig_heat = px.imshow(
    loadings,
    color_continuous_scale="RdBu_r",
    aspect="auto",
    labels=dict(x="Principal Component", y="Feature", color="Loading"),
)
fig_heat.update_layout(
    height=420,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.85)",
    font=dict(color="#e5e7eb"),
    margin=dict(l=20, r=20, t=20, b=20),
)
st.plotly_chart(fig_heat, use_container_width=True)

st.markdown(
    "<p class='small-muted'>Higher absolute values indicate stronger influence of a feature on that principal component.</p>",
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# 11. Footer (based on your reference)
# ---------------------------------------------------
st.markdown("""
<hr style="margin-top:2.5rem; border-color:#ec4899; border-width:1px;">
<div style="text-align:center; padding:1.2rem 0; color:#e5e7eb; font-size:0.85rem;">

  <div style="margin-bottom:6px; font-weight:600; color:#f9a8d4;">
    © 2025 CellarScope PCA Studio · Wine PCA Analytics · Built by
    <span style="color:#f973c9; font-weight:800;">Mayank Goyal</span>
  </div>

  <div style="margin-bottom:4px;">
    <a href="https://www.linkedin.com/in/mayank-goyal-4b8756363" target="_blank"
       style="color:#93c5fd; text-decoration:none; margin-right:18px; font-weight:600;">
        🔗 LinkedIn
    </a>
    <a href="https://github.com/mayank-goyal09" target="_blank"
       style="color:#a5b4fc; text-decoration:none; font-weight:600;">
        💻 GitHub
    </a>
  </div>

  <div style="margin-top:6px; font-size:0.78rem; color:#9ca3af;">
    🍷 PCA · Standardized Features · Interactive Variance & Loadings Visualization
  </div>

</div>
""", unsafe_allow_html=True)
