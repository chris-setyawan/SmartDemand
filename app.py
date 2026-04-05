import streamlit as st
import joblib
import json
import numpy as np

# ─────────────────────────────────────────────
# Page config — harus paling atas
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart-Demand",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

* { font-family: 'DM Sans', sans-serif; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 1200px !important; }

.stApp { background: #f0f2f6; }

.topbar {
    background: #0f1923;
    margin: 0 -2rem 2rem -2rem;
    padding: 1rem 2.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.topbar-logo { font-size: 1.4rem; font-weight: 700; color: #fff; letter-spacing: -0.5px; }
.topbar-logo span { color: #3b82f6; }
.topbar-badge {
    margin-left: auto;
    background: #1e3a5f;
    color: #93c5fd;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    font-family: 'DM Mono', monospace;
}

.model-banner {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.model-icon {
    width: 40px; height: 40px;
    background: #eff6ff;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
}
.model-title  { font-weight: 600; font-size: 0.95rem; color: #0f172a; }
.model-sub    { font-size: 0.8rem; color: #64748b; margin-top: 1px; }
.model-status {
    margin-left: auto;
    background: #f0fdf4;
    color: #16a34a;
    font-size: 0.78rem;
    font-weight: 500;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    border: 1px solid #bbf7d0;
}

.card-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 1rem;
}

.result-card {
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 140px; height: 140px;
    border-radius: 50%;
    opacity: 0.12;
    background: white;
}
.result-low      { background: linear-gradient(135deg, #0ea5e9, #0284c7); }
.result-moderate { background: linear-gradient(135deg, #f59e0b, #d97706); }
.result-high     { background: linear-gradient(135deg, #10b981, #059669); }
.result-veryhigh { background: linear-gradient(135deg, #8b5cf6, #7c3aed); }

.result-label  { font-size: 0.78rem; font-weight: 500; color: rgba(255,255,255,0.8); text-transform: uppercase; letter-spacing: 0.1em; }
.result-level  { font-size: 2.1rem; font-weight: 700; color: #fff; margin: 0.2rem 0 0.6rem 0; line-height: 1.1; }
.result-units  { font-size: 0.88rem; color: rgba(255,255,255,0.75); }

.result-score-label { position: absolute; top: 1.8rem; right: 1.8rem; text-align: right; }
.result-score-title { font-size: 0.72rem; color: rgba(255,255,255,0.7); font-weight: 500; }
.result-score-num   { font-size: 2.8rem; font-weight: 700; color: #fff; line-height: 1; }
.result-score-denom { font-size: 0.78rem; color: rgba(255,255,255,0.6); }

.ready-card {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 3.5rem 2rem;
    text-align: center;
    margin-bottom: 1rem;
}
.ready-icon  { font-size: 2.5rem; margin-bottom: 1rem; }
.ready-title { font-size: 1.2rem; font-weight: 600; color: #0f172a; }
.ready-sub   { font-size: 0.88rem; color: #64748b; margin-top: 0.4rem; line-height: 1.5; }

.dist-card {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.dist-title { font-size: 0.82rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 1.2rem; }
.dist-row   { margin-bottom: 0.9rem; }
.dist-header { display: flex; justify-content: space-between; margin-bottom: 0.3rem; }
.dist-name  { font-size: 0.83rem; font-weight: 500; color: #374151; }
.dist-pct   { font-size: 0.83rem; font-weight: 600; color: #374151; }
.dist-track { height: 8px; background: #f1f5f9; border-radius: 999px; overflow: hidden; }
.dist-fill  { height: 100%; border-radius: 999px; }
.fill-low      { background: #0ea5e9; }
.fill-moderate { background: #f59e0b; }
.fill-high     { background: #10b981; }
.fill-veryhigh { background: #8b5cf6; }

.summary-card {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.5rem;
}
.summary-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; }
.summary-item { background: #f8fafc; border-radius: 8px; padding: 0.6rem 0.8rem; }
.summary-key  { font-size: 0.72rem; color: #94a3b8; font-weight: 500; }
.summary-val  { font-size: 0.88rem; color: #0f172a; font-weight: 600; margin-top: 1px; font-family: 'DM Mono', monospace; }

.stSlider > div > div > div { background: #3b82f6 !important; }
.stSelectbox label, .stSlider label, .stNumberInput label, .stRadio label {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #374151 !important;
}
div[data-testid="stButton"] button {
    background: #1d4ed8 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.65rem !important;
    width: 100% !important;
}
div[data-testid="stButton"] button:hover { background: #1e40af !important; }
.stRadio > div { gap: 0.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load assets
# ─────────────────────────────────────────────
@st.cache_resource
def load_assets():
    rf_model = joblib.load('models/random_forest_model.joblib')
    lr_model = joblib.load('models/linear_regression_model.joblib')
    enc      = joblib.load('models/label_encoder.joblib')
    with open('models/config.json') as f:
        config = json.load(f)
    return rf_model, lr_model, enc, config

rf_model, lr_model, label_enc, config = load_assets()

categories        = config['categories']
seasonality_index = {int(k): v for k, v in config['seasonality_index'].items()}
rf_metrics        = config['model_metrics']['random_forest']
lr_metrics        = config['model_metrics']['linear_regression']

MONTH_NAMES = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def classify_demand(units):
    if units <= 3:
        return "Low Demand", "result-low"
    elif units <= 15:
        return "Moderate Demand", "result-moderate"
    elif units <= 40:
        return "High Demand", "result-high"
    else:
        return "Very High Demand", "result-veryhigh"

def demand_score(units, cap=80):
    return min(100, int((units / cap) * 100))

def demand_distribution(units):
    if units <= 3:
        return {"Low": 75, "Moderate": 18, "High": 5, "Very High": 2}
    elif units <= 15:
        pct_mod = int(40 + (units / 15) * 35)
        pct_low = max(5, 45 - pct_mod)
        pct_hi  = max(5, 100 - pct_mod - pct_low - 3)
        return {"Low": pct_low, "Moderate": pct_mod, "High": pct_hi, "Very High": 3}
    elif units <= 40:
        pct_hi  = int(45 + ((units - 15) / 25) * 35)
        pct_mod = max(8, 50 - pct_hi)
        pct_vhi = max(5, 100 - pct_hi - pct_mod - 5)
        return {"Low": 5, "Moderate": pct_mod, "High": pct_hi, "Very High": pct_vhi}
    else:
        pct_vhi = min(85, int(50 + (units - 40) / 40 * 30))
        pct_hi  = max(8, 90 - pct_vhi)
        return {"Low": 2, "Moderate": 5, "High": pct_hi, "Very High": pct_vhi}

# ─────────────────────────────────────────────
# Top bar
# ─────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div style="font-size:1.4rem">📦</div>
    <div class="topbar-logo">Smart<span>Demand</span></div>
    <div class="topbar-badge">Random Forest · Classical ML</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Model banner
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="model-banner">
    <div class="model-icon">🌲</div>
    <div>
        <div class="model-title">Random Forest Regressor</div>
        <div class="model-sub">Test R² {rf_metrics['test_r2']} &nbsp;·&nbsp; MAE {rf_metrics['test_mae']} &nbsp;·&nbsp; Brazilian E-Commerce Dataset by Olist</div>
    </div>
    <div class="model-status">● Model Ready</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="card-title">⚙️ &nbsp;Product Parameters</div>', unsafe_allow_html=True)

    category = st.selectbox("Product Category", options=categories)

    c1, c2 = st.columns(2)
    with c1:
        price = st.number_input("Price (BRL)", min_value=0.0, max_value=10000.0, value=100.0, step=10.0)
    with c2:
        freight = st.number_input("Freight Cost (BRL)", min_value=0.0, max_value=500.0, value=20.0, step=5.0)

    rating   = st.slider("Product Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
    discount = st.slider("Discount Rate", min_value=0.0, max_value=0.5, value=0.0, step=0.01, format="%.2f")

    c3, c4 = st.columns(2)
    with c3:
        month = st.selectbox("Target Month", options=list(range(1,13)), format_func=lambda m: MONTH_NAMES[m-1])
    with c4:
        last_sales = st.number_input("Last Month Sales (units)", min_value=0, max_value=500, value=10, step=1)

    model_choice = st.radio("Model", options=["Random Forest (Main)", "Linear Regression (Baseline)"], horizontal=True)

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    predict_btn = st.button("Predict Sales Volume", use_container_width=True)

with right:
    if not predict_btn:
        st.markdown("""
        <div class="ready-card">
            <div class="ready-icon">📊</div>
            <div class="ready-title">Ready to Predict</div>
            <div class="ready-sub">Fill in the product parameters on the left<br>and click <strong>Predict Sales Volume</strong><br>to get your demand forecast.</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="summary-card">
            <div class="card-title">📋 &nbsp;Model Performance</div>
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="summary-key">Random Forest — Test R²</div>
                    <div class="summary-val">{rf_metrics['test_r2']}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-key">Linear Regression — Test R²</div>
                    <div class="summary-val">{lr_metrics['test_r2']}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-key">Random Forest — MAE</div>
                    <div class="summary-val">{rf_metrics['test_mae']}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-key">Linear Regression — MAE</div>
                    <div class="summary-val">{lr_metrics['test_mae']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Build input
        cat_encoded  = int(label_enc.transform([category])[0]) if category in label_enc.classes_ else 0
        seas_index   = seasonality_index.get(month, 0.5)
        input_vector = np.array([[price, discount, freight, rating, cat_encoded, seas_index, last_sales]])

        if "Random Forest" in model_choice:
            raw_pred   = rf_model.predict(input_vector)[0]
            model_name = "Random Forest"
            metrics    = rf_metrics
        else:
            raw_pred   = lr_model.predict(input_vector)[0]
            model_name = "Linear Regression"
            metrics    = lr_metrics

        units        = max(0, round(raw_pred))
        level, card_class = classify_demand(units)
        score        = demand_score(units)
        dist         = demand_distribution(units)

        # Result card
        st.markdown(f"""
        <div class="result-card {card_class}">
            <div class="result-score-label">
                <div class="result-score-title">Demand Score</div>
                <div class="result-score-num">{score}</div>
                <div class="result-score-denom">out of 100</div>
            </div>
            <div class="result-label">Predicted Demand Level</div>
            <div class="result-level">{level}</div>
            <div class="result-units">Estimated <strong style="color:white">{units} units</strong> sold this month &nbsp;·&nbsp; {model_name}</div>
        </div>
        """, unsafe_allow_html=True)

        # Demand distribution
        fill_map = {"Low":"fill-low","Moderate":"fill-moderate","High":"fill-high","Very High":"fill-veryhigh"}
        rows = ""
        for label_d, pct in dist.items():
            rows += f"""
            <div class="dist-row">
                <div class="dist-header">
                    <span class="dist-name">{label_d}</span>
                    <span class="dist-pct">{pct}%</span>
                </div>
                <div class="dist-track">
                    <div class="dist-fill {fill_map[label_d]}" style="width:{pct}%"></div>
                </div>
            </div>"""

        st.markdown(f"""
        <div class="dist-card">
            <div class="dist-title">📈 &nbsp;Demand Probability Distribution</div>
            {rows}
        </div>
        """, unsafe_allow_html=True)

        # Input summary
        st.markdown(f"""
        <div class="summary-card">
            <div class="card-title">🔍 &nbsp;Input Parameters</div>
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="summary-key">Category</div>
                    <div class="summary-val" style="font-family:'DM Sans',sans-serif;font-size:0.78rem">{category[:30]}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-key">Month</div>
                    <div class="summary-val">{MONTH_NAMES[month-1]}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-key">Price (BRL)</div>
                    <div class="summary-val">{price:.2f}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-key">Freight (BRL)</div>
                    <div class="summary-val">{freight:.2f}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-key">Rating</div>
                    <div class="summary-val">{rating:.1f} / 5.0</div>
                </div>
                <div class="summary-item">
                    <div class="summary-key">Discount Rate</div>
                    <div class="summary-val">{discount*100:.0f}%</div>
                </div>
                <div class="summary-item">
                    <div class="summary-key">Last Month Sales</div>
                    <div class="summary-val">{last_sales} units</div>
                </div>
                <div class="summary-item">
                    <div class="summary-key">Seasonality Index</div>
                    <div class="summary-val">{seas_index:.3f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
