import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import os, time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="AI Disease Diagnostic System", page_icon="🩺", layout="wide")

# ── Dark / Light Mode ────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

with st.sidebar:
    st.markdown("## 🩺 AI Disease Diagnostic")
    st.caption("ML-Powered Clinical Decision Support")
    st.divider()
    st.session_state.dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)

# Theme variables
if st.session_state.dark_mode:
    _bg, _card, _accent = "#0e1117", "#1a1d23", "#00d2ff"
    _text, _sub, _border = "#e0e0e0", "#8899aa", "#2a2d35"
    _tmpl = "plotly_dark"
else:
    _bg, _card, _accent = "#f5f7fa", "#ffffff", "#0077b6"
    _text, _sub, _border = "#1a1a2e", "#555555", "#dde1e7"
    _tmpl = "plotly_white"

st.markdown(f"""
<style>
.stApp {{ background-color: {_bg}; }}
section[data-testid="stSidebar"] {{ background-color: {_card}; border-right: 1px solid {_border}; }}
h1,h2,h3,h4 {{ color: {_text} !important; }}
p,span,label,.stMarkdown {{ color: {_text}; }}
div[data-testid="stMetric"] {{
    background: linear-gradient(135deg, {_card}, {_border});
    border: 1px solid {_border}; border-left: 4px solid {_accent};
    border-radius: 12px; padding: 16px 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.12);
}}
div[data-testid="stMetricValue"] {{ color: {_accent} !important; font-weight: 700; }}
button[data-baseweb="tab"] {{ font-weight: 600; font-size: 1.05rem; }}
button[data-baseweb="tab"][aria-selected="true"] {{ border-bottom-color: {_accent} !important; color: {_accent} !important; }}
.stButton > button {{
    background: linear-gradient(135deg, {_accent}, #0090d9);
    color: #fff !important; border: none; border-radius: 10px;
    font-weight: 700; font-size: 1.05rem; padding: 0.6rem 1.2rem;
    transition: transform 0.15s, box-shadow 0.15s;
}}
.stButton > button:hover {{ transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,210,255,0.35); }}
.stMultiSelect [data-baseweb="tag"] {{ background-color: {_accent} !important; border-radius: 8px; }}
.risk-card {{
    border-radius: 14px; padding: 24px; text-align: center;
    font-size: 1.1rem; font-weight: 600;
    box-shadow: 0 4px 16px rgba(0,0,0,0.15); margin: 10px 0;
    animation: fadeIn 0.8s ease-out;
}}
@keyframes fadeIn {{ from {{ opacity:0; transform:translateY(20px); }} to {{ opacity:1; transform:translateY(0); }} }}
.risk-low  {{ background:linear-gradient(135deg,#064e3b,#065f46); color:#6ee7b7; border:1px solid #10b981; }}
.risk-med  {{ background:linear-gradient(135deg,#78350f,#92400e); color:#fcd34d; border:1px solid #f59e0b; }}
.risk-high {{ background:linear-gradient(135deg,#7f1d1d,#991b1b); color:#fca5a5; border:1px solid #ef4444; }}
.pred-card {{
    background: linear-gradient(135deg, {_card}, {_border});
    border: 2px solid {_accent}; border-radius: 16px; padding: 28px;
    text-align: center; box-shadow: 0 4px 24px rgba(0,210,255,0.15);
    animation: fadeIn 0.6s ease-out;
}}
.pred-card h2 {{ margin:0; color:{_accent} !important; font-size:2rem; }}
.pred-card p  {{ color:{_sub}; margin:4px 0 0 0; font-size:1.1rem; }}
</style>
""", unsafe_allow_html=True)

# ── Data & Model (cached) ────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), "disease_dataset.csv"))

@st.cache_resource
def train_model(df_hash):
    df = load_data()
    feat = [c for c in df.columns if c != "disease"]
    X, y_raw = df[feat].values, df["disease"].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    cm = confusion_matrix(y_te, y_pred)
    cr = classification_report(y_te, y_pred, target_names=le.classes_, output_dict=True)
    return rf, le, feat, acc, cm, cr, rf.feature_importances_, le.classes_

df = load_data()
rf, le, feat, acc, cm, cr, importances, class_names = train_model(len(df))

with st.sidebar:
    st.divider()
    st.metric("Model Accuracy", f"{acc:.1%}")
    st.caption(f"**{len(df):,}** samples • **{len(feat)}** features • **{len(class_names)}** diseases")
    st.divider()
    st.caption("© 2026 AI Disease Diagnostic System")

# Helper for common plotly layout
def _layout(fig, h=380, **kw):
    fig.update_layout(height=h, margin=dict(l=0, r=0, t=10, b=0),
                      coloraxis_showscale=False, template=_tmpl, **kw)
    return fig

# ═══════════════════ TABS ════════════════════════════
tab_dash, tab_pred, tab_perf = st.tabs(["🏠 Dashboard", "🔍 Prediction", "📊 Model Performance"])

# ── DASHBOARD ──
with tab_dash:
    st.markdown("# 🩺 AI-Based Disease Diagnostic System")
    st.info("ML-powered clinical decision support — predicts diseases from patient symptoms using Random Forest.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📋 Samples", f"{len(df):,}")
    c2.metric("🧬 Features", len(feat))
    c3.metric("🦠 Diseases", len(class_names))
    c4.metric("🎯 Accuracy", f"{acc:.1%}")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📄 Dataset Preview")
        st.dataframe(df.head(12), use_container_width=True, height=400)
    with col2:
        st.subheader("📊 Disease Distribution")
        vc = df["disease"].value_counts().reset_index()
        vc.columns = ["Disease", "Count"]
        fig = _layout(px.bar(vc, y="Disease", x="Count", orientation="h", color="Count",
                             color_continuous_scale=["#0e4d92", _accent]),
                      yaxis_title="", xaxis_title="Samples")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("⭐ Feature Importance")
    imp_df = pd.DataFrame({"Feature": feat, "Importance": importances}).sort_values("Importance")
    fig_imp = _layout(px.bar(imp_df, y="Feature", x="Importance", orientation="h",
                             color="Importance", color_continuous_scale=["#2d3436", _accent]),
                      h=450, yaxis_title="", xaxis_title="Importance Score")
    st.plotly_chart(fig_imp, use_container_width=True)

    with st.expander("📈 Summary Statistics"):
        st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)

# ── PREDICTION ──
with tab_pred:
    st.markdown("# 🔍 Disease Prediction")
    st.info("Select patient symptoms & age → AI predicts the most likely disease.")
    st.markdown("---")

    syms = [c for c in feat if c != "age"]
    sym_labels = [s.replace("_", " ").title() for s in syms]
    label_to_col = dict(zip(sym_labels, syms))

    col_input, col_age = st.columns([3, 1])
    with col_input:
        selected = st.multiselect("🩻 Select Symptoms", sym_labels, placeholder="Choose symptoms...")
    with col_age:
        age = st.slider("🎂 Patient Age", 1, 100, 30)

    if st.button("🔬  Run Diagnosis", use_container_width=True):
        if not selected:
            st.warning("⚠️ Please select at least one symptom.")
        else:
            with st.spinner("🔬 Analyzing symptoms..."):
                time.sleep(1.2)
                sel_cols = {label_to_col[l] for l in selected}
                inp = {s: int(s in sel_cols) for s in syms}
                inp["age"] = age
                arr = np.array([[inp[c] for c in feat]])
                pred = rf.predict(arr)[0]
                proba = rf.predict_proba(arr)[0]
                disease = le.inverse_transform([pred])[0]
                confidence = proba[pred] * 100

            st.balloons()

            # Result card
            st.markdown(f"""
            <div class="pred-card">
                <h2>🦠 {disease}</h2>
                <p>Predicted with <strong>{confidence:.1f}%</strong> confidence</p>
            </div>
            """, unsafe_allow_html=True)

            # Risk level
            if confidence > 75:
                rc, ri, rl = "risk-high", "🔴", "HIGH RISK"
            elif confidence > 50:
                rc, ri, rl = "risk-med", "🟡", "MODERATE RISK"
            else:
                rc, ri, rl = "risk-low", "🟢", "LOW RISK"

            r1, r2 = st.columns(2)
            with r1:
                st.markdown(f"""
                <div class="risk-card {rc}">
                    <div style="font-size:2.5rem;">{ri}</div>
                    <div style="font-size:1.4rem;margin-top:8px;">{rl}</div>
                    <div style="font-size:0.9rem;margin-top:4px;">Confidence: {confidence:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            with r2:
                st.markdown("#### 📊 Confidence Gauge")
                st.progress(min(confidence / 100, 1.0))
                st.markdown(f"**Diagnosis:** {disease}  \n**Risk Level:** {rl}  \n**Symptoms:** {', '.join(selected)}")

            st.markdown("---")

            col_prob, col_fi = st.columns(2)
            with col_prob:
                st.subheader("🏥 Probability by Disease")
                prob_df = pd.DataFrame({"Disease": le.classes_, "Prob": proba * 100}).sort_values("Prob")
                colors = [_accent if d == disease else "#4a4a5a" for d in prob_df["Disease"]]
                fig_p = go.Figure(go.Bar(y=prob_df["Disease"], x=prob_df["Prob"], orientation="h",
                                        marker_color=colors,
                                        text=[f"{v:.1f}%" for v in prob_df["Prob"]], textposition="outside"))
                _layout(fig_p, h=350, xaxis_title="Probability (%)", yaxis_title="")
                st.plotly_chart(fig_p, use_container_width=True)

            with col_fi:
                st.subheader("⭐ Why This Prediction?")
                fi_df = pd.DataFrame({"Feature": feat, "Importance": importances, "Value": arr[0]}).sort_values("Importance")
                fi_df["Status"] = fi_df["Value"].apply(lambda v: "Selected" if v == 1 else ("Age" if v > 1 else "—"))
                fig_fi = _layout(px.bar(fi_df, y="Feature", x="Importance", orientation="h", color="Status",
                                       color_discrete_map={"Selected": _accent, "—": "#3a3a4a", "Age": "#f59e0b"}),
                                 h=350, yaxis_title="", xaxis_title="Importance", legend_title="Status")
                st.plotly_chart(fig_fi, use_container_width=True)

# ── MODEL PERFORMANCE ──
with tab_perf:
    st.markdown("# 📊 Model Performance")
    st.info("Detailed evaluation metrics for the Random Forest classifier.")
    st.markdown("---")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🎯 Accuracy", f"{acc:.1%}")
    m2.metric("🌲 Estimators", 200)
    m3.metric("📊 Test Size", "20%")
    m4.metric("🧬 Features", len(feat))
    st.markdown("---")

    col_cm, col_cr = st.columns(2)
    with col_cm:
        st.subheader("🔥 Confusion Matrix")
        fig_cm = ff.create_annotated_heatmap(
            z=cm.tolist(), x=list(class_names), y=list(class_names),
            annotation_text=[[str(v) for v in row] for row in cm],
            colorscale=[[0, "#0e1117"], [0.5, "#0077b6"], [1, _accent]], showscale=True)
        fig_cm.update_layout(template=_tmpl, height=420, margin=dict(l=0,r=0,t=30,b=0),
                             xaxis_title="Predicted", yaxis_title="Actual", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_cr:
        st.subheader("📋 Classification Report")
        cr_df = pd.DataFrame(cr).T
        rows = [r for r in list(class_names) + ["macro avg", "weighted avg"] if r in cr_df.index]
        cr_disp = cr_df.loc[rows, ["precision", "recall", "f1-score", "support"]]
        cr_disp.columns = ["Precision", "Recall", "F1-Score", "Support"]
        color_f1 = lambda v: f"color: {'#6ee7b7' if v >= 0.95 else '#fcd34d' if v >= 0.85 else '#fca5a5'}" if isinstance(v, (int, float)) else ""
        st.dataframe(cr_disp.style.format({"Precision":"{:.3f}","Recall":"{:.3f}","F1-Score":"{:.3f}","Support":"{:.0f}"})
                     .map(color_f1, subset=["F1-Score"]), use_container_width=True, height=420)

    st.markdown("---")
    st.subheader("📈 Per-Class F1-Score")
    f1_df = pd.DataFrame({"Disease": list(class_names),
                          "F1": [cr[c]["f1-score"] for c in class_names]}).sort_values("F1")
    fig_f1 = _layout(px.bar(f1_df, y="Disease", x="F1", orientation="h", color="F1",
                            color_continuous_scale=["#ef4444","#f59e0b","#10b981"], range_x=[0,1]),
                     h=350, yaxis_title="", xaxis_title="F1-Score")
    st.plotly_chart(fig_f1, use_container_width=True)
