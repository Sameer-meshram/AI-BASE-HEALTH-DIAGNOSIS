import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI Disease Diagnostic System", page_icon="🩺", layout="wide")

# ── Data & Model (cached) ───────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), "disease_dataset.csv"))

@st.cache_resource
def train_model(df):
    feat = [c for c in df.columns if c != "disease"]
    X, y_raw = df[feat].values, df["disease"].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, rf.predict(X_te))
    return rf, le, feat, acc

df = load_data()
rf, le, feat, acc = train_model(df)

# ── Sidebar ─────────────────────────────────────────
st.sidebar.markdown("## 🩺 AI Disease Diagnostic")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Prediction"])

# ═══════════════════ HOME ═══════════════════════════
if page == "🏠 Home":
    st.title("🩺 AI-Based Disease Diagnostic System")
    st.info("ML-powered clinical decision support — predicts diseases from patient symptoms.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples", len(df))
    c2.metric("Features", len(feat))
    c3.metric("Diseases", df["disease"].nunique())
    c4.metric("Model Accuracy", f"{acc:.1%}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
    with col2:
        st.subheader("Disease Distribution")
        fig, ax = plt.subplots()
        df["disease"].value_counts().plot.barh(ax=ax, color=sns.color_palette("mako", df["disease"].nunique()))
        ax.set_xlabel("Count")
        st.pyplot(fig)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)

# ═══════════════════ PREDICTION ═════════════════════
elif page == "🔍 Prediction":
    st.title("🔍 Disease Prediction")
    st.info("Select symptoms & age → Random Forest predicts the disease.")

    syms = [c for c in feat if c != "age"]
    cols = st.columns(4)
    inp = {s: int(cols[i % 4].checkbox(s.replace("_", " ").title(), key=s)) for i, s in enumerate(syms)}
    inp["age"] = st.slider("Patient Age", 1, 100, 30)

    if st.button("🔬 Predict", use_container_width=True):
        arr = np.array([[inp[c] for c in feat]])
        pred = rf.predict(arr)[0]
        proba = rf.predict_proba(arr)[0]
        disease = le.inverse_transform([pred])[0]

        st.success(f"**Predicted Disease: {disease}** — Confidence: {proba[pred]*100:.1f}%")

        prob_df = pd.DataFrame({"Disease": le.classes_, "Probability (%)": proba * 100}).sort_values("Probability (%)")
        fig, ax = plt.subplots()
        colors = ["#00d2ff" if d == disease else "#cccccc" for d in prob_df["Disease"]]
        ax.barh(prob_df["Disease"], prob_df["Probability (%)"], color=colors)
        ax.set_xlabel("Probability (%)")
        st.pyplot(fig)
