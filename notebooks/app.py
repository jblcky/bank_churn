import streamlit as st
import pandas as pd
import joblib
import numpy as np
import random
import plotly.graph_objects as go
import plotly.express as px
import shap

# =========================
# Load model & scaler
# =========================
model = joblib.load("notebooks/log_reg_model_17-9-2025.pkl")
scaler = joblib.load("notebooks/scaler_19-9-2025.pkl")

# =========================
# App Config
# =========================
st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Bank Churn Prediction")
st.markdown("Minimalist dashboard to predict if a customer will churn.")

# =========================
# Initialize random defaults using session_state
# =========================
if "credit_score" not in st.session_state:
    st.session_state.credit_score = random.randint(300, 850)
if "tenure" not in st.session_state:
    st.session_state.tenure = random.randint(0, 15)
if "is_active" not in st.session_state:
    st.session_state.is_active = random.choice(["No", "Yes"])
if "satisfaction" not in st.session_state:
    st.session_state.satisfaction = random.randint(1, 5)
if "points" not in st.session_state:
    st.session_state.points = random.randint(0, 20000)
if "engagement" not in st.session_state:
    st.session_state.engagement = random.randint(0, 10)

# =========================
# Sidebar Input
# =========================
st.sidebar.header("Customer Features")

credit_score = st.sidebar.number_input(
    "Credit Score", 300, 850, st.session_state.credit_score
)
age = st.sidebar.slider("Age", 18, 92, 35)
tenure = st.sidebar.slider("Tenure (Years)", 0, 15, st.session_state.tenure)
balance = st.sidebar.number_input("Balance", 0.0, 250000.0, 50000.0, step=1000.0)
num_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
has_card = st.sidebar.selectbox("Has Credit Card?", ["No", "Yes"])
is_active = st.sidebar.selectbox(
    "Active Member?", ["No", "Yes"], index=["No", "Yes"].index(st.session_state.is_active)
)
salary = st.sidebar.number_input("Estimated Salary", 0.0, 200000.0, 50000.0, step=1000.0)

geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

complain = st.sidebar.selectbox("Complain?", ["No", "Yes"])
satisfaction = st.sidebar.slider(
    "Satisfaction Score", 1, 5, st.session_state.satisfaction
)
points = st.sidebar.number_input(
    "Points Earned", 0, 20000, st.session_state.points
)
card_type = st.sidebar.selectbox("Card Type Rank", ["Silver", "Gold", "Platinum"])
engagement = st.sidebar.slider(
    "Engagement Score", 0, 10, st.session_state.engagement
)

# =========================
# Encode Features
# =========================
geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0

gender_bin = 1 if gender == "Male" else 0
has_card_bin = 1 if has_card == "Yes" else 0
is_active_bin = 1 if is_active == "Yes" else 0
complain_bin = 1 if complain == "Yes" else 0
card_type_map = {"Silver": 1, "Gold": 2, "Platinum": 3}
card_type_num = card_type_map[card_type]

features = np.array([[
    credit_score, gender_bin, age, tenure, balance,
    num_products, has_card_bin, is_active_bin, salary,
    complain_bin, satisfaction, points,
    balance / max(num_products,1),
    balance / max(salary,1),
    engagement, tenure, card_type_num,
    geo_germany, geo_spain
]])

scale_cols = [0, 2, 3, 4, 5, 8, 10, 11, 12, 13, 14]
features_scaled = features.copy()
features_scaled[:, scale_cols] = scaler.transform(features_scaled[:, scale_cols])

# =========================
# Predict and Explain
# =========================
if st.sidebar.button("Predict Churn"):
    prob = model.predict_proba(features_scaled)[:,1][0]
    pred = int(prob >= 0.5)

    st.subheader("📊 Prediction Result")
    st.metric("Churn Probability", f"{prob:.2%}")

    if pred == 1:
        st.error("⚠️ Customer is **likely to churn**.")
    else:
        st.success("✅ Customer is **not likely to churn**.")

    st.progress(float(prob))

    # =========================
    # Circular gauge using Plotly
    # =========================
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob*100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#ff4d4d" if prob>0.5 else "#4caf50"},
            'steps': [
                {'range': [0, 50], 'color': '#4caf50'},
                {'range': [50, 100], 'color': '#ff4d4d'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # SHAP Interactive Explanation using Plotly
    # =========================
    st.subheader("🔍 Feature Contribution (SHAP)")

    explainer = shap.LinearExplainer(model, features_scaled, feature_perturbation="correlation")
    shap_values = explainer.shap_values(features_scaled)

    feature_names = [
        "Credit Score", "Gender", "Age", "Tenure", "Balance",
        "Num Products", "Has Card", "Active Member", "Salary",
        "Complain", "Satisfaction", "Points",
        "BalancePerProduct", "BalanceToSalary", "Engagement",
        "Tenure Duplicate", "Card Type", "Geo Germany", "Geo Spain"
    ]

    shap_vals_single = shap_values[0]  # single prediction
    shap_contrib = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_vals_single
    }).sort_values(by="SHAP Value", key=abs, ascending=False)

    fig_shap = px.bar(
        shap_contrib,
        x="SHAP Value",
        y="Feature",
        orientation="h",
        text="SHAP Value",
        color="SHAP Value",
        color_continuous_scale=["#4caf50", "#ff4d4d"],
        title="Feature Contribution to Churn Probability"
    )
    fig_shap.update_layout(yaxis=dict(autorange="reversed"), height=600)
    st.plotly_chart(fig_shap, use_container_width=True)
