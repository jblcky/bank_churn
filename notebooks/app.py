import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =========================
# Load model & scaler
# =========================
model = joblib.load("notebooks/log_reg_model_17-9-2025.pkl")       # Logistic Regression
scaler = joblib.load("notebooks/scaler_19-9-2025.pkl")            # StandardScaler (if used)

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
# Sidebar Input
# =========================
st.sidebar.header("Customer Features")

credit_score = st.sidebar.number_input("Credit Score", 300, 850, 650)
age = st.sidebar.slider("Age", 18, 92, 35)
tenure = st.sidebar.slider("Tenure (Years)", 0, 15, 5)
balance = st.sidebar.number_input("Balance", 0.0, 250000.0, 50000.0, step=1000.0)
num_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
has_card = st.sidebar.selectbox("Has Credit Card?", [0, 1])
is_active = st.sidebar.selectbox("Active Member?", [0, 1])
salary = st.sidebar.number_input("Estimated Salary", 0.0, 200000.0, 50000.0, step=1000.0)

geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

complain = st.sidebar.selectbox("Complain?", [0, 1])
satisfaction = st.sidebar.slider("Satisfaction Score", 1, 5, 3)
points = st.sidebar.number_input("Points Earned", 0, 20000, 5000)
card_type = st.sidebar.selectbox("Card Type Rank", [1, 2, 3])  # 1=Silver,2=Gold,3=Platinum
engagement = st.sidebar.slider("Engagement Score", 0, 10, 5)

# =========================
# Encode Features
# =========================
# Geography One-hot
geo_germany = 1 if geography == "Germany" else 0
geo_spain   = 1 if geography == "Spain" else 0
# France is implicit baseline

# Gender binary
gender_bin = 1 if gender == "Male" else 0

# Feature vector
features = np.array([[
    credit_score, gender_bin, age, tenure, balance,
    num_products, has_card, is_active, salary,
    complain, satisfaction, points,
    balance / max(num_products,1),      # BalancePerProduct
    balance / max(salary,1),            # BalanceToSalary
    engagement, tenure, card_type,
    geo_germany, geo_spain
]])

# Scale if used in training
features_scaled = scaler.transform(features)

# =========================
# Predict
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

    # Fancy progress bar
    st.progress(float(prob))

    # Gauge-like visualization
    st.write("### Probability Gauge")
    st.write(
        f"""
        <div style="width:100%; background:#eee; border-radius:20px;">
          <div style="width:{prob*100}%; background:{'#ff4d4d' if prob>0.5 else '#4caf50'};
                      padding:10px; border-radius:20px; text-align:center; color:white;">
            {prob*100:.1f}%
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
