import streamlit as st
import joblib
import numpy as np
import random

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
# Predict and Display
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

    # =========================
    # Simple HTML/CSS gauge
    # =========================
    st.write("### Probability Gauge")
    gauge_color = "#ff4d4d" if prob > 0.5 else "#4caf50"
    st.write(f"""
        <div style="width:100%; background:#eee; border-radius:20px; height:30px;">
          <div style="width:{prob*100}%; background:{gauge_color};
                      height:30px; border-radius:20px; text-align:center; color:white;">
            {prob*100:.1f}%
          </div>
        </div>
    """, unsafe_allow_html=True)
