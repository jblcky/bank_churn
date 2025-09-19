import streamlit as st
import joblib
import numpy as np
import random

# --- 1. Load Model & Scaler ---
# Note: Using relative paths like this assumes you run the Streamlit app
# from the root directory of your project. Consider using more robust paths
# for production, and perhaps versioning models without dates in the filename.
try:
    model = joblib.load("notebooks/log_reg_model_17-9-2025.pkl")
    scaler = joblib.load("notebooks/scaler_19-9-2025.pkl")
except FileNotFoundError:
    st.error("Model or scaler files not found. Please ensure the paths are correct.")
    st.stop()


# --- 2. App Configuration ---
st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Bank Churn Prediction")
st.markdown("This dashboard predicts whether a bank customer is likely to churn (leave the bank).")


# --- 3. Initialize Session State for Random Defaults ---
# This makes the app experience more dynamic on first load.
# The keys are checked to prevent re-randomizing on every interaction.
if "credit_score" not in st.session_state:
    st.session_state.credit_score = random.randint(300, 850)
if "tenure" not in st.session_state:
    st.session_state.tenure = random.randint(0, 15)
if "is_active" not in st.session_state:
    st.session_state.is_active = random.choice(["No", "Yes"])
if "satisfaction" not in st.session_state:
    st.session_state.satisfaction = random.randint(1, 5)
if "points" not in st.session_state:
    st.session_state.points = random.randint(0, 1000) # Reduced max for more realistic defaults
if "engagement" not in st.session_state:
    st.session_state.engagement = random.randint(1, 10)


# --- 4. Sidebar Inputs for Customer Features ---
st.sidebar.header("Customer Features")

# --- Demographics ---
credit_score = st.sidebar.number_input("Credit Score", 300, 850, st.session_state.credit_score)
age = st.sidebar.slider("Age", 18, 92, 35)
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
salary = st.sidebar.number_input("Estimated Salary", 0.0, 200000.0, 50000.0, step=1000.0)

# --- Banking History ---
tenure = st.sidebar.slider("Tenure (Years)", 0, 15, st.session_state.tenure)
balance = st.sidebar.number_input("Balance", 0.0, 250000.0, 50000.0, step=1000.0)
num_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
has_card = st.sidebar.selectbox("Has Credit Card?", ["No", "Yes"])
card_type = st.sidebar.selectbox("Card Type", ["Silver", "Gold", "Platinum", "Diamond"])

# --- Engagement & Satisfaction ---
is_active = st.sidebar.selectbox("Is Active Member?", ["No", "Yes"], index=["No", "Yes"].index(st.session_state.is_active))
satisfaction = st.sidebar.slider("Satisfaction Score (1-5)", 1, 5, st.session_state.satisfaction)
complain = st.sidebar.selectbox("Has a Complaint?", ["No", "Yes"])
points = st.sidebar.number_input("Points Earned", 0, 20000, st.session_state.points)
engagement = st.sidebar.slider("Engagement Score (1-10)", 1, 10, st.session_state.engagement)


# --- 5. Feature Engineering & Preprocessing ---
# This section must match the preprocessing steps used to train the model.

# Convert categorical inputs to numerical format
geo_germany = int(geography == "Germany")
geo_spain = int(geography == "Spain")
gender_female = int(gender == "Female") # Male = 0, Female = 1
has_card_bin = int(has_card == "Yes")
is_active_bin = int(is_active == "Yes")
complain_bin = int(complain == "Yes")

# Map ordinal features
card_type_map = {"Silver": 1, "Gold": 2, "Platinum": 3, "Diamond": 4}
card_type_num = card_type_map[card_type]

# Create tenure groups
if tenure <= 3:
    tenure_group_val = 1
elif 4 <= tenure <= 7:
    tenure_group_val = 2
else:
    tenure_group_val = 3

# Create the feature array in the correct order for the model
# Using max(..., 1) is a safe way to avoid division by zero errors
features_list = [
    credit_score, gender_female, age, tenure, balance,
    num_products, has_card_bin, is_active_bin, salary,
    complain_bin, satisfaction, points,
    balance / max(num_products, 1),  # Engineered feature 1
    balance / max(salary, 1),        # Engineered feature 2
    engagement, tenure_group_val, card_type_num,
    geo_germany, geo_spain
]
features = np.array([features_list])

# Scale numerical features using the pre-trained scaler
# Note: The indices must correspond to the columns the scaler was trained on.
scale_cols_indices = [0, 2, 3, 4, 5, 8, 10, 11, 12, 13, 14]
# Column names for reference:
# ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary',
#  'Satisfaction Score', 'Point Earned', 'BalancePerProduct', 'BalanceSalaryRatio', 'EngagementScore']

features_scaled = features.copy()
features_scaled[:, scale_cols_indices] = scaler.transform(features_scaled[:, scale_cols_indices])


# --- 6. Prediction and Output ---
if st.sidebar.button("Predict Churn", type="primary"):

    # Get the probability of churn (which is the probability of class '1')
    churn_probability = model.predict_proba(features_scaled)[:, 1][0]

    st.subheader("📊 Prediction Result")
    st.metric("Churn Probability", f"{churn_probability:.2%}")

    # Display result based on a 50% threshold
    if churn_probability >= 0.5:
        st.error("⚠️ This customer is **likely to churn**.", icon="🚨")
    else:
        st.success("✅ This customer is **not likely to churn**.", icon="👍")

    # Display a visual probability gauge
    st.write("### Probability Gauge")
    gauge_color = "#ff4d4d" if churn_probability >= 0.5 else "#4caf50"
    gauge_html = f"""
        <div style="background: #eee; border-radius: 10px; padding: 4px;">
            <div style="background: {gauge_color}; width: {churn_probability*100}%;
                        border-radius: 6px; height: 24px; text-align: center;
                        color: white; font-weight: bold; line-height: 24px;">
                {churn_probability*100:.1f}%
            </div>
        </div>
    """
    st.markdown(gauge_html, unsafe_allow_html=True)
