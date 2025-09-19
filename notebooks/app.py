"""
A minimalist Streamlit dashboard to predict bank customer churn.

This application loads a pre-trained logistic regression model and a scaler
to predict the probability of a customer churning based on user-provided features.
"""

import streamlit as st
import joblib
import numpy as np
import random
from pathlib import Path

# --- PATHS AND CONSTANTS ---
# NOTE: The original file paths included future dates. They have been simplified.
# Ensure your model and scaler files are named accordingly and placed in a 'models' directory.
MODEL_DIR = Path("notebooks")
MODEL_PATH = MODEL_DIR / "log_reg_model_17-9-2025.pkl"
SCALER_PATH = MODEL_DIR / "scaler_19-9-2025.pkl"

# Define columns that need to be scaled based on the model training
# This is less error-prone than manually listing indices.
# NOTE: Update this list if the feature engineering or model changes.
COLS_TO_SCALE = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary',
    'Satisfaction Score', 'Point Earned', 'Balance_per_Product', 'Balance_to_Salary_Ratio',
    'Engagement_Score'
]


# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="💳",
    layout="centered"
)


# --- LOAD MODEL AND SCALER ---
@st.cache_resource
def load_assets(model_path, scaler_path):
    """
    Loads and caches the machine learning model and scaler from disk.
    Using @st.cache_resource ensures these assets are loaded only once.
    """
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError:
        st.error(
            f"Model or scaler not found. Please make sure '{model_path}' and "
            f"'{scaler_path}' exist."
        )
        return None, None

model, scaler = load_assets(MODEL_PATH, SCALER_PATH)


# --- HELPER FUNCTIONS ---
def initialize_session_state():
    """
    Initializes session state with random default values for a more dynamic demo.
    This runs only once per session.
    """
    defaults = {
        "credit_score": random.randint(300, 850),
        "tenure": random.randint(0, 15),
        "is_active": random.choice(["No", "Yes"]),
        "satisfaction": random.randint(1, 5),
        "points": random.randint(0, 20000),
        "engagement": random.randint(0, 10),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_sidebar_inputs():
    """
    Creates all sidebar input widgets and returns their values in a dictionary.
    """
    st.sidebar.header("Customer Features")
    inputs = {
        "credit_score": st.sidebar.number_input("Credit Score", 300, 850, st.session_state.credit_score),
        "age": st.sidebar.slider("Age", 18, 92, 35),
        "tenure": st.sidebar.slider("Tenure (Years)", 0, 15, st.session_state.tenure),
        "balance": st.sidebar.number_input("Balance", 0.0, 250000.0, 50000.0, step=1000.0),
        "num_products": st.sidebar.selectbox("Number of Products", [1, 2, 3, 4]),
        "has_card": st.sidebar.selectbox("Has Credit Card?", ["No", "Yes"]),
        "is_active": st.sidebar.selectbox("Active Member?", ["No", "Yes"], index=["No", "Yes"].index(st.session_state.is_active)),
        "salary": st.sidebar.number_input("Estimated Salary", 0.0, 200000.0, 50000.0, step=1000.0),
        "geography": st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"]),
        "gender": st.sidebar.selectbox("Gender", ["Male", "Female"]),
        "complain": st.sidebar.selectbox("Has Complained?", ["No", "Yes"]),
        "satisfaction": st.sidebar.slider("Satisfaction Score", 1, 5, st.session_state.satisfaction),
        "points": st.sidebar.number_input("Points Earned", 0, 20000, st.session_state.points),
        "card_type": st.sidebar.selectbox("Card Type", ["Silver", "Gold", "Platinum", "Diamond"]),
        "engagement": st.sidebar.slider("Engagement Score", 0, 10, st.session_state.engagement),
    }
    return inputs


def preprocess_features(inputs, scaler_obj):
    """
    Preprocesses raw user inputs into a scaled numpy array for the model.
    """
    # Feature Engineering
    inputs['balance_per_product'] = inputs['balance'] / max(inputs['num_products'], 1)
    inputs['balance_to_salary_ratio'] = inputs['balance'] / max(inputs['salary'], 1)

    # --- New Feature: Tenure Group ---
    tenure = inputs['tenure']
    if tenure <= 3:
        tenure_group_val = 1  # Short: 0-3 years
    elif 4 <= tenure <= 7:
        tenure_group_val = 2  # Mid: 4-7 years
    else:
        tenure_group_val = 3  # Long: 8+ years

    # Encoding
    card_type_map = {"Silver": 1, "Gold": 2, "Platinum": 3, "Diamond": 4}

    # Create a feature dictionary that matches the model's training order
    feature_dict = {
        'CreditScore': inputs['credit_score'],
        'Gender': 1 if inputs['gender'] == "Male" else 0,
        'Age': inputs['age'],
        'Tenure': inputs['tenure'],
        'TenureGroup': tenure_group_val,
        'Balance': inputs['balance'],
        'NumOfProducts': inputs['num_products'],
        'HasCrCard': 1 if inputs['has_card'] == "Yes" else 0,
        'IsActiveMember': 1 if inputs['is_active'] == "Yes" else 0,
        'EstimatedSalary': inputs['salary'],
        'Complain': 1 if inputs['complain'] == "Yes" else 0,
        'Satisfaction Score': inputs['satisfaction'],
        'Point Earned': inputs['points'],
        'Balance_per_Product': inputs['balance_per_product'],
        'Balance_to_Salary_Ratio': inputs['balance_to_salary_ratio'],
        'Engagement_Score': inputs['engagement'],
        'Card Type': card_type_map[inputs['card_type']],
        'Geography_Germany': 1 if inputs['geography'] == "Germany" else 0,
        'Geography_Spain': 1 if inputs['geography'] == "Spain" else 0,
    }

    # Convert to DataFrame-like structure for scaling
    feature_names = list(feature_dict.keys())
    feature_values = np.array([list(feature_dict.values())])

    # Identify indices of columns to scale
    scale_indices = [i for i, name in enumerate(feature_names) if name in COLS_TO_SCALE]

    # Scale the appropriate features
    scaled_features = feature_values.copy()
    scaled_features[:, scale_indices] = scaler_obj.transform(feature_values[:, scale_indices])

    return scaled_features


def display_prediction(probability):
    """
    Displays the prediction result and a probability gauge.
    """
    is_churn = probability >= 0.5
    st.subheader("📊 Prediction Result")
    st.metric("Churn Probability", f"{probability:.2%}")

    if is_churn:
        st.error("⚠️ Customer is **likely to churn**.")
    else:
        st.success("✅ Customer is **not likely to churn**.")

    # --- Simple HTML/CSS gauge for visualization ---
    st.write("### Probability Gauge")
    gauge_color = "#ff4d4d" if is_churn else "#4caf50" # Red for churn, Green for stay
    gauge_html = f"""
        <div style="background-color: #eee; border-radius: 20px; padding: 4px;">
          <div style="background-color: {gauge_color}; width: {probability*100}%;
                      height: 25px; border-radius: 16px; display: flex;
                      align-items: center; justify-content: center;
                      color: white; font-weight: bold;">
            {probability*100:.1f}%
          </div>
        </div>
    """
    st.markdown(gauge_html, unsafe_allow_html=True)


# --- MAIN APP LOGIC ---
def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("💳 Bank Churn Prediction")
    st.markdown("A minimalist dashboard to predict if a bank customer will churn based on their features.")

    initialize_session_state()
    user_inputs = get_sidebar_inputs()

    if st.sidebar.button("Predict Churn"):
        if model and scaler:
            # Preprocess, predict, and display
            processed_features = preprocess_features(user_inputs, scaler)
            churn_probability = model.predict_proba(processed_features)[:, 1][0]
            display_prediction(churn_probability)
        else:
            st.warning("Cannot predict because the model assets are not loaded.")

if __name__ == "__main__":
    main()
