# Bank Churn Prediction App

## Step 1: Project Overview
This project is a **CPU-friendly Bank Churn Prediction model** using **Logistic Regression**.
It predicts whether a customer is likely to churn based on features such as credit score, age, balance, tenure, engagement, and more.

**Key Features:**
- Handles **imbalanced datasets** using class weighting.
- Includes **feature engineering**, e.g., `BalancePerProduct`, `BalanceToSalary`, `EngagementScore`.
- Supports **real-time prediction** via a **Streamlit front end**.
- Minimalist and responsive dashboard with a **probability gauge**.

## Step 2: Setup & Installation

### 1. Clone the Repository
```bash
git clone <your-github-repo-url>
cd <your-repo-folder>

### 2. Create a Virtual Environment (Recommended)
# Using Python venv
python -m venv venv
# Activate the environment
# Windows (cmd)
venv\Scripts\activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1
# Ubuntu / macOS
source venv/bin/activate


## Step 3: Prepare Data & Preprocessing

### 1. Dataset
- The project uses a **bank customer dataset** with columns such as:
CreditScore, Gender, Age, Tenure, Balance, NumOfProducts,
HasCrCard, IsActiveMember, EstimatedSalary, Exited, Complain,
Satisfaction Score, Point Earned, BalancePerProduct, BalanceToSalary,
EngagementScore, TenureGroup, CardTypeRank, Geography_Germany, Geography_Spain

- Target variable: `Exited` (1 = churn, 0 = stayed)

### 2. Feature Engineering
- **BalancePerProduct** = Balance ÷ NumOfProducts
- **BalanceToSalary** = Balance ÷ EstimatedSalary
- **EngagementScore** = custom score derived from customer activity
- **TenureGroup** = categorize tenure into Short, Mid, Long
- **CardTypeRank** = ordinal encoding of card type

### 3. Encoding & Scaling
- Binary columns (Gender, HasCrCard, IsActiveMember, Complain, Geography_*) are **already encoded** as 0/1.
- Continuous numeric columns (CreditScore, Age, Tenure, Balance, NumOfProducts, EstimatedSalary, Satisfaction Score, Point Earned, BalancePerProduct, BalanceToSalary, EngagementScore) are **scaled using StandardScaler**.
- Target variable `Exited` is kept as 0/1.

### 4. Train-Test Split
- Split dataset into **80% training** and **20% testing**.
- Stratified split to maintain class distribution.
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=42, stratify=y
)


## Step 4: Train & Evaluate Model

### 1. Train Logistic Regression Model
- Use **class weighting** to handle imbalanced classes.
- Scale numeric features before training.
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train_scaled, y_train)


### 2. Make Predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]  # probability of churn

### 3. Evaluate Performance
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print(confusion_matrix(y_test, y_pred))


## Step 5: Deploy on Streamlit Cloud

### 1. Prepare Files
- Ensure your repository contains:
  - `app.py` (Streamlit app)
  - `logreg_model.pkl` (trained Logistic Regression model)
  - `scaler.pkl` (StandardScaler used during training)
  - `requirements.txt` (all dependencies)

### 2. Push to GitHub
```bash
git add .
git commit -m "Add Streamlit app and model"
git push origin main

### 3. Deploy on Streamlit Cloud
Go to Streamlit Cloud and sign in.
Click “New app” → Connect your GitHub repository.
Select branch (main) and file path (app.py) → Click Deploy.

### 4. Test App
Open the provided URL.
Use sidebar inputs to enter customer details.
Click Predict Churn to see probability and prediction.
