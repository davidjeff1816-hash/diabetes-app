import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------
# APP TITLE
# -----------------------------
st.set_page_config(page_title="Diabetes Prediction App")
st.title("🩺 Diabetes Prediction using Logistic Regression")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("diabetes (1).xlsx")
    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# DATA PREPARATION
# -----------------------------
# Change column names here IF your dataset differs
X = df.drop("Outcome", axis=1)   # Features
y = df["Outcome"]                # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# MODEL TRAINING
# -----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

st.success(f"Model Accuracy: {accuracy:.2f}")

# -----------------------------
# USER INPUT
# -----------------------------
st.subheader("Enter Patient Details")

pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose Level", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin Level", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0, step=1)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict"):
    input_data = [[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, dpf, age
    ]]
    
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ The patient is likely DIABETIC")
    else:
        st.success("✅ The patient is NOT diabetic")
