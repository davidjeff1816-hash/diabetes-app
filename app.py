import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Diabetes Prediction App")
st.title("🩺 Diabetes Prediction App")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_excel("diabetes (1).xlsx")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# SPLIT FEATURES & TARGET
# -----------------------------
target_col = df.columns[-1]   # LAST column is target
X = df.iloc[:, :-1]           # All except last
y = df[target_col]

# Encode target if it's text
if y.dtype == "object":
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# SCALING
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# MODEL
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
st.success(f"Model Accuracy: {accuracy:.2f}")

# -----------------------------
# USER INPUT
# -----------------------------
st.subheader("Enter Patient Details")

inputs = []
for col in X.columns:
    val = st.number_input(col, value=0.0)
    inputs.append(val)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict"):
    input_df = pd.DataFrame([inputs], columns=X.columns)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("⚠️ Diabetic")
    else:
        st.success("✅ Not Diabetic")
