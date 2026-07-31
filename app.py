import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")
st.write("Enter the customer details below to predict the probability of churn.")

# -------------------- Load Model & Preprocessors --------------------
model = tf.keras.models.load_model("model.keras")

with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# -------------------- User Inputs --------------------
geography = st.selectbox(
    "Geography",
    onehot_encoder_geo.categories_[0]
)

gender = st.selectbox(
    "Gender",
    label_encoder_gender.classes_
)

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

age = st.slider(
    "Age",
    min_value=18,
    max_value=92,
    value=35
)

tenure = st.slider(
    "Tenure",
    min_value=0,
    max_value=10,
    value=5
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    max_value=300000.0,
    value=60000.0
)

num_of_products = st.slider(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=1
)

has_cr_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

is_active_member = st.selectbox(
    "Is Active Member",
    [0, 1]
)

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    max_value=250000.0,
    value=50000.0
)

# -------------------- Prediction --------------------
if st.button("Predict Churn"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [label_encoder_gender.transform([gender])[0]],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary]
    })

    # One-Hot Encode Geography
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
    )

    # Combine DataFrames
    input_data = pd.concat(
        [input_data.reset_index(drop=True), geo_encoded_df],
        axis=1
    )

    # Ensure correct feature order
    input_data = input_data[scaler.feature_names_in_]

    # Scale input
    input_data_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data_scaled, verbose=0)
    prediction_proba = float(prediction[0][0])

    # Display Results
    st.subheader("Prediction Result")

    st.metric(
        label="Churn Probability",
        value=f"{prediction_proba:.2%}"
    )

    st.progress(prediction_proba)

    if prediction_proba > 0.5:
        st.error("⚠️ The customer is likely to churn.")
    else:
        st.success("✅ The customer is not likely to churn.")