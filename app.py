import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


## streamlit app
st.title('Customer Churn Predictor')
st.write("""
This application predicts the likelihood of customer churn based on various input features.
Fill in the details in the sidebar to get the prediction.
""")

# Sidebar for user input
st.sidebar.title("User Input")
geography = st.sidebar.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.sidebar.selectbox('Gender', label_encoder_gender.classes_)
age = st.sidebar.slider('Age', 18, 92)
balance = st.sidebar.number_input('Balance', min_value=0.0)
credit_score = st.sidebar.number_input('Credit Score', min_value=0)
estimated_salary = st.sidebar.number_input('Estimated Salary', min_value=0.0)
tenure = st.sidebar.slider('Tenure', 0, 10)
num_of_products = st.sidebar.slider('Number of Products', 1, 4)
has_cr_card = st.sidebar.selectbox('Has Credit Card', [0, 1])
is_active_member = st.sidebar.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Display input data
st.subheader("Input Data")
st.write(input_data)

# Prediction button
if st.sidebar.button('Predict'):
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    st.subheader('Prediction Result')
    st.write(f'Churn Probability: {prediction_proba:.2f}')
    if prediction_proba > 0.5:
        st.write('The customer is likely to churn.')
    else:
        st.write('The customer is not likely to churn.')


