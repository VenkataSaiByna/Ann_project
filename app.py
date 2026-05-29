import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import tensorflow as tf
import pickle

# Load model and encoders
model = tf.keras.models.load_model('model.h5')

scaler = pickle.load(open("StandardScalar.pkl", 'rb'))
labelencoder = pickle.load(open("Labelencoder.pkl", 'rb'))
onehotencoder = pickle.load(open("Onehotencoder.pkl", 'rb'))

# Streamlit UI
st.title('Bank Customer Churn Prediction')

# Inputs
geography = st.selectbox(
    'Geography',
    onehotencoder.categories_[0]
)

gender = st.selectbox(
    'Gender',
    labelencoder.classes_
)

age = st.slider('Age', 18, 92)

balance = st.number_input('Balance')

credit_score = st.number_input('Credit Score')

estimated_salary = st.number_input('Estimated Salary')

tenure = st.slider('Tenure', 0, 10)

num_of_products = st.slider('Number of Products', 1, 4)

has_cr_card = st.selectbox('Has Credit Card', [0, 1])

is_active_member = st.selectbox('Is Active Member', [0, 1])

# Encode Gender
gender_encoded = labelencoder.transform([gender])[0]

# Create input dataframe
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Geography Encoding
geo_df = pd.DataFrame({
    'Geography': [geography]
})

geo_encoded = onehotencoder.transform(geo_df).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehotencoder.get_feature_names_out(['Geography'])
)

# Combine all features
final_input = pd.concat(
    [input_data.reset_index(drop=True),
     geo_encoded_df.reset_index(drop=True)],
    axis=1
)

# Scale input
final_input_scaled = scaler.transform(final_input)

# Prediction
prediction = model.predict(final_input_scaled)

prediction_proba = prediction[0][0]

# Output
st.subheader(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.error('Customer is likely to churn')
else:
    st.success('Customer is not likely to churn')