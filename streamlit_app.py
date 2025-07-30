
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load dataset
import pandas as pd
df = pd.read_csv('data_file.csv')

# Remove ids
df.drop(['FileName', 'md5Hash'], axis=1, inplace=True)

# Separate X and y
X = df.iloc[:, :-1]
y = df['Benign']

# Load the trained model and the scaler
try:
    model = joblib.load('best_rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Model or scaler not found. Please run the training in the notebook before running the app.")
    st.stop()

st.title('Ransomware Detector')
st.write('Enter the feature values to predict whether a file is benign or ransomware.')

# Display the feature names (columns) so the user knows what to input
st.sidebar.subheader("Features")
feature_names = X.columns.tolist()  # Use the columns from the original DataFrame X
input_values = {}
for feature in feature_names:
    input_values[feature] = st.sidebar.number_input(f'Value for {feature}', value=0.0)

# Create a DataFrame with the user's inputs
input_df = pd.DataFrame([input_values])

# Pre-process the input data (scale)
input_scaled = scaler.transform(input_df)

# Make the prediction
if st.button('Predict'):
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    st.subheader('Prediction Result')
    if prediction[0] == 1:
        st.success(f'The file is **BENIGN** (Probability: {prediction_proba[0][1]:.4f})')
    else:
        st.warning(f'The file is **RANSOMWARE** (Probability: {prediction_proba[0][0]:.4f})')

    st.subheader('Input Values')
    st.write(input_df)
