
import streamlit as st
import pandas as pd
import joblib
import numpy as np

#carregar dataset
import pandas as pd
df = pd.read_csv('data_file.csv')

#remover ids
df.drop(['FileName','md5Hash'], axis=1, inplace=True)

#separar X e y
X = df.iloc[:,:-1]
y = df['Benign']

# Carregar o modelo treinado e o scaler
try:
    model = joblib.load('best_rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Modelo ou scaler não encontrados. Por favor, execute o treinamento no notebook antes de rodar o app.")
    st.stop()

st.title('Detector de Ransomware')
st.write('Insira os valores das features para prever se um arquivo é benigno ou ransomware.')

# Exibir os nomes das features (colunas) para que o usuário saiba o que inserir
st.sidebar.subheader("Features")
feature_names = X.columns.tolist() # Usar as colunas do DataFrame original X
input_values = {}
for feature in feature_names:
    input_values[feature] = st.sidebar.number_input(f'Valor para {feature}', value=0.0)

# Criar um DataFrame com os inputs do usuário
input_df = pd.DataFrame([input_values])

# Pré-processar os dados de entrada (escalar)
input_scaled = scaler.transform(input_df)

# Fazer a previsão
if st.button('Prever'):
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    st.subheader('Resultado da Previsão')
    if prediction[0] == 1:
        st.success(f'O arquivo é **BENIGNO** (Probabilidade: {prediction_proba[0][1]:.4f})')
    else:
        st.warning(f'O arquivo é **RANSOMWARE** (Probabilidade: {prediction_proba[0][0]:.4f})')

    st.subheader('Valores de Entrada')
    st.write(input_df)

