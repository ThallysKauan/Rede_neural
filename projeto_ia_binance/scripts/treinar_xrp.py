import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import os

# 1. Carregar Dados do XRP (O Ativo Sólido)
df = pd.read_csv('dados/xrp_com_sentimento.csv')
df = df.dropna() # Remover NaNs
df = df.sort_values('timestamp')

# 2. Normalização
scaler_p = MinMaxScaler()
scaler_s = MinMaxScaler()

df['price_scaled'] = scaler_p.fit_transform(df[['close']]).astype('float32')
df['sent_scaled'] = scaler_s.fit_transform(df[['Score_Sentimento']]).astype('float32')

# 3. Preparar Janelas (60h -> 24h futuro)
def create_sequences(df_subset, window=60):
    # Converter explicitamente para float32 aqui
    data = df_subset.values.astype('float32')
    X, y = [], []
    for i in range(len(data) - window - 24):
        X.append(data[i:(i+window), :]) # pega as 2 colunas
        y.append(data[i+window+23, 0])  # pega o preço scaled (coluna 0 do subset)
    return np.array(X, dtype='float32'), np.array(y, dtype='float32')

X, y = create_sequences(df[['price_scaled', 'sent_scaled']])

# 4. Modelo LSTM Especialista
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(60, 2)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

print("🚀 INICIANDO TREINAMENTO DO ESPECIALISTA XRP...")
model.fit(X, y, epochs=10, batch_size=32, verbose=1)

# 5. Salvar
model.save('modelos/xrp_modelo_hibrido.h5')
print("✅ MODELO XRP SALVO: modelos/xrp_modelo_hibrido.h5")
