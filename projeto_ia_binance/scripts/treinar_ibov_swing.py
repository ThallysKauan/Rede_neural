import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
import os

def treinar_ibov_swing():
    print("🇧🇷 TREINANDO ESPECIALISTA IBOV SWING (BOLSA BRASILEIRA)...")
    
    df = pd.read_csv('dados/ibov_processado.csv')
    
    # 1. Selecionar features (Preço + Técnicos)
    features = ['close', 'ma7', 'ma21', 'rsi', 'upper_band', 'lower_band']
    
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features])
    
    # 2. Criar Janelas (60h -> Prever 12h futuro para Swing rápido)
    window = 60
    X, y = [], []
    for i in range(len(df_scaled) - window - 12):
        X.append(df_scaled[i:i+window])
        # Alvo: Preço de fechamento em t+12 (relacionado à coluna index 0)
        y.append(df_scaled[i+window+11, 0])
        
    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='float32')

    # 3. Modelo LSTM Multi-Dimensional
    model = Sequential([
        Input(shape=(window, len(features))),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    print(f"🚀 Treinando com {X.shape[0]} amostras e {len(features)} indicadores...")
    # 5 épocas para treinamento base (será refinado no backtest adaptativo)
    model.fit(X, y, epochs=5, batch_size=32, verbose=1)

    # 4. Salvar
    os.makedirs('modelos', exist_ok=True)
    model.save('modelos/ibov_swing_modelo.h5')
    print("✅ MODELO IBOV SALVO: modelos/ibov_swing_modelo.h5")

if __name__ == "__main__":
    treinar_ibov_swing()
