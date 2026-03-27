import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import os

def treinar_ibov_swing():
    print("🇧🇷 TREINANDO CÉREBRO ESPECIALISTA IBOV (MODO CONSISTÊNCIA)...")
    
    caminho_dados = 'dados/ibov_processado.csv'
    if not os.path.exists(caminho_dados):
        print(f"❌ Erro: Arquivo {caminho_dados} não encontrado. Gere os dados primeiro.")
        return

    df = pd.read_csv(caminho_dados)
    
    # 1. Selecionar features
    features = ['close', 'ema9', 'ema21', 'rsi', 'macd', 'stoch_k', 'atr']
    
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features])
    
    # 2. Criar Janelas
    window = 60
    X, y = [], []
    for i in range(len(df_scaled) - window - 12):
        X.append(df_scaled[i:i+window])
        y.append(df_scaled[i+window+11, 0])
        
    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='float32')

    # 3. Modelo LSTM
    model = Sequential([
        Input(shape=(window, len(features))),
        LSTM(150, return_sequences=True),
        Dropout(0.3),
        LSTM(80, return_sequences=True),
        Dropout(0.2),
        LSTM(40),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # --- PROTEÇÃO CONTRA DECOREBA (ANTI-OVERFITTING) ---
    # Para de treinar se o erro no "futuro invisível" parar de cair.
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # Salva apenas a versão que melhor previu o futuro invisível.
    checkpoint = ModelCheckpoint('modelos/ibov_swing_modelo.h5', monitor='val_loss', save_best_only=True)

    print(f"🚀 Iniciando treinamento com {X.shape[0]} amostras...")
    # Usamos 20% dos dados APENAS para validação (a rede nunca treina neles)
    model.fit(X, y, 
              epochs=20, 
              batch_size=64, 
              validation_split=0.2, 
              callbacks=[early_stop, checkpoint],
              verbose=1)

    print("✅ CÉREBRO IBOV ATUALIZADO E PROTEGIDO: modelos/ibov_swing_modelo.h5")

if __name__ == "__main__":
    treinar_ibov_swing()
