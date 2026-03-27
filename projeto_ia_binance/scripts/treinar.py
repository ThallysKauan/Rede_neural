import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import os
import sys

# =================================================================
# SCRIPT DE TREINAMENTO UNIFICADO (v2.0)
# =================================================================

def treinar(dados_csv='dados/btc_usdt_24m.csv', epochs=10):
    if not os.path.exists(dados_csv):
        print(f"Erro: Arquivo '{dados_csv}' não encontrado.")
        return

    # 1. CARREGAR
    print(f"Carregando dados de {dados_csv}...")
    df = pd.read_csv(dados_csv)
    dados = df['close'].values.reshape(-1, 1)

    # 2. NORMALIZAR
    scaler = MinMaxScaler(feature_range=(0, 1))
    dados_normalizados = scaler.fit_transform(dados)

    # 3. JANELAMENTO
    janela = 60
    X = []
    y = []

    for i in range(janela, len(dados_normalizados)):
        X.append(dados_normalizados[i-janela:i, 0])
        y.append(dados_normalizados[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # 4. MODELO
    print("Construindo rede neural...")
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # 5. TREINAR
    print(f"Iniciando treinamento ({epochs} épocas)...")
    model.fit(X, y, epochs=epochs, batch_size=32)

    # 6. SALVAR
    if not os.path.exists('modelos'):
        os.makedirs('modelos')
        
    nome_modelo = f"modelos/modelo_{os.path.basename(dados_csv).replace('.csv', '')}.h5"
    model.save(nome_modelo)
    print(f"IA Treinada e salva como '{nome_modelo}'!")

if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else 'dados/btc_usdt_24m.csv'
    e = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    treinar(d, e)
