import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
import sys

# =================================================================
# SCRIPT DE TREINAMENTO 24h (PRO)
# Objetivo: Prever o preço de 24 horas no futuro.
# =================================================================

def treinar_24h(dados_csv='dados/eth_usdt_24m.csv', epochs=20):
    if not os.path.exists(dados_csv):
        print(f"Erro: Arquivo '{dados_csv}' não encontrado.")
        return

    # 1. CARREGAR
    print(f"Carregando dados para treino de 24h: {dados_csv}...")
    df = pd.read_csv(dados_csv)
    dados = df['close'].values.reshape(-1, 1)

    # 2. NORMALIZAR
    scaler = MinMaxScaler(feature_range=(0, 1))
    dados_normalizados = scaler.fit_transform(dados)

    # 3. JANELAMENTO (AVANÇADO)
    # X: últimos 60 preços
    # y: preço de DAQUI A 24 HORAS
    janela = 60
    previsao_futura = 24 # Horas no futuro
    
    X = []
    y = []

    # Precisamos parar 24 horas antes do fim do arquivo para ter o "alvo"
    for i in range(janela, len(dados_normalizados) - previsao_futura):
        X.append(dados_normalizados[i-janela:i, 0])
        y.append(dados_normalizados[i + previsao_futura, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # 4. MODELO REFORÇADO (Camada extra para o desafio de 24h)
    print("Construindo rede neural Pro (3 camadas LSTM)...")
    model = Sequential([
        LSTM(units=64, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=64, return_sequences=True),
        Dropout(0.2),
        LSTM(units=64),
        Dropout(0.2),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # 5. TREINAR
    print(f"Iniciando treinamento pesado ({epochs} épocas)...")
    model.fit(X, y, epochs=epochs, batch_size=32)

    # 6. SALVAR
    if not os.path.exists('modelos'):
        os.makedirs('modelos')
        
    base_nome = os.path.basename(dados_csv).replace('.csv', '')
    nome_modelo = f"modelos/modelo_v24h_{base_nome}.h5"
    model.save(nome_modelo)
    print(f"\n--- SUCESSO! ---")
    print(f"IA de 24h treinada e salva como '{nome_modelo}'!")

if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else 'dados/eth_usdt_24m.csv'
    e = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    treinar_24h(d, e)
