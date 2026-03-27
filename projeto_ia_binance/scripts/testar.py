import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import sys

# =================================================================
# SCRIPT DE TESTE UNIFICADO (v2.0)
# =================================================================

def testar(modelo_path='modelos/modelo_btc.h5', dados_csv='dados/eth_usdt_24m.csv'):
    if not os.path.exists(modelo_path):
        print(f"Erro: Modelo '{modelo_path}' não encontrado.")
        return
    if not os.path.exists(dados_csv):
        print(f"Erro: Dados '{dados_csv}' não encontrados.")
        return

    print(f"Testando {modelo_path} com dados de {dados_csv}...")
    model = load_model(modelo_path)
    df = pd.read_csv(dados_csv)
    dados = df['close'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    dados_normalizados = scaler.fit_transform(dados)

    janela = 60
    X = []
    y_real = []

    for i in range(janela, len(dados_normalizados)):
        X.append(dados_normalizados[i-janela:i, 0])
        y_real.append(dados_normalizados[i, 0])

    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    previsoes = model.predict(X)
    previsoes_preço = scaler.inverse_transform(previsoes)
    y_real_preço = scaler.inverse_transform(np.array(y_real).reshape(-1, 1))

    plt.figure(figsize=(12, 6))
    plt.plot(y_real_preço, color='blue', label='Real')
    plt.plot(previsoes_preço, color='red', label='Previsão')
    plt.title(f'Teste: {os.path.basename(modelo_path)} em {os.path.basename(dados_csv)}')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else 'modelos/modelo_btc.h5'
    d = sys.argv[2] if len(sys.argv) > 2 else 'dados/eth_usdt_24m.csv'
    testar(m, d)
