import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import sys

# =================================================================
# SCRIPT DE TESTE 24h (PRO)
# Mostra a previsão 24 horas à frente em relação ao preço real.
# =================================================================

def testar_24h(modelo_path='modelos/modelo_v24h_eth_usdt_24m.h5', dados_csv='dados/eth_usdt_24m.csv'):
    if not os.path.exists(modelo_path):
        print(f"Erro: Modelo '{modelo_path}' não encontrado.")
        return
    if not os.path.exists(dados_csv):
        print(f"Erro: Dados '{dados_csv}' não encontrados.")
        return

    print(f"Lendo modelo de 24h: {modelo_path}")
    model = load_model(modelo_path)
    df = pd.read_csv(dados_csv)
    dados = df['close'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    dados_normalizados = scaler.fit_transform(dados)

    janela = 60
    previsao_futura = 24
    
    X = []
    y_real = []

    # Mesma lógica do treino para alinhar os dados corretamente
    for i in range(janela, len(dados_normalizados) - previsao_futura):
        X.append(dados_normalizados[i-janela:i, 0])
        y_real.append(dados_normalizados[i + previsao_futura, 0])

    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    print(f"Fazendo previsões de 24 horas à frente...")
    previsoes = model.predict(X, verbose=0)
    
    # Des-normalizar para preço real
    previsoes_preço = scaler.inverse_transform(previsoes)
    y_real_preço = scaler.inverse_transform(np.array(y_real).reshape(-1, 1))

    # GRÁFICO
    plt.figure(figsize=(14, 7))
    plt.plot(y_real_preço, color='blue', alpha=0.5, label='Preço Real (O que aconteceu)')
    plt.plot(previsoes_preço, color='green', alpha=0.7, label='Previsão da IA (O que ela disse 24h antes)')
    
    plt.title(f'Desafio Pro: Previsão de 24 Horas ({os.path.basename(modelo_path)})')
    plt.xlabel('Tempo (Horas)')
    plt.ylabel('Preço (USDT)')
    plt.legend()
    plt.grid(True)
    
    print("\nTeste concluído! Observe se a linha verde 'antecipa' os movimentos da azul.")
    plt.show()

if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else 'modelos/modelo_v24h_eth_usdt_24m.h5'
    d = sys.argv[2] if len(sys.argv) > 2 else 'dados/eth_usdt_24m.csv'
    testar_24h(m, d)
