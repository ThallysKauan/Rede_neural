import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import sys

# =================================================================
# SCRIPT DE SIMULAÇÃO (BACKTESTING) - v2.0
# Percorre o histórico e avalia o acerto da IA mês a mês.
# =================================================================

def simular(modelo_path='modelos/modelo_btc.h5', dados_path='dados/eth_usdt_24m.csv'):
    if not os.path.exists(modelo_path):
        print(f"Erro: Modelo '{modelo_path}' não encontrado.")
        return

    if not os.path.exists(dados_path):
        print(f"Erro: Dados '{dados_path}' não encontrados. Rode 'coletar.py' primeiro.")
        return

    # 1. CARREGAR
    print(f"Lendo modelo e dados históricos...")
    model = load_model(modelo_path)
    df = pd.read_csv(dados_path)
    
    # Converter timestamp para data se necessário
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    dados = df['close'].values.reshape(-1, 1)
    
    # 2. NORMALIZAR
    scaler = MinMaxScaler(feature_range=(0, 1))
    dados_normalizados = scaler.fit_transform(dados)

    # 3. JANELAMENTO
    janela = 60
    X = []
    y_real = []

    for i in range(janela, len(dados_normalizados)):
        X.append(dados_normalizados[i-janela:i, 0])
        y_real.append(dados_normalizados[i, 0])

    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # 4. PREVISÕES
    print(f"Simulando em {len(X)} pontos de dados...")
    previsoes = model.predict(X)

    # 5. DES-NORMALIZAR
    previsoes_preço = scaler.inverse_transform(previsoes)
    y_real_preço = scaler.inverse_transform(np.array(y_real).reshape(-1, 1))

    # 6. ANÁLISE MENSAL
    # Vamos adicionar as previsões ao DataFrame original (ajustando o índice devido à janela)
    df_sim = df.iloc[janela:].copy()
    df_sim['previsao'] = previsoes_preço.flatten()
    df_sim['erro_abs'] = abs(df_sim['close'] - df_sim['previsao'])
    df_sim['erro_pct'] = (df_sim['erro_abs'] / df_sim['close']) * 100

    # Agrupar por Mês/Ano
    mensal = df_sim.groupby(df_sim['timestamp'].dt.to_period('M')).agg({
        'erro_pct': 'mean',
        'close': 'mean'
    })

    print("\n--- PERFORMANCE POR MÊS ---")
    print(mensal)

    # 7. GRÁFICO
    plt.figure(figsize=(14, 7))
    plt.plot(df_sim['timestamp'], df_sim['close'], color='blue', alpha=0.6, label='Preço Real')
    plt.plot(df_sim['timestamp'], df_sim['previsao'], color='red', alpha=0.6, label='Previsão da IA')
    plt.title(f'Simulação Histórica: {modelo_path} vs {dados_path}')
    plt.xlabel('Data')
    plt.ylabel('Preço (USDT)')
    plt.legend()
    plt.grid(True)
    
    print("\nSimulação concluída! Exibindo gráfico...")
    plt.show()

if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else 'modelos/modelo_btc.h5'
    d = sys.argv[2] if len(sys.argv) > 2 else 'dados/eth_usdt_24m.csv'
    simular(m, d)
