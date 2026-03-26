import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import sys

# =================================================================
# SCRIPT DE SIMULAÇÃO EM LOTE (BACKTESTING MULTI-MOEDAS) - v2.1
# Gera ranking de performance e SALVA os gráficos em /graficos
# =================================================================

def simular_lote(modelo_path, folder_dados='dados'):
    if not os.path.exists(modelo_path):
        print(f"Erro: Modelo '{modelo_path}' não encontrado.")
        return

    # Criar pasta para os gráficos
    folder_graficos = 'graficos'
    if not os.path.exists(folder_graficos):
        os.makedirs(folder_graficos)

    # Listar arquivos CSV de 24 meses
    arquivos = [f for f in os.listdir(folder_dados) if f.endswith('24m.csv')]
    
    if not arquivos:
        print(f"Erro: Nenhum arquivo de 24 meses encontrado em '{folder_dados}'.")
        return

    print(f"Lendo modelo: {modelo_path}")
    model = load_model(modelo_path)
    
    resultados = []

    print(f"\n--- Iniciando Simulação em Lote ({len(arquivos)} arquivos) ---")
    
    for arquivo in arquivos:
        dados_path = os.path.join(folder_dados, arquivo)
        moeda_nome = arquivo.replace('_24m.csv', '').upper()
        
        print(f"Simulando {moeda_nome}...")
        
        try:
            df = pd.read_csv(dados_path)
            # Garantir ordenação temporal
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values(by='timestamp')
            
            dados = df['close'].values.reshape(-1, 1)
            
            # Normalizar
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

            # Previsões
            previsoes = model.predict(X, verbose=0)

            # Des-normalizar
            previsoes_preço = scaler.inverse_transform(previsoes)
            y_real_preço = scaler.inverse_transform(np.array(y_real).reshape(-1, 1))

            # Calcular Erro Médio Percentual (MAPE)
            erro_abs = np.abs(y_real_preço - previsoes_preço)
            erro_pct = np.mean((erro_abs / y_real_preço) * 100)
            
            resultados.append({
                'Moeda': moeda_nome,
                'Erro Médio (%)': round(erro_pct, 4)
            })

            # SALVAR GRÁFICO
            plt.figure(figsize=(12, 6))
            plt.plot(df['timestamp'].iloc[janela:], y_real_preço, color='blue', alpha=0.6, label='Real')
            plt.plot(df['timestamp'].iloc[janela:], previsoes_preço, color='red', alpha=0.6, label='IA')
            plt.title(f'Performance: {moeda_nome} | Erro: {round(erro_pct, 2)}%')
            plt.legend()
            plt.grid(True)
            
            grafico_path = os.path.join(folder_graficos, f"{moeda_nome}.png")
            plt.savefig(grafico_path)
            plt.close() # Importante fechar para não consumir memória
            
        except Exception as e:
            print(f"Erro ao processar {moeda_nome}: {e}")

    # Exibir Tabela Final
    df_res = pd.DataFrame(resultados).sort_values(by='Erro Médio (%)')
    
    print("\n" + "="*40)
    print("      RANKING DE PERFORMANCE DA IA")
    print("="*40)
    print(df_res.to_string(index=False))
    print("="*40)
    print(f"\n--- SUCESSO! ---")
    print(f"Os gráficos de cada moeda foram salvos na pasta: /{folder_graficos}")

if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else 'modelos/modelo_eth_usdt_24m.h5'
    simular_lote(m)
