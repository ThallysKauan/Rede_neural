import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
import sys

# =================================================================
# SCRIPT DE TREINAMENTO HÍBRIDO (Preço + Notícias/NLP)
# Objetivo: Prever o preço de 24 horas no futuro usando duas variáveis.
# =================================================================

def treinar_sentimento(dados_csv='dados/btc_com_sentimento.csv', epochs=20):
    if not os.path.exists(dados_csv):
        print(f"Erro: Arquivo '{dados_csv}' não encontrado. Rode o mesclar_dados.py primeiro.")
        return

    # 1. CARREGAR
    print(f"Carregando base de dados turbinada: {dados_csv}...")
    df = pd.read_csv(dados_csv)
    
    # Vamos usar Fechamento e Score de Sentimento combinados
    df_features = df[['close', 'Score_Sentimento']].copy()
    df_features.fillna(0.0, inplace=True)
    
    dados = df_features.values

    # 2. NORMALIZAR MÚLTIPLAS VARIÁVEIS
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    dados_normalizados = scaler_features.fit_transform(dados)

    # 3. JANELAMENTO (AVANÇADO MULTIVARIADO)
    janela = 60
    previsao_futura = 24 # Prever o fechamento de amanhã
    
    X = []
    y = []

    # O alvo y é especificamente a coluna 0 (close) daqui a 24h
    for i in range(janela, len(dados_normalizados) - previsao_futura):
        X.append(dados_normalizados[i-janela:i, :]) # Pega todas as colunas
        y.append(dados_normalizados[i + previsao_futura, 0]) # Pega só o close (Preço)

    X, y = np.array(X), np.array(y)
    
    print(f"Formato da Memória Curta: X tem formato {X.shape}")
    print(f"Ou seja: {X.shape[0]} amostras, olhando {X.shape[1]} horas para trás, analisando {X.shape[2]} variáveis simultâneas.")

    # 4. MODELO REFORÇADO (LSTM Multivariada)
    print("\nConstruindo cérebro neural Multivariado (3 camadas LSTM)...")
    model = Sequential([
        LSTM(units=64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(units=64, return_sequences=True),
        Dropout(0.2),
        LSTM(units=64),
        Dropout(0.2),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # 5. TREINAR
    print(f"\nIniciando treinamento cognitivo super-intensivo de {epochs} épocas com Sentimento de Mercado...")
    model.fit(X, y, epochs=epochs, batch_size=32)

    # 6. SALVAR
    if not os.path.exists('modelos'):
        os.makedirs('modelos')
        
    base_nome = os.path.basename(dados_csv).replace('.csv', '')
    nome_modelo = f"modelos/modelo_nlp_v24h_{base_nome}.h5"
    model.save(nome_modelo)
    
    print(f"\n--- SUCESSO ABSOLUTO! ---")
    print(f"IA Cíbrida (Gráfico + NLP) treinada e salva como '{nome_modelo}'!")

if __name__ == "__main__":
    # Pega os argumentos do terminal, ou usa os padrões se não houver
    d = sys.argv[1] if len(sys.argv) > 1 else 'dados/btc_com_sentimento.csv'
    e = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    treinar_sentimento(d, e)
