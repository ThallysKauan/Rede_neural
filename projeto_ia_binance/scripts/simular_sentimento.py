import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import sys

# =================================================================
# SCRIPT DE SIMULAÇÃO (BACKTESTING CÍBRIDO/MULTIVARIADO)
# =================================================================

def simular_sentimento(modelo_path='modelos/modelo_nlp_v24h_btc_com_sentimento.h5', dados_path='dados/btc_com_sentimento.csv'):
    if not os.path.exists(modelo_path):
        print(f"Erro: Cérebro Neural '{modelo_path}' não encontrado. Foi treinado?")
        return

    if not os.path.exists(dados_path):
        print(f"Erro: Dados Históricos '{dados_path}' não encontrados.")
        return

    # 1. CARREGAR BASES
    print(f"Acoplando Cérebro Neural à base Histórica (Preço + NLP)...")
    model = load_model(modelo_path)
    df = pd.read_csv(dados_path)
    
    # Lidar com vazios previnindo NullPointer
    df.fillna(0.0, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extrair as DUAS colunas em um único tensor
    dados_features = df[['close', 'Score_Sentimento']].values
    dados_close = df[['close']].values
    
    # 2. NORMALIZAR (DUAS ESCALAS)
    # Escala conjunta (Preço e Notícias são traduzidos pra linguagem da rede)
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    dados_normalizados = scaler_features.fit_transform(dados_features)

    # Escala paralela usada APENAS para devolver o Preço em Dólar Puro
    scaler_close = MinMaxScaler(feature_range=(0, 1))
    scaler_close.fit(dados_close)

    # 3. JANELAMENTO HÍBRIDO
    janela = 60
    previsao_futura = 24
    
    X = []
    y_real = []

    # O loop vai caminhar pela história lendo 60 horas mistas pra prever a frente.
    for i in range(janela, len(dados_normalizados) - previsao_futura):
        X.append(dados_normalizados[i-janela:i, :]) # Captura as matrizes Preço+NLP
        y_real.append(dados_close[i + previsao_futura, 0])

    X = np.array(X)
    
    print(f"\nSimulando matematicamente em {len(X)} horas de negociação (Correlacionando Notícias com a Tendência Gráfica)...")

    # 4. INFERÊNCIA DA REDE NEURAL (PREVISÕES CÍBRIDAS)
    previsoes_normalizadas = model.predict(X)

    # 5. DES-NORMALIZAR PREVISÕES (Traduzindo de volta pra USDT)
    previsoes_preco = scaler_close.inverse_transform(previsoes_normalizadas)

    # 6. ANÁLISE ESTATÍSTICA MATRIZ
    # Sincronizamos os dias previstos com as datas reais da panilha
    df_sim = df.iloc[janela + previsao_futura : janela + previsao_futura + len(previsoes_preco)].copy()
    
    df_sim['previsao'] = previsoes_preco.flatten()
    df_sim['erro_abs'] = abs(df_sim['close'] - df_sim['previsao'])
    df_sim['erro_pct'] = (df_sim['erro_abs'] / df_sim['close']) * 100

    erro_medio = df_sim['erro_pct'].mean()
    print(f"\n✅ Precisão do Modelo Validada. Erro Médio Absoluto: {erro_medio:.2f}%")

    # 7. EXIBIÇÃO VISUAL (DASHBOARD)
    plt.figure(figsize=(15, 8))
    plt.plot(df_sim['timestamp'], df_sim['close'], color='blue', alpha=0.5, label='Curva de Preço Real do Mercado')
    plt.plot(df_sim['timestamp'], df_sim['previsao'], color='orange', alpha=0.9, label='Pensamento Híbrido da IA (Grafico + Sentimento Textual)', linewidth=2)
    plt.title(f'Simulação Híbrida de 24 meses (Acerto Médio: {100 - erro_medio:.2f}%)')
    plt.xlabel('Linha do Tempo (2 Anos)')
    plt.ylabel('Cotação em USDT (Dólar Tether)')
    plt.legend()
    plt.grid(True)
    
    print("\nSimulação concluída! Disparando imagem gráfica do Python na sua tela...")
    plt.show()

if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else 'modelos/modelo_nlp_v24h_btc_com_sentimento.h5'
    d = sys.argv[2] if len(sys.argv) > 2 else 'dados/btc_com_sentimento.csv'
    simular_sentimento(m, d)
