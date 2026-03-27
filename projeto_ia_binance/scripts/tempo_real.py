import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import time

# =================================================================
# DASHBOARD DE DECISÃO EM TEMPO REAL (ANIMADO)
# =================================================================

def tempo_real(modelo_path='modelos/modelo_nlp_v24h_btc_com_sentimento.h5', dados_path='dados/btc_com_sentimento.csv'):
    if not os.path.exists(modelo_path) or not os.path.exists(dados_path):
        print("Erro: Arquivos necessários não encontrados.")
        return

    # 1. CARREGAR E PREPARAR
    print("Iniciando Motor de Simulação 'Ao Vivo'...")
    model = load_model(modelo_path)
    df = pd.read_csv(dados_path)
    df.fillna(0.0, inplace=True)
    
    # Preparação de Tensors
    dados_features = df[['close', 'Score_Sentimento']].values
    dados_close = df[['close']].values
    
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    dados_norm = scaler_features.fit_transform(dados_features)
    
    scaler_close = MinMaxScaler(feature_range=(0, 1))
    scaler_close.fit(dados_close)

    janela = 60
    # Vamos simular as últimas 100 horas do arquivo pra ter velocidade
    ponto_inicio = len(df) - 150 
    
    plt.ion() # Modo interativo do Matplotlib
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    precos_reais = []
    previsoes_ia = []
    tempos = []

    print("\n" + "="*50)
    print("⚡ IA EXECUTANDO EM MODO 'ORÁCULO' (TEMPO REAL)")
    print("="*50)

    for i in range(ponto_inicio, len(df) - 24):
        # 1. Preparar o input das últimas 60h para a IA
        X_input = dados_norm[i-janela:i, :]
        X_input = np.reshape(X_input, (1, janela, 2))
        
        # 2. IA Pensa e Decide
        pred_norm = model.predict(X_input, verbose=0)
        preco_futuro = scaler_close.inverse_transform(pred_norm)[0][0]
        
        # 3. Pegar dados atuais
        preco_atual = df.iloc[i]['close']
        sentimento_atual = df.iloc[i]['Score_Sentimento']
        timestamp = df.iloc[i]['timestamp']
        
        # 4. Lógica de Decisão
        diff_pct = ((preco_futuro - preco_atual) / preco_atual) * 100
        decisao = " Aguardando... "
        cor_decisao = "gray"
        
        if diff_pct > 0.8:
            decisao = "💰 COMPRAR (ALTA DETECTADA) "
            cor_decisao = "green"
        elif diff_pct < -0.8:
            decisao = "⚠️ VENDER (QUEDA DETECTADA) "
            cor_decisao = "red"

        # 5. Atualizar Listas para o Gráfico
        precos_reais.append(preco_atual)
        previsoes_ia.append(preco_futuro)
        tempos.append(i - ponto_inicio)

        # 6. Limpar e Plotar ax1 (Preço)
        ax1.clear()
        ax1.plot(tempos, precos_reais, label='Preço Real (Live)', color='blue', linewidth=2)
        ax1.scatter(tempos[-1], precos_reais[-1], color='blue', s=100)
        ax1.set_title(f"Monitoramento: {timestamp} | Preço: ${preco_atual:,.2f}")
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 7. Plotar ax2 (Sentimento/Decisão)
        ax2.clear()
        barra_cor = 'green' if sentimento_atual > 0 else ('red' if sentimento_atual < 0 else 'gray')
        ax2.bar(["Sentimento News"], [sentimento_atual], color=barra_cor)
        ax2.set_ylim(-1, 1)
        ax2.set_title(f"Decisão da IA: {decisao} ({diff_pct:+.2f}%)")
        
        plt.pause(0.5) # Efeito de "Tempo Real"
        
        # Log no terminal
        print(f"[{timestamp}] Preço: ${preco_atual:,.2f} | Sentimento: {sentimento_atual:+.2f} | IA diz: {decisao}")

    plt.ioff()
    print("\nSimulação de Tempo Real Finalizada.")
    plt.show()

if __name__ == "__main__":
    tempo_real()
