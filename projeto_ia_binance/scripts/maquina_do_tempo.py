import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import sys

# =================================================================
# ⏳ MÁQUINA DO TEMPO: SIMULAÇÃO ACELERADA (24 MESES)
# =================================================================

def maquina_do_tempo(modelo_path='modelos/modelo_nlp_v24h_btc_com_sentimento.h5', dados_path='dados/btc_com_sentimento.csv'):
    if not os.path.exists(modelo_path) or not os.path.exists(dados_path):
        print("Erro: Arquivos necessários não encontrados.")
        return

    # 1. CARREGAR IA E DADOS
    print("🛸 Ligando Máquina do Tempo... Voltando para Abril de 2024.")
    model = load_model(modelo_path)
    df = pd.read_csv(dados_path)
    df.fillna(0.0, inplace=True)
    
    # Preparar Tensor
    dados_features = df[['close', 'Score_Sentimento']].values
    dados_close = df[['close']].values
    
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    dados_norm = scaler_features.fit_transform(dados_features)
    
    scaler_close = MinMaxScaler(feature_range=(0, 1))
    scaler_close.fit(dados_close)

    janela = 60
    
    # 2. CONFIGURAÇÃO DO GRÁFICO
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    precos_visiveis = []
    previsoes_ia = []
    tempos = []
    
    saldo_fake = 1000 # Começamos com 1.000 USDT
    posicao = 0 # 0 = fora, 1 = dentro
    
    print("\n🚀 VIAGEM INICIADA! Cruzando 17.000 horas de histórico em alta velocidade...")

    # Começamos logo após a primeira janela de 60h
    for i in range(janela, len(df) - 24):
        # --- IA Pensa no "ado" ---
        X_input = dados_norm[i-janela:i, :]
        X_input = np.reshape(X_input, (1, janela, 2))
        
        pred_norm = model.predict(X_input, verbose=0)
        preco_previsto = scaler_close.inverse_transform(pred_norm)[0][0]
        
        preco_atual = df.iloc[i]['close']
        timestamp = df.iloc[i]['timestamp']
        sentimento = df.iloc[i]['Score_Sentimento']
        
        # --- Lógica de Simulação de Trade ---
        diff_pct = ((preco_previsto - preco_atual) / preco_atual) * 100
        decisao = "AGUARDAR"
        cor_decisao = "gray"

        if diff_pct > 0.8:
            decisao = "BUY 💰"
            cor_decisao = "green"
        elif diff_pct < -0.8:
            decisao = "SELL ⚠️"
            cor_decisao = "red"

        # --- Atualizar Dados Visuais ---
        precos_visiveis.append(preco_atual)
        tempos.append(i)
        
        # Manter apenas as últimas 100h na tela para velocidade
        if len(precos_visiveis) > 100:
            precos_visiveis.pop(0)
            tempos.pop(0)

        # Atualizar Gráfico (Otimizado)
        if i % 5 == 0: # Só redesenha a cada 5 horas simuladas para IR MAIS RÁPIDO
            ax1.clear()
            ax1.plot(tempos, precos_visiveis, color='blue', linewidth=2)
            ax1.set_title(f"MÁQUINA DO TEMPO | DATA: {timestamp} | Preço: ${preco_atual:,.2f}")
            ax1.set_ylabel("BTC / USDT")
            ax1.grid(True, alpha=0.3)
            
            # Label de Decisão
            ax1.text(0.95, 0.95, decisao, transform=ax1.transAxes, color=cor_decisao, 
                     fontsize=25, fontweight='bold', ha='right', va='top', bbox=dict(facecolor='white', alpha=0.8))

            ax2.clear()
            ax2.bar(["Sentimento na Época"], [sentimento], color=cor_decisao)
            ax2.set_ylim(-1, 1)
            ax2.set_title(f"Inteligência IA: Score de Sentimento no momento: {sentimento:+.2f}")
            
            plt.pause(0.001) # Ultra rápido

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    maquina_do_tempo()
