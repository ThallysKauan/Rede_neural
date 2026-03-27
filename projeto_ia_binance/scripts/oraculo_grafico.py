import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import requests
import time
from textblob import TextBlob
from gnews import GNews
from datetime import datetime

# =================================================================
# 🖥️ DASHBOARD VISUAL LIVE (ORÁCULO GRÁFICO)
# O Robô que caça notícias na internet e desenha o gráfico ao vivo.
# =================================================================

def pegar_preco_binance():
    try:
        url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
        res = requests.get(url).json()
        return float(res['price'])
    except:
        return None

def main():
    print("📡 Iniciando Interface Gráfica do Oráculo...")
    
    # 1. CARREGAR IA E DADOS DE APOIO
    modelo_path = 'modelos/modelo_nlp_v24h_btc_com_sentimento.h5'
    dados_path = 'dados/btc_com_sentimento.csv'
    
    if not os.path.exists(modelo_path) or not os.path.exists(dados_path):
        print("Erro: Arquivos de treinamento não encontrados.")
        return

    model = load_model(modelo_path)
    df_hist = pd.read_csv(dados_path)
    df_hist.fillna(0.0, inplace=True)
    
    # Scalers
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_features.fit(df_hist[['close', 'Score_Sentimento']].values)
    
    scaler_close = MinMaxScaler(feature_range=(0, 1))
    scaler_close.fit(df_hist[['close']].values)

    google_news = GNews(language='en', country='US', max_results=3)

    # 2. CONFIGURAÇÃO DO GRÁFICO LIVE
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    precos_plot = []
    sentimentos_plot = []
    tempos_plot = []
    janela = 60

    print("🟢 DASHBOARD OPERACIONAL. Conectado ao Google News e Binance.")

    try:
        while True:
            # --- AÇÃO DO SCRAPPY (NOTÍCIAS AGORA) ---
            noticias = google_news.get_news("bitcoin")
            score_agora = 0.0
            if noticias:
                scores = [TextBlob(n.get('title', '')).sentiment.polarity for n in noticias]
                score_agora = sum(scores) / len(scores)

            # --- PREÇO AGORA ---
            preco_agora = pegar_preco_binance()
            if not preco_agora: 
                time.sleep(5)
                continue

            # --- PENSAMENTO DA IA ---
            memoria = df_hist[['close', 'Score_Sentimento']].tail(59).values
            input_ia = np.vstack([memoria, [preco_agora, score_agora]])
            input_norm = scaler_features.transform(input_ia)
            input_tensor = np.reshape(input_norm, (1, 60, 2))
            
            pred_norm = model.predict(input_tensor, verbose=0)
            preco_previsto = scaler_close.inverse_transform(pred_norm)[0][0]
            
            diff_pct = ((preco_previsto - preco_agora) / preco_agora) * 100
            
            # --- ATUALIZAR LISTAS DO GRÁFICO ---
            agora_str = datetime.now().strftime('%H:%M:%S')
            precos_plot.append(preco_agora)
            sentimentos_plot.append(score_agora)
            tempos_plot.append(agora_str)
            
            # Manter apenas os últimos 50 pontos no gráfico para não pesar
            if len(precos_plot) > 50:
                precos_plot.pop(0)
                sentimentos_plot.pop(0)
                tempos_plot.pop(0)

            # --- DESENHAR ---
            ax1.clear()
            ax1.plot(tempos_plot, precos_plot, marker='o', color='gold', label='Preço Live (Binance)', linewidth=2)
            ax1.set_title(f"ORÁCULO LIVE: BTC/USDT ${preco_agora:,.2f} | Previsão 24h: {diff_pct:+.2f}%")
            ax1.set_ylabel("Preço em Dólar")
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # Sinal de Trade no Gráfico
            if diff_pct > 0.8:
                ax1.text(0.5, 0.9, "💰 SINAL: COMPRA", transform=ax1.transAxes, color='green', fontsize=20, fontweight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.8))
            elif diff_pct < -0.8:
                ax1.text(0.5, 0.9, "⚠️ SINAL: VENDA", transform=ax1.transAxes, color='red', fontsize=20, fontweight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.8))
            else:
                ax1.text(0.5, 0.9, "⚖️ SINAL: AGUARDAR", transform=ax1.transAxes, color='gray', fontsize=20, ha='center')

            ax2.clear()
            cor_bar = 'green' if score_agora > 0 else ('red' if score_agora < 0 else 'gray')
            ax2.bar(["Sentimento Web (Spider)"], [score_agora], color=cor_bar)
            ax2.set_ylim(-1, 1)
            ax2.set_title(f"Última Manchete: {noticias[0].get('title', '')[:50]}...")

            plt.pause(20) # Atualiza a cada 20 segundos para dar tempo do Google/Binance responderem

    except KeyboardInterrupt:
        print("\nDesligando Oráculo...")
        plt.close()

if __name__ == "__main__":
    main()
