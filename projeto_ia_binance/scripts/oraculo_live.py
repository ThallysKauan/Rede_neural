import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os
import time
import requests
from textblob import TextBlob
from gnews import GNews
from datetime import datetime, timedelta

# =================================================================
# 🔮 ORÁCULO LIVE - SCRAPER + PREÇO EM TEMPO REAL
# O Robô que lê a internet agora e decide o futuro.
# =================================================================

def pegar_preco_binance():
    try:
        url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
        res = requests.get(url).json()
        return float(res['price'])
    except:
        return None

def main():
    print("="*60)
    print("🔮  ORÁCULO BITCOIN v3.0 - MODO OPERAÇÃO VIVA")
    print("📡  Conectando ao Google News Scraper e Binance API...")
    print("="*60)

    # 1. CARREGAR CÉREBRO
    modelo_path = 'modelos/modelo_nlp_v24h_btc_com_sentimento.h5'
    if not os.path.exists(modelo_path):
        print("Erro: Cérebro (modelo .h5) não encontrado.")
        return
    
    model = load_model(modelo_path)
    
    # 2. CARREGAR BASE HISTÓRICA PARA ALIMENTAR A "MEMÓRIA" DA IA
    # A IA precisa saber as últimas 60 horas para poder decidir agora.
    dados_path = 'dados/btc_com_sentimento.csv'
    df_historico = pd.read_csv(dados_path)
    df_historico.fillna(0.0, inplace=True)
    
    # Scalers (re-instanciando com a base de treino)
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_features.fit(df_historico[['close', 'Score_Sentimento']].values)
    
    scaler_close = MinMaxScaler(feature_range=(0, 1))
    scaler_close.fit(df_historico[['close']].values)

    google_news = GNews(language='en', country='US', max_results=5)

    print("\n🟢 SISTEMA ONLINE. Iniciando loop de monitoramento infinto...")
    
    try:
        while True:
            agora = datetime.now().strftime('%H:%M:%S')
            print(f"\n[{agora}] 🕵️  Scrappy saindo para caçar notícias frescas...")
            
            # --- SCRAPER EM TEMPO REAL ---
            noticias = google_news.get_news("bitcoin")
            sentimento_medio = 0.0
            manchetes_vistas = 0
            
            if noticias:
                scores = []
                for n in noticias[:3]: # Pega as 3 mais quentes
                    texto = n.get('title', '')
                    blob = TextBlob(texto)
                    score = blob.sentiment.polarity
                    scores.append(score)
                    print(f"   📰 Notícia: {texto[:60]}... (Sentimento: {score:+.2f})")
                sentimento_medio = sum(scores) / len(scores)
                manchetes_vistas = len(scores)
            else:
                print("   📭 Nenhuma notícia bombástica encontrada no Google agora.")

            # --- PREÇO EM TEMPO REAL ---
            preco_atual = pegar_preco_binance()
            if not preco_atual:
                print("   ❌ Erro ao conectar na Binance. Tentando de novo...")
                time.sleep(10)
                continue
                
            print(f"   💰 Preço atual detectado: ${preco_atual:,.2f}")

            # --- PENSAMENTO DA IA ---
            # Pegamos as últimas 59 horas do CSV + o dado de AGORA (total 60)
            memoria_recente = df_historico[['close', 'Score_Sentimento']].tail(59).values
            dado_agora = np.array([[preco_atual, sentimento_medio]])
            input_ia = np.vstack([memoria_recente, dado_agora])
            
            # Normalizar para a IA
            input_norm = scaler_features.transform(input_ia)
            input_tensor = np.reshape(input_norm, (1, 60, 2))
            
            # Predição
            pred_norm = model.predict(input_tensor, verbose=0)
            preco_previsto = scaler_close.inverse_transform(pred_norm)[0][0]
            
            diff_pct = ((preco_previsto - preco_atual) / preco_atual) * 100
            
            # --- VEREDITO ---
            print("\n" + "-"*40)
            print(f"🤖 DECISÃO DO CÉREBRO NEURAL (Próximas 24h):")
            
            if diff_pct > 0.8:
                print(f"🔥 COMPRAR FORTE! Previsão de alta: ${preco_previsto:,.2f} ({diff_pct:+.2f}%)")
            elif diff_pct < -0.8:
                print(f"❄️ VENDER! Previsão de queda: ${preco_previsto:,.2f} ({diff_pct:+.2f}%)")
            else:
                print(f"⚖️ AGUARDAR (Mercado em estabilidade). Previsão: ${preco_previsto:,.2f} ({diff_pct:+.2f}%)")
            print("-"*40)

            print("\n⏰ Próxima varredura em 60 segundos... (ou aperte Ctrl+C para parar)")
            time.sleep(60)

    except KeyboardInterrupt:
        print("\n\n🛑 Oráculo desligado com segurança. Até a próxima!")

if __name__ == "__main__":
    main()
