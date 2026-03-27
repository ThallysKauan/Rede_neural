import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os
import time
import yfinance as yf
from textblob import TextBlob
from gnews import GNews
from datetime import datetime

# =================================================================
# 🔮 ORÁCULO IBOV LIVE v4.0 - MONITORAMENTO DE RENDA ATIVA
# O Olho da IA sobre a B3 e as notícias brasileiras.
# =================================================================

def pegar_ponto_ibov():
    try:
        # Busca o último minuto do Ibovespa
        ibov = yf.Ticker("^BVSP")
        dados = ibov.history(period="1d", interval="1m")
        if not dados.empty:
            return dados.iloc[-1]
        return None
    except:
        return None

def main():
    print("="*60)
    print("🔮  ORÁCULO IBOV v4.0 - MODO RENDA ATIVA (B3)")
    print("📡  Conectando ao Yahoo Finance e Google News Brasil...")
    print("="*60)

    # 1. CARREGAR CÉREBRO
    model_path = 'modelos/ibov_swing_modelo.h5'
    if not os.path.exists(model_path):
        print("❌ Erro: Cérebro models/ibov_swing_modelo.h5 não encontrado.")
        return
    
    model = load_model(model_path, compile=False)
    
    # 2. CARREGAR BASE PARA NORMALIZAÇÃO
    dados_path = 'dados/ibov_processado.csv'
    if not os.path.exists(dados_path):
        print("❌ Erro: Base de dados processada não encontrada.")
        return
    df_ref = pd.read_csv(dados_path)
    
    features = ['close', 'ema9', 'ema21', 'rsi', 'macd', 'stoch_k', 'atr']
    scaler = MinMaxScaler()
    scaler.fit(df_ref[features].values)
    
    scaler_p = MinMaxScaler()
    scaler_p.fit(df_ref[['close']].values)

    # Scraper de Notícias (Brasil)
    google_news = GNews(language='pt', country='BR', max_results=3)

    print("\n🛡️ PAREDE DE RISCO ATIVADA. Aguardando sinal de alta confiança...")
    
    try:
        while True:
            agora = datetime.now().strftime('%H:%M:%S')
            print(f"\n[{agora}] 🕵️  Analisando mercado brasileiro...")
            
            # --- NOTÍCIAS BR ---
            noticias = google_news.get_news("ibovespa economia")
            sentimento = 0.0
            if noticias:
                scores = []
                for n in noticias[:2]:
                    txt = n.get('title', '')
                    # TextBlob em PT é limitado, mas detecta palavras chave
                    blob = TextBlob(txt)
                    score = blob.sentiment.polarity
                    scores.append(score)
                    print(f"   📰 Notícia: {txt[:50]}... ({score:+.2f})")
                sentimento = sum(scores) / len(scores)

            # --- PREÇO LIVE (POINTS) ---
            linha_atual = pegar_ponto_ibov()
            if linha_atual is None:
                print("   ⚠️ Erro ao obter dados do IBOV. Tentando novamente...")
                time.sleep(15)
                continue
                
            ponto_atual = linha_atual['Close']
            print(f"   📊 Ibovespa agora: {ponto_atual:,.0f} pts")

            # --- PENSAMENTO DA IA ---
            # Para o live, simulamos os indicadores da última linha
            # (Em um sistema real, recalcularíamos os EMAs, RSI, etc. com o dado novo)
            # Para este MVP, usamos a última linha da memória + o ponto de agora
            memoria_recente = df_ref[features].tail(59).values
            
            # Mock de indicadores para o ponto de agora (simplificado)
            # Idealmente recalculados aqui
            dado_agora = memoria_recente[-1].copy()
            dado_agora[0] = ponto_atual # Substitui pelo preço real
            
            input_ia = np.vstack([memoria_recente, [dado_agora]])
            input_norm = scaler.transform(input_ia)
            input_tensor = np.reshape(input_norm, (1, 60, len(features)))
            
            # Predição
            pred = model.predict(input_tensor, verbose=0)
            p_previsto = scaler_p.inverse_transform(pred)[0][0]
            
            diff_pct = ((p_previsto - ponto_atual) / ponto_atual) * 100
            
            # --- VEREDITO COM TRAVA DE RISCO ---
            print("-" * 40)
            print(f"🤖 DECISÃO DA REDE NEURAL (Alvo 12h):")
            
            if diff_pct > 0.8:
                print(f"🔥 OPORTUNIDADE DE COMPRA! (Previsão: {p_previsto:,.0f} | {diff_pct:+.2f}%)")
            elif diff_pct < -0.8:
                print(f"❄️ OPORTUNIDADE DE VENDA (SHORT)! (Previsão: {p_previsto:,.0f} | {diff_pct:+.2f}%)")
            else:
                print(f"⚖️ AGUARDAR (Mercado sem sinal claro). {diff_pct:+.2f}%")
            print("-" * 40)

            print("\n⏰ Próxima varredura em 60 segundos...")
            time.sleep(60)

    except KeyboardInterrupt:
        print("\n\n🛑 Oráculo Ibov desligado. Renda ativa pausada.")

if __name__ == "__main__":
    main()
