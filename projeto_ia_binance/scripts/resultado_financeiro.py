import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os

# =================================================================
# 💹 RELATÓRIO DE PERFORMANCE FINANCEIRA (ROI 24 MESES)
# =================================================================

def auditoria_ia():
    modelo_path = 'modelos/modelo_nlp_v24h_btc_com_sentimento.h5'
    dados_path = 'dados/btc_com_sentimento.csv'
    
    if not os.path.exists(modelo_path) or not os.path.exists(dados_path):
        print("Erro: Arquivos necessários não encontrados.")
        return

    # 1. SETUP
    print("📋 Iniciando Auditoria Financeira Detalhada (2024 - 2026)...")
    model = load_model(modelo_path)
    df = pd.read_csv(dados_path)
    df.fillna(0.0, inplace=True)
    
    # Dados para a IA
    dados_features = df[['close', 'Score_Sentimento']].values
    dados_close = df[['close']].values
    
    scaler_f = MinMaxScaler(feature_range=(0, 1))
    dados_norm = scaler_f.fit_transform(dados_features)
    
    scaler_c = MinMaxScaler(feature_range=(0, 1))
    scaler_c.fit(dados_close)

    janela = 60
    
    # --- VARIÁVEIS DA CARTEIRA ---
    SALDO_INICIAL = 1000.0 # Começamos com $1.000,00
    usdt = SALDO_INICIAL
    btc = 0.0
    taxa_binance = 0.001 # 0.1% de taxa por trade
    
    historico_patrimonio = []
    total_trades = 0
    esta_posicionado = False

    print(f"💰 Capital Inicial: ${SALDO_INICIAL:,.2f} USDT")
    print("⏳ Processando 17.000 horas de trade... Aguarde.")

    # Loop de Trade (Simulando o passado hora a hora)
    for i in range(janela, len(df) - 24, 24): # Pulamos de 24 em 24h para simular swings diários
        # Input IA
        X = np.reshape(dados_norm[i-janela:i, :], (1, janela, 2))
        
        # IA Prediz
        pred = model.predict(X, verbose=0)
        preco_previsto = scaler_c.inverse_transform(pred)[0][0]
        
        preco_atual = df.iloc[i]['close']
        diff_pct = ((preco_previsto - preco_atual) / preco_atual) * 100
        
        # --- ESTRATÉGIA DE EXECUÇÃO ---
        # Se IA prevê alta > 1% e temos USDT -> COMPRAMOS
        if diff_pct > 1.0 and not esta_posicionado:
            btc = (usdt * (1 - taxa_binance)) / preco_atual
            usdt = 0
            esta_posicionado = True
            total_trades += 1
            # print(f"   [BUY]  @ ${preco_atual:,.2f}")

        # Se IA prevê queda < -1% e temos BTC -> VENDEMOS
        elif diff_pct < -1.0 and esta_posicionado:
            usdt = (btc * preco_atual) * (1 - taxa_binance)
            btc = 0
            esta_posicionado = False
            total_trades += 1
            # print(f"   [SELL] @ ${preco_atual:,.2f}")

    # --- RESULTADO FINAL ---
    # Caso tenha sobrado BTC no final, vendemos pelo preço atual pra fechar a conta
    if esta_posicionado:
        preco_final = df.iloc[-1]['close']
        usdt = (btc * preco_final) * (1 - taxa_binance)
        btc = 0

    lucro_liquido = usdt - SALDO_INICIAL
    roi_percentual = (lucro_liquido / SALDO_INICIAL) * 100

    print("\n" + "="*50)
    print("🏆 EXTRATO FINAL DA INTELIGÊNCIA ARTIFICIAL")
    print("="*50)
    print(f"💵 Investimento Inicial:   ${SALDO_INICIAL:,.2f}")
    print(f"💰 Saldo Final Estimado:   ${usdt:,.2f}")
    print(f"📈 Lucro Líquido:          ${lucro_liquido:,.2f}")
    print(f"🚀 Retorno (ROI):          {roi_percentual:.2f}%")
    print(f"🔄 Total de Operações:      {total_trades}")
    print("="*50)
    
    if roi_percentual > 0:
        print("\nResultado: A IA foi ALTAMENTE LUCRATIVA no período testado! ✅")
    else:
        print("\nResultado: A IA não conseguiu superar as taxas ou o mercado. ❌")

if __name__ == "__main__":
    auditoria_ia()
