import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os

# =================================================================
# 🛡️ AUDITORIA MASSIVA (10+ MOEDAS) COM CÁLCULO DE DRAWDOWN
# =================================================================

def calcular_drawdown(patrimonios):
    """Calcula a queda máxima (pico ao vale) da carteira"""
    patrimonios = np.array(patrimonios)
    max_acumulado = np.maximum.accumulate(patrimonios)
    quedas = (patrimonios - max_acumulado) / max_acumulado
    return np.min(quedas) * 100 # Em percentual

def auditar_moeda(modelo, moeda, scaler_f, scaler_c):
    dados_path = f'dados/{moeda}_com_sentimento.csv'
    if not os.path.exists(dados_path): return None
    
    df = pd.read_csv(dados_path)
    df.fillna(0.0, inplace=True)
    
    janela = 60
    dados_norm = scaler_f.transform(df[['close', 'Score_Sentimento']].values)
    
    # Simulação Financeira
    saldo = 1000.0
    btc_pos = 0.0
    taxa = 0.001
    historico_patrimonio = [saldo]
    trades = 0
    posicionado = False
    
    # Processamos em saltos de 24h para agilizar a validação massiva
    for i in range(janela, len(df) - 24, 24):
        X = np.reshape(dados_norm[i-janela:i, :], (1, janela, 2))
        pred = modelo.predict(X, verbose=0)
        preco_futuro = scaler_c.inverse_transform(pred)[0][0]
        
        preco_atual = df.iloc[i]['close']
        diff = ((preco_futuro - preco_atual) / preco_atual) * 100
        
        if diff > 1.0 and not posicionado:
            btc_pos = (saldo * (1 - taxa)) / preco_atual
            saldo = 0
            posicionado = True
            trades += 1
        elif diff < -1.0 and posicionado:
            saldo = (btc_pos * preco_atual) * (1 - taxa)
            btc_pos = 0
            posicionado = False
            trades += 1
            
        # Registrar patrimônio atual (USDT + valor do BTC se tiver)
        valor_agora = saldo + (btc_pos * preco_atual)
        historico_patrimonio.append(valor_agora)

    # Fechar posição final
    if posicionado:
        saldo = (btc_pos * df.iloc[-1]['close']) * (1 - taxa)
        historico_patrimonio.append(saldo)

    roi = ((saldo - 1000.0) / 1000.0) * 100
    drawdown = calcular_drawdown(historico_patrimonio)
    
    return {"ROI": roi, "Drawdown": drawdown, "Trades": trades, "Final": saldo}

def main():
    modelo_path = 'modelos/modelo_nlp_v24h_btc_com_sentimento.h5'
    model = load_model(modelo_path)
    
    # Preparar Scalers baseados no BTC (Linguagem que a IA conhece)
    df_ref = pd.read_csv('dados/btc_com_sentimento.csv')
    scaler_f = MinMaxScaler(feature_range=(0, 1))
    scaler_f.fit(df_ref[['close', 'Score_Sentimento']].values)
    scaler_c = MinMaxScaler(feature_range=(0, 1))
    scaler_c.fit(df_ref[['close']].values)

    moedas = ['eth', 'sol', 'xrp', 'ada', 'avax', 'doge', 'dot', 'link', 'ltc', 'near']
    resultados = []

    print("\n" + "="*60)
    print("🛡️  BATERIA DE TESTES DE ELITE - 10 CRIPTOMOEDAS")
    print("="*60)

    for m in moedas:
        print(f"🧐 Analisando {m.upper()}...")
        res = auditar_moeda(model, m, scaler_f, scaler_c)
        if res:
            res['Moeda'] = m.upper()
            resultados.append(res)

    # Exibir Tabela Final
    df_res = pd.DataFrame(resultados)
    print("\n📊 TABELA DE COMPROVAÇÃO DE LUCRATIVIDADE:")
    print(df_res[['Moeda', 'ROI', 'Drawdown', 'Trades', 'Final']].to_string(index=False))
    
    media_roi = df_res['ROI'].mean()
    media_dd = df_res['Drawdown'].mean()
    
    print("\n" + "="*60)
    print(f"📉 PERFORMANCE MÉDIA DO GRUPO: {media_roi:.2f}% ROI")
    print(f"⚠️ RISCO MÉDIO (DRAWDOWN): {media_dd:.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()
