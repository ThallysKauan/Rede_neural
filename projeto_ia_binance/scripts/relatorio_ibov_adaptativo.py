import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, clone_model
import os
import matplotlib.pyplot as plt
from datetime import datetime

# =================================================================
# 📅 AUDITORIA IBOV ADAPTATIVA (APRENDE COM O TEMPO)
# Re-treina a rede neural a cada 3 meses para prever mudanças.
# =================================================================

def auditoria_ibov_adaptativa():
    print("🇧🇷 INICIANDO AUDITORIA ADAPTATIVA IBOV (24 MESES)...")
    
    df = pd.read_csv('dados/ibov_processado.csv')
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.sort_values('datetime')

    # Features: close, ma7, ma21, rsi, upper_band, lower_band
    features = ['close', 'ma7', 'ma21', 'rsi', 'upper_band', 'lower_band']
    
    # Modelo Base
    model = load_model('modelos/ibov_swing_modelo.h5', compile=False)
    model.compile(optimizer='adam', loss='mse')
    
    scaler = MinMaxScaler()
    scaler.fit(df[features].values)
    
    # Scaler para desnormalizar apenas o preço (coluna 0)
    scaler_p = MinMaxScaler()
    scaler_p.fit(df[['close']].values)

    janela = 60
    banca_total = 1000.0
    relatorio = []
    
    df['mes_ano'] = df['datetime'].dt.to_period('M')
    meses_unicos = df['mes_ano'].unique()
    
    # Variáveis de Estado
    posicionado = False
    shares = 0.0
    usdt = 1000.0
    preco_entrada = 0.0
    TAXA = 0.0003 
    STOP_LOSS = 0.025 # 2.5% para giros mais curtos e seguros
    GATILHO_COMPRA = 0.7 # 0.7% de previsão já dispara a compra (Mais agressivo)
    
    # Memória para Aprendizado
    memoria_X, memoria_y = [], []

    for idx_mes, periodo in enumerate(meses_unicos):
        df_mes = df[df['mes_ano'] == periodo]
        if len(df_mes) < 24: continue
        
        banca_inicio_mes = usdt if not posicionado else (shares * df_mes.iloc[0]['close'])
        trades_mes = 0
        
        # 🧠 APRENDIZADO ADAPTATIVO: Re-treina com o que aprendeu (Epochs aumentadas para 5)
        if idx_mes > 0 and idx_mes % 3 == 0 and len(memoria_X) > 100:
            print(f"🧠 [ADAPTATIVO] Evoluindo IA...")
            mX = np.array(memoria_X, dtype='float32')
            my = np.array(memoria_y, dtype='float32')
            model.fit(mX, my, epochs=5, batch_size=32, verbose=0)
            memoria_X, memoria_y = [], [] 

        # Simular o mês
        for i in range(df_mes.index[0], df_mes.index[-1]):
            if i < janela: continue
            
            preco_atual = df.iloc[i]['close']
            
            if posicionado:
                queda = (preco_atual - preco_entrada) / preco_entrada
                if queda <= -STOP_LOSS:
                    usdt = (shares * preco_atual) * (1 - TAXA)
                    shares = 0
                    posicionado = False
                    trades_mes += 1
                    continue

            # Decisão da IA
            janela_dados = df.iloc[i-janela:i][features].values
            X = np.reshape(scaler.transform(janela_dados), (1, janela, len(features)))
            
            pred = model.predict(X, verbose=0)
            p_previsto = scaler_p.inverse_transform(pred)[0][0]
            
            diff = ((p_previsto - preco_atual) / preco_atual) * 100
            rsi_atual = df.iloc[i]['rsi']
            
            # Estratégia de "Oportunidade Agressiva"
            if diff > GATILHO_COMPRA and not posicionado and rsi_atual < 75:
                shares = (usdt * (1 - TAXA)) / preco_atual
                usdt = 0
                preco_entrada = preco_atual
                posicionado = True
                trades_mes += 1
            elif (diff < 0.2 or rsi_atual > 80) and posicionado:
                usdt = (shares * preco_atual) * (1 - TAXA)
                shares = 0
                posicionado = False
                trades_mes += 1
            
            # Guardar na memória para o próximo re-treinamento
            # Alvo real: o que aconteceu 12h depois (se disponível)
            if i + 12 < len(df):
                memoria_X.append(X[0])
                p_futuro_scaled = scaler_p.transform([[df.iloc[i+12]['close']]])[0][0]
                memoria_y.append(p_futuro_scaled)

        # Saldo Final do Mês
        banca_final_mes = usdt if not posicionado else (shares * df_mes.iloc[-1]['close'])
        roi_mes = ((banca_final_mes - banca_inicio_mes) / banca_inicio_mes) * 100
        
        relatorio.append({
            "Mês": str(periodo),
            "ROI %": f"{roi_mes:+.2f}%",
            "Saldo": f"${banca_final_mes:,.2f}",
            "Trades": trades_mes
        })
        print(f"✅ Mês {periodo} concluído | ROI: {roi_mes:+.2f}%")

    # Resultados
    df_rel = pd.DataFrame(relatorio)
    print("\n" + "="*60)
    print("🏆 RESULTADO FINAL IBOV ADAPTATIVO")
    print(df_rel.to_string(index=False))
    print("="*60)
    
    # Gráfico
    plt.figure(figsize=(10,5))
    df_rel['ROI_Num'] = df_rel['ROI %'].str.replace('%','').astype(float)
    plt.plot(df_rel['Mês'], df_rel['ROI_Num'].cumsum(), marker='o', color='green', label='ROI Acumulado')
    plt.title('Curva de Capital Adaptativa - IBOV (Aprendizado em Tempo Real)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/relatorio_ibov_adaptativo.png')
    print("📊 Gráfico salvo: plots/relatorio_ibov_adaptativo.png")

if __name__ == "__main__":
    auditoria_ibov_adaptativa()
