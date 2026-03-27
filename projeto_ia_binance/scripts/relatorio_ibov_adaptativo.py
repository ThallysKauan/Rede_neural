import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt
from datetime import datetime

# =================================================================
# 📅 AUDITORIA IBOV V4.1 - MODO CONSISTÊNCIA E CONFIANÇA
# O robô só opera se a rede neural provar que está acertando.
# =================================================================

def auditoria_ibov_adaptativa():
    print("🇧🇷 INICIANDO LABORATÓRIO DE ESTRATÉGIAS IBOV (MODO CONFIANÇA)...")
    
    caminho_dados = 'dados/ibov_processado.csv'
    if not os.path.exists(caminho_dados):
        print(f"❌ Erro: Arquivo {caminho_dados} não encontrado.")
        return

    df = pd.read_csv(caminho_dados)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.sort_values('datetime')

    features = ['close', 'ema9', 'ema21', 'rsi', 'macd', 'stoch_k', 'atr']
    
    model_path = 'modelos/ibov_swing_modelo.h5'
    if not os.path.exists(model_path):
        print("❌ Erro: Modelo base não encontrado.")
        return
        
    model = load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='mse')
    
    scaler = MinMaxScaler()
    scaler.fit(df[features].values)
    
    scaler_p = MinMaxScaler()
    scaler_p.fit(df[['close']].values)

    # --- CONFIGURAÇÕES ---
    ESTRATEGIAS = [
        {"nome": "Conservadora", "gatilho": 1.2, "rsi_max": 65, "stop": 0.01},
        {"nome": "Moderada", "gatilho": 0.8, "rsi_max": 75, "stop": 0.02},
        {"nome": "Scalping", "gatilho": 0.5, "rsi_max": 80, "stop": 0.008},
    ]
    
    ALAVANCAGEM = 3
    TAXA = 0.0003
    CIRCUIT_BREAKER_DIARIO = 0.025
    MIN_CONFIANCA = 0.55 # 55% de acerto recente nas previsões (últimas 24h)
    
    janela = 60
    banca = 1000.0
    relatorio = []
    
    df['mes_ano'] = df['datetime'].dt.to_period('M')
    meses_unicos = df['mes_ano'].unique()
    
    # Variáveis de Estado
    estrat_ativa = ESTRATEGIAS[0]
    posicao = None
    preco_entrada = 0.0
    stop_dinamico = 0.0
    
    memoria_acertos = [] # Guarda 1 (acertou direção) ou 0 (errou)

    for idx_mes, periodo in enumerate(meses_unicos):
        df_mes = df[df['mes_ano'] == periodo]
        if len(df_mes) < janela: continue
        
        banca_inicio_mes = banca
        saldo_dia_inicio = banca
        data_atual = None
        
        print(f"🎯 [LAB] Período {periodo} | Perfil Ativo: {estrat_ativa['nome']}")

        for i in range(df_mes.index[0], df_mes.index[-1]):
            if i < janela: continue
            
            linha = df.iloc[i]
            preco_atual = linha['close']
            hoje = linha['datetime'].date()
            
            # Circuit Breaker
            if data_atual != hoje:
                data_atual = hoje
                saldo_dia_inicio = banca
                
            perda_dia = (banca - saldo_dia_inicio) / saldo_dia_inicio if saldo_dia_inicio > 0 else 0
            if perda_dia <= -CIRCUIT_BREAKER_DIARIO:
                if posicao:
                    renda = (preco_atual - preco_entrada) / preco_entrada if posicao == 'long' else (preco_entrada - preco_atual) / preco_entrada
                    banca += (renda * ALAVANCAGEM * banca) - (banca * TAXA)
                    posicao = None
                continue

            # --- GESTÃO DE RISCO ---
            if posicao:
                rend = (preco_atual - preco_entrada) / preco_entrada if posicao == 'long' else (preco_entrada - preco_atual) / preco_entrada
                if rend > 0.005:
                    if posicao == 'long': stop_dinamico = max(stop_dinamico, preco_entrada)
                    else: stop_dinamico = min(stop_dinamico, preco_entrada)

                if rend <= -estrat_ativa['stop'] or (posicao == 'long' and preco_atual < stop_dinamico) or (posicao == 'short' and preco_atual > stop_dinamico):
                    banca += (rend * ALAVANCAGEM * banca) - (banca * TAXA)
                    posicao = None
                    continue

            # --- DECISÃO DA IA ---
            janelas_dados = df.iloc[i-janela:i][features].values
            X_input = np.reshape(scaler.transform(janelas_dados), (1, janela, len(features)))
            
            pred = model.predict(X_input, verbose=0)
            p_previsto = scaler_p.inverse_transform(pred)[0][0]
            
            diff_pct = ((p_previsto - preco_atual) / preco_atual) * 100
            rsi = linha['rsi']
            
            # --- FILTRO DE CONFIANÇA RECENTE ---
            # Verifica se a IA acertou o movimento nas últimas 24h (se disponível o dado real 12h depois)
            # (Simplificado: Olhamos se a previsão de 12h atrás acertou a direção atual)
            if i > 12:
                jan_passada = df.iloc[i-12-janela:i-12][features].values
                X_passado = np.reshape(scaler.transform(jan_passada), (1, janela, len(features)))
                pred_passada = model.predict(X_passado, verbose=0)
                p_prev_passado = scaler_p.inverse_transform(pred_passada)[0][0]
                
                direcao_prevista = 1 if p_prev_passado > df.iloc[i-12]['close'] else -1
                direcao_real = 1 if preco_atual > df.iloc[i-12]['close'] else -1
                
                memoria_acertos.append(1 if direcao_prevista == direcao_real else 0)
                if len(memoria_acertos) > 24: memoria_acertos.pop(0)

            taxa_acerto_recente = sum(memoria_acertos) / len(memoria_acertos) if memoria_acertos else 1.0
            
            # GATILHOS (Só opera SE tiver confiança alta)
            if not posicao and taxa_acerto_recente >= MIN_CONFIANCA:
                if diff_pct > estrat_ativa['gatilho'] and rsi < estrat_ativa['rsi_max']:
                    posicao = 'long'
                    preco_entrada = preco_atual
                    stop_dinamico = preco_entrada * (1 - estrat_ativa['stop'])
                elif diff_pct < -estrat_ativa['gatilho'] and rsi > (100 - estrat_ativa['rsi_max']):
                    posicao = 'short'
                    preco_entrada = preco_atual
                    stop_dinamico = preco_entrada * (1 + estrat_ativa['stop'])
            elif posicao:
                # Saída por Alvo ou Reversão
                if (posicao == 'long' and diff_pct < 0.2) or (posicao == 'short' and diff_pct > -0.2):
                    lucro = (preco_atual - preco_entrada) / preco_entrada if posicao == 'long' else (preco_entrada - preco_atual) / preco_entrada
                    banca += (lucro * ALAVANCAGEM * banca) - (banca * TAXA)
                    posicao = None

        roi_mes = ((banca - banca_inicio_mes) / banca_inicio_mes) * 100
        relatorio.append({
            "Mês": str(periodo),
            "Saldo": f"R$ {banca:,.2f}",
            "ROI %": f"{roi_mes:+.2f}%",
            "Confiança": f"{taxa_acerto_recente*100:,.0f}%"
        })
        print(f"✅ {periodo} finalizado | ROI: {roi_mes:+.2f}% | Memória IA: {taxa_acerto_recente*100:,.0f}%")

    df_rel = pd.DataFrame(relatorio)
    print("\n" + "="*60)
    print("🏆 PERFORMANCE FINAL (IBOV CONSISTÊNCIA v4.1)")
    print(df_rel.to_string(index=False))
    print("="*60)
    
    plt.figure(figsize=(12,6))
    df_rel['ROI_Num'] = df_rel['ROI %'].str.replace('%','').astype(float)
    plt.plot(df_rel['Mês'], df_rel['ROI_Num'].cumsum(), color='#00ff00', marker='o', label='Curva de Capital')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Evolução do Patrimônio - IA com Filtro de Confiança Recente (IBOV)')
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/performance_consistente_ibov.png')
    print("📊 Gráfico final salvo: plots/performance_consistente_ibov.png")

if __name__ == "__main__":
    auditoria_ibov_adaptativa()
