import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt
from datetime import datetime

# =================================================================
# 📅 EXTRATO MENSAL COM TRAVA DE SEGURANÇA (STOP LOSS)
# Mostra quanto você ganharia em cada mês específico.
# =================================================================

def auditoria_mensal(moeda='btc'):
    if moeda == 'xrp':
        modelo_path = 'modelos/xrp_modelo_hibrido.h5'
        campo_sentimento = 'Score_Sentimento'
    else:
        modelo_path = 'modelos/modelo_nlp_v24h_btc_com_sentimento.h5'
        campo_sentimento = 'Score_Sentimento' # Agora usamos o mesmo padrão
        
    dados_path = f'dados/{moeda}_com_sentimento.csv'
    
    if not os.path.exists(modelo_path) or not os.path.exists(dados_path):
        print(f"Erro: Arquivos para {moeda} não encontrados ({modelo_path} ou {dados_path}).")
        return

    # 1. CARREGAR DISPOSITIVOS
    model = load_model(modelo_path, compile=False)
    df = pd.read_csv(dados_path)
    df.fillna(0.0, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Preparar Scalers (Base Próprio Ativo para XRP ser preciso)
    scaler_f = MinMaxScaler(feature_range=(0, 1))
    scaler_f.fit(df[['close', campo_sentimento]].values)
    scaler_c = MinMaxScaler(feature_range=(0, 1))
    scaler_c.fit(df[['close']].values)

    janela = 60
    STOP_LOSS = 0.055 # 5.5% de Trava (Sólido para Crypto)
    TAXA = 0.001
    GATILHO_IA = 2.0 # 2% de Filtro de Ruído
    
    # Adicionar coluna de Mês/Ano para agrupar
    df['mes_ano'] = df['timestamp'].dt.to_period('M')
    meses_unicos = df['mes_ano'].unique()
    
    banca_total = 1000.0
    relatorio = []

    print(f"\n" + "="*60)
    print(f"📊 EXTRATO MENSAL DETALHADO - Ativo: {moeda.upper()}")
    print(f"🔒 SEGURANÇA ATIVADA: Stop-Loss de {STOP_LOSS*100}%")
    print("="*60)

    for periodo in meses_unicos:
        df_mes = df[df['mes_ano'] == periodo]
        if len(df_mes) < janela + 24: continue
        
        banca_inicio_mes = banca_total
        usdt = banca_total
        btc_pos = 0.0
        posicionado = False
        preco_entrada = 0.0
        trades_mes = 0
        stops_acionados = 0
        
        # Simular o mês hora a hora
        for i in range(df_mes.index[0] + janela, df_mes.index[-1] - 24):
            # 1. Checar Stop Loss se estiver posicionado
            preco_atual = df.iloc[i]['close']
            if posicionado:
                queda = (preco_atual - preco_entrada) / preco_entrada
                if queda <= -STOP_LOSS:
                    usdt = (btc_pos * preco_atual) * (1 - TAXA)
                    btc_pos = 0
                    posicionado = False
                    stops_acionados += 1
                    trades_mes += 1
                    continue

            # 2. IA Decide
            X = np.reshape(scaler_f.transform(df.iloc[i-janela:i][['close', campo_sentimento]].values), (1, janela, 2))
            pred = model.predict(X, verbose=0)
            p_previsto = scaler_c.inverse_transform(pred)[0][0]
            
            diff = ((p_previsto - preco_atual) / preco_atual) * 100

            # Gatilho Sólido: Compra com 2% de certeza
            if diff > GATILHO_IA and not posicionado:
                btc_pos = (usdt * (1 - TAXA)) / preco_atual
                usdt = 0
                preco_entrada = preco_atual
                posicionado = True
                trades_mes += 1
            elif diff < -1.0 and posicionado:
                usdt = (btc_pos * preco_atual) * (1 - TAXA)
                btc_pos = 0
                posicionado = False
                trades_mes += 1

        # Fechar mês
        if posicionado:
            usdt = (btc_pos * df_mes.iloc[-1]['close']) * (1 - TAXA)
            btc_pos = 0
            posicionado = False
        
        banca_total = usdt
        lucro_mes = banca_total - banca_inicio_mes
        roi_mes = (lucro_mes / banca_inicio_mes) * 100
        
        relatorio.append({
            "Mês": str(periodo),
            "ROI %": f"{roi_mes:+.2f}%",
            "Lucro $": f"${lucro_mes:,.2f}",
            "Trades": trades_mes,
            "Stops 🛡️": stops_acionados,
            "Saldo Final": f"${banca_total:,.2f}"
        })
        print(f"✅ Processado: {periodo} | ROI: {roi_mes:+.2f}%")

    # Exibir Tabela
    print("\n" + "="*80)
    print(f"🏆 RESULTADO CONSOLIDADO POR MÊS ({moeda.upper()})")
    print("="*80)
    df_rel = pd.DataFrame(relatorio)
    print(df_rel.to_string(index=False))
    print("="*80)
    
    total_roi = ((banca_total - 1000.0) / 1000.0) * 100
    print(f"\n🚀 ROI TOTAL ACUMULADO COM TRAVA DE SEGURANÇA: {total_roi:.2f}%")

    # === GERAÇÃO DO GRÁFICO ===
    df_plot = df_rel.copy()
    df_plot['ROI_Num'] = df_plot['ROI %'].str.replace('%', '').astype(float)
    df_plot['Saldo_Num'] = df_plot['Saldo Final'].str.replace('$', '').str.replace(',', '').astype(float)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # ROI em Barras
    cores = ['red' if x < 0 else 'green' for x in df_plot['ROI_Num']]
    ax1.bar(df_plot['Mês'], df_plot['ROI_Num'], color=cores, alpha=0.3, label='ROI Mensal (%)')
    ax1.set_ylabel('ROI (%)', color='green')
    ax1.tick_params(axis='y', labelcolor='green')

    # Saldo em Linha
    ax2 = ax1.twinx()
    ax2.plot(df_plot['Mês'], df_plot['Saldo_Num'], color='blue', marker='o', linewidth=2, label='Curva de Capital ($)')
    ax2.set_ylabel('Saldo Final ($)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    plt.title(f'Performance Mensal (IA + Stop Loss) - {moeda.upper()}')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Salvar Gráfico
    os.makedirs('plots', exist_ok=True)
    caminho_grafico = f'plots/relatorio_mensal_{moeda}.png'
    plt.savefig(caminho_grafico)
    print(f"\n📊 GRÁFICO SALVO COM SUCESSO: {caminho_grafico}")
    plt.close()

if __name__ == "__main__":
    # RODAR AUDITORIA DO XRP (O MAIS SÓLIDO)
    auditoria_mensal('xrp')
