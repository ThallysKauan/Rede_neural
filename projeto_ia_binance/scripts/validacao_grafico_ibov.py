import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# =================================================================
# 📊 TESTE DE FOGO: REAL VS PREVISTO (30 DIAS)
# Prova visual se a IA está prevendo o futuro ou apenas chutando.
# =================================================================

def validar_modelo():
    print("🇧🇷 GERANDO GRÁFICO DE VALIDAÇÃO (ÚLTIMOS 30 DIAS)...")
    
    # 1. Carregar Dados
    caminho_dados = 'dados/ibov_processado.csv'
    if not os.path.exists(caminho_dados):
        print("❌ Erro: Dados não encontrados.")
        return
        
    df = pd.read_csv(caminho_dados)
    
    # 2. Carregar Modelo
    model_path = 'modelos/ibov_swing_modelo.h5'
    if not os.path.exists(model_path):
        print("❌ Erro: Modelo não encontrado.")
        return
    model = load_model(model_path, compile=False)

    # 3. Preparar Features e Scalers
    features = ['close', 'ema9', 'ema21', 'rsi', 'macd', 'stoch_k', 'atr']
    scaler = MinMaxScaler()
    scaler.fit(df[features].values)
    
    scaler_p = MinMaxScaler()
    scaler_p.fit(df[['close']].values)

    # 4. Pegar os últimos 30 dias (aprox 720 horas se for 1h)
    amostras_teste = 720
    if len(df) < amostras_teste + 60:
        amostras_teste = len(df) - 61
        
    df_teste = df.tail(amostras_teste + 60)
    
    reais = []
    previstos = []
    janela = 60

    print(f"   🔎 Analisando {amostras_teste} pontos de dados...")
    
    for i in range(janela, len(df_teste)):
        # O preço REAL 12h no futuro (o que a rede deve acertar)
        # (Nós pegamos o 'close' da linha atual como referência e comparamos com a previsão)
        # Mas para o gráfico, queremos comparar a PREVISÃO para t+12 com o que REALMENTE ocorreu em t+12.
        
        if i + 12 >= len(df_teste): break
        
        preço_real_futuro = df_teste.iloc[i+12]['close']
        
        id_df = df_teste.index[i]
        janela_dados = df_teste.iloc[i-janela:i][features].values
        X_input = np.reshape(scaler.transform(janela_dados), (1, janela, len(features)))
        
        pred = model.predict(X_input, verbose=0)
        p_previsto = scaler_p.inverse_transform(pred)[0][0]
        
        reais.append(preço_real_futuro)
        previstos.append(p_previsto)

    # 5. Plotar
    plt.figure(figsize=(14,7))
    plt.plot(reais, label='Preço Real (12h depois)', color='#2ecc71', alpha=0.6, linewidth=2)
    plt.plot(previstos, label='Previsão da IA', color='#e74c3c', linestyle='--', linewidth=2)
    
    plt.title('Validação da Rede Neural: Real vs Previsto (Foco em Swing de 12h)')
    plt.xlabel('Tempo (Horas)')
    plt.ylabel('Pontos Ibovespa')
    plt.legend()
    plt.grid(alpha=0.2)
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/validacao_real_vs_previsto.png')
    print("✅ GRÁFICO GERADO: plots/validacao_real_vs_previsto.png")
    print("DICA: Se as linhas seguirem a mesma tendência, a rede aprendeu a ler o futuro!")

if __name__ == "__main__":
    validar_modelo()
