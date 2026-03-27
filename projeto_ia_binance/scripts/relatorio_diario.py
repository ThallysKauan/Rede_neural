import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os
import sys
from datetime import datetime, timedelta

# =================================================================
# SCRIPT DE RELATÓRIO "MODO IDIOTA" (SIMPLIFICADO) - v3.0
# Agora aceita número de dias personalizado como argumento.
# =================================================================

def gerar_relatorio(modelo_path='modelos/modelo_v24h_eth_usdt_24m.h5', dados_csv='dados/eth_usdt_24m.csv', dias=7):
    if not os.path.exists(modelo_path):
        print(f"Erro: Modelo '{modelo_path}' não encontrado.")
        return
    if not os.path.exists(dados_csv):
        print(f"Erro: Dados '{dados_csv}' não encontrados.")
        return

    # 1. Carregar Dados
    df = pd.read_csv(dados_csv)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Pegar o modelo
    model = load_model(modelo_path)
    
    # Precisamos de N dias + 60h de buffer
    horas_totais = ((dias + 1) * 24) + 60
    df_recente = df.tail(horas_totais).copy()
    
    dados = df_recente['close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df['close'].values.reshape(-1, 1))
    dados_norm = scaler.transform(dados)

    janela = 60
    
    print("\n" + "="*65)
    print(f"      RELATÓRIO DE {dias} DIAS DA IA (MODO SIMPLIFICADO)")
    print("      O que a IA disse 24h atrás vs O que aconteceu hoje")
    print("="*65)
    print(f"{'DATA':<10} | {'IA DISSE':<10} | {'MERCADO FEZ':<12} | {'RESULTADO'}")
    print("-" * 65)

    total_len = len(dados_norm)
    # O último índice possível para i é total_len - 25
    indices_teste = []
    curr = total_len - 25
    for _ in range(dias):
        if curr < janela: break
        indices_teste.append(curr)
        curr -= 24
    indices_teste.reverse()

    acertos = 0
    total = 0

    for i in indices_teste:
        X_input = dados_norm[i-janela:i, 0].reshape(1, janela, 1)
        
        # Previsão da IA
        pred_norm = model.predict(X_input, verbose=0)
        pred_preco = scaler.inverse_transform(pred_norm)[0][0]
        
        # Preço no momento da previsão
        preco_no_momento = dados[i][0]
        preco_real_chegou = dados[i + 24][0]
        
        data_analise = df_recente.iloc[i+24]['timestamp'].strftime('%d/%m')
        
        # Lógica de Direção
        ia_direcao = "SUBIR" if pred_preco > preco_no_momento else "DESCER"
        mercado_direcao = "SUBIU" if preco_real_chegou > preco_no_momento else "DESCEU"
        
        if (ia_direcao == "SUBIR" and mercado_direcao == "SUBIU") or \
           (ia_direcao == "DESCER" and mercado_direcao == "DESCEU"):
            resultado = "ACERTOU!"
            emoji = "[ OK ]"
            acertos += 1
        else:
            resultado = "ERROU..."
            emoji = "[ XX ]"
            
        total += 1
        print(f"{data_analise:<10} | {ia_direcao:<10} | {mercado_direcao:<12} | {emoji} {resultado}")

    print("-" * 65)
    taxa = (acertos / total) * 100
    print(f"RESUMO: Em {total} dias, a IA acertou a direção {acertos} vezes ({taxa:.1f}%).")
    print("="*65)

if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else 'modelos/modelo_v24h_eth_usdt_24m.h5'
    d_csv = sys.argv[2] if len(sys.argv) > 2 else 'dados/eth_usdt_24m.csv'
    n_dias = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    gerar_relatorio(m, d_csv, n_dias)
