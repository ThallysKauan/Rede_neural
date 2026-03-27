import pandas as pd
import numpy as np
import os

def adicionar_indicadores():
    print("🇧🇷 ENRIQUECENDO DADOS DO IBOV COM INDICADORES DE SWING...")
    
    df = pd.read_csv('dados/ibov_24m.csv')
    
    # Médias Móveis (Tendência)
    df['ma7'] = df['close'].rolling(window=7).mean()
    df['ma21'] = df['close'].rolling(window=21).mean()
    
    # RSI (Força Relativa - Sobrecompra/Sobrevenda)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bandas de Bollinger (Volatilidade)
    df['std'] = df['close'].rolling(window=20).std()
    df['upper_band'] = df['ma21'] + (df['std'] * 2)
    df['lower_band'] = df['ma21'] - (df['std'] * 2)
    
    # Alvo de 'Oportunidade': % de variação nas próximas 24h
    # Isso ensina a IA a buscar onde o preço vai subir mais.
    df['target_swing'] = df['close'].shift(-24) / df['close'] - 1
    
    df = df.dropna()
    
    df.to_csv('dados/ibov_processado.csv', index=False)
    print("✅ DADOS PROCESSADOS: dados/ibov_processado.csv")
    print(df[['close', 'rsi', 'upper_band', 'target_swing']].tail())

if __name__ == "__main__":
    adicionar_indicadores()
