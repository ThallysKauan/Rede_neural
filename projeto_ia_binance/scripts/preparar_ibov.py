import pandas as pd
import numpy as np
import os

def adicionar_indicadores():
    print("🇧🇷 ENRIQUECENDO DADOS DO IBOV COM INDICADORES DE ELITE...")
    
    df = pd.read_csv('dados/ibov_24m.csv')
    
    # Médias Móveis Exponenciais (Tendência Curta e Longa)
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # RSI (Força Relativa)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD (Trend Momentum)
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # CCI (Commodity Channel Index - Para Ciclos)
    tp = (df['high'] + df['low'] + df['close']) / 3
    df['cci'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
    
    # Stochastic Oscillator (%K e %D)
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    
    # ATR (Average True Range - Volatilidade para Stop-Loss)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    # Alvo: % de variação em 12h (Swing Rápido)
    df['target_swing'] = df['close'].shift(-12) / df['close'] - 1
    
    df = df.dropna()
    
    os.makedirs('dados', exist_ok=True)
    df.to_csv('dados/ibov_processado.csv', index=False)
    print("✅ DADOS PROCESSADOS COM INDICADORES DE ALTA ASSERTIVIDADE.")
    print(df[['close', 'rsi', 'macd', 'stoch_k', 'atr']].tail())

if __name__ == "__main__":
    adicionar_indicadores()
