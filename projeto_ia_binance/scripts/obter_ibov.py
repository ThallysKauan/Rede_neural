import yfinance as yf
import pandas as pd
import os

def baixar_ibov():
    print("🇧🇷 BAIXANDO DADOS DO IBOV (^BVSP) - ÚLTIMOS 24 MESES...")
    
    # Ticker do Índice Bovespa
    ibov = yf.Ticker("^BVSP")
    
    # Baixar dados horários (máximo permitido pelo Yahoo é 730 dias)
    # 24 meses aprox 730 dias.
    df = ibov.history(period="2y", interval="1h")
    
    if df.empty:
        print("Erro: Não foi possível obter dados do IBOV.")
        return

    # Limpeza básica
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    
    # No IBOV não temos o 'Score_Sentimento' nativo no Yahoo, 
    # vamos criar uma coluna neutra inicial para o treinamento base 
    # (depois a IA aprenderá com as notícias reais na live).
    df['Score_Sentimento'] = 0.5 
    
    # Salvar
    os.makedirs('dados', exist_ok=True)
    caminho = 'dados/ibov_24m.csv'
    df.to_csv(caminho, index=False)
    
    print(f"✅ DADOS SALVOS EM: {caminho}")
    print(f"📊 Total de registros: {len(df)}")
    print(df.head())

if __name__ == "__main__":
    baixar_ibov()
