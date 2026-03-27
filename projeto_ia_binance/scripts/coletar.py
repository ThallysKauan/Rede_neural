import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
import os

# =================================================================
# SCRIPT DE COLETA UNIFICADO (v2.0)
# Suporta: Símbolo customizado e Histórico Longo (ex: 24 meses)
# =================================================================

def baixar_historico(simbolo='ETH/USDT', meses=24, tempo_grafico='1h'):
    exchange = ccxt.binance({'enableRateLimit': True})
    
    # Pasta de destino
    if not os.path.exists('dados'):
        os.makedirs('dados')
        
    arquivo_nome = f"dados/{simbolo.replace('/', '_').lower()}_{meses}m.csv"
    
    print(f"--- Iniciando coleta de {simbolo} (Últimos {meses} meses) ---")
    
    # Calcular o tempo de início (since)
    since = exchange.parse8601((datetime.now() - timedelta(days=meses*30)).isoformat())
    
    todos_ohlcv = []
    
    while since < exchange.milliseconds():
        try:
            print(f"Baixando a partir de: {exchange.iso8601(since)}")
            ohlcv = exchange.fetch_ohlcv(simbolo, timeframe=tempo_grafico, since=since, limit=1000)
            
            if not ohlcv:
                break
                
            todos_ohlcv.extend(ohlcv)
            # Atualiza o 'since' para o último timestamp recebido + 1ms
            since = ohlcv[-1][0] + 1 
            
            # Pequeno delay para respeitar o Rate Limit
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            print(f"Erro na coleta: {e}")
            break

    # Consolidar dados
    df = pd.DataFrame(todos_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    df.to_csv(arquivo_nome, index=False)
    print(f"\n--- SUCESSO! ---")
    print(f"Foram coletadas {len(df)} linhas.")
    print(f"Arquivo salvo em: {arquivo_nome}")

if __name__ == "__main__":
    # Exemplo: Baixar 24 meses de ETH (Padrão)
    import sys
    
    simbolo = sys.argv[1] if len(sys.argv) > 1 else 'ETH/USDT'
    meses = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    
    baixar_historico(simbolo, meses)
