import subprocess
import time

# =================================================================
# SCRIPT DE COLETA EM MASSA (10 MOEDAS)
# =================================================================

moedas = [
    'SOL/USDT', 'ADA/USDT', 'LINK/USDT', 'LTC/USDT', 'AVAX/USDT',
    'TRX/USDT', 'DOGE/USDT', 'DOT/USDT', 'NEAR/USDT', 'MATIC/USDT'
]

print(f"Iniciando a coleta de histórica de {len(moedas)} moedas (24 meses cada)...")
print("Isso pode levar alguns minutos.\n")

for moeda in moedas:
    print(f"--- Coletando {moeda} ---")
    try:
        # Chama o nosso script coletar.py individualmente
        subprocess.run(['python', 'scripts/coletar.py', moeda, '24'], check=True)
        time.sleep(2) # Pequena pausa entre moedas
    except Exception as e:
        print(f"Erro ao coletar {moeda}: {e}")

print("\n--- Coleta em massa concluída! Todos os arquivos estão na pasta /dados ---")
