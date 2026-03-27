import pandas as pd
import os

# =================================================================
# SCRIPT DE FUSÃO MULTIMOEDAS (PREPARATIVO PARA VALIDAÇÃO)
# =================================================================

def fundir_dados():
    pasta_dados = 'dados/'
    # Carregamos o arquivo mestre do BTC que contém o sentimento (NLP)
    df_sentimento = pd.read_csv('dados/btc_com_sentimento.csv')
    df_sentimento['timestamp'] = pd.to_datetime(df_sentimento['timestamp'])
    sentimento_global = df_sentimento[['timestamp', 'Score_Sentimento']].copy()

    # Moedas que vamos processar (As 10 principais + extras)
    moedas = ['eth', 'sol', 'xrp', 'ada', 'avax', 'doge', 'dot', 'link', 'ltc', 'near']

    print("🧩 Iniciando Fusão de Dados Multimoeda (Sentiment Proxy)...")

    for moeda in moedas:
        arquivo_preco = f'dados/{moeda}_usdt_24m.csv'
        if not os.path.exists(arquivo_preco):
            print(f"⚠️  Aviso: Arquivo de preço para {moeda} não encontrado. Pulando.")
            continue
        
        df_preco = pd.read_csv(arquivo_preco)
        # Ajuste: A coluna nas altcoins já se chama 'timestamp' e já está em texto legível
        df_preco['timestamp'] = pd.to_datetime(df_preco['timestamp'])
        
        # Mesclar o sentimento do Bitcoin neste ativo (Padrão de Correlacionamento)
        df_final = pd.merge_asof(
            df_preco.sort_values('timestamp'), 
            sentimento_global.sort_values('timestamp'), 
            on='timestamp', 
            direction='backward'
        )
        
        df_final.fillna(0.0, inplace=True)
        
        # Salvar para validação massiva
        output_path = f'dados/{moeda}_com_sentimento.csv'
        df_final.to_csv(output_path, index=False)
        print(f"✅ Arquivo Completo Gerado: {output_path}")

if __name__ == "__main__":
    fundir_dados()
