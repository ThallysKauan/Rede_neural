import pandas as pd
import os

def main():
    print("Iniciando a Sincronização Temporal (Preços + Notícias)...")
    
    pasta_raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pasta_dados = os.path.join(pasta_raiz, "dados")
    
    arquivo_precos = os.path.join(pasta_dados, "btc_usdt_24m.csv")
    arquivo_noticias = os.path.join(pasta_dados, "noticias_historicas_bitcoin.csv")
    arquivo_saida = os.path.join(pasta_dados, "btc_com_sentimento.csv")
    
    # 1. Carregar Preços
    print("Carregando base de preços do Bitcoin...")
    try:
        df_precos = pd.read_csv(arquivo_precos)
        # Garantir tipo datetime
        df_precos['timestamp'] = pd.to_datetime(df_precos['timestamp'])
    except Exception as e:
        print(f"Erro fatal: Não achou arquivo de preço. Detalhes: {e}")
        return

    # 2. Carregar Notícias (A nova Feature)
    print("Carregando base de Sentimentos (NLP)...")
    try:
        df_noticias = pd.read_csv(arquivo_noticias)
        df_noticias['Data_Publicacao'] = pd.to_datetime(df_noticias['Data_Publicacao'])
    except Exception as e:
        print(f"Erro: Rode o coletar_noticias.py primeiro. Detalhes: {e}")
        return

    # 3. Arredondamento Temporal SEGURO (Prevenção de Data Leakage)
    print("Aplicando Algoritmo Prevenção de Look-Ahead Bias...")
    # .dt.ceil('h') pega 14:01 ou 14:59 e empurra para a janela de 15:00 cravado
    df_noticias['Hora_Alvo'] = df_noticias['Data_Publicacao'].dt.ceil('h')
    
    # 4. Agrupar notícias concorrentes na mesma hora
    print("Calculando Média Emocional da hora...")
    sentimento_por_hora = df_noticias.groupby('Hora_Alvo')['Score_IA'].mean().reset_index()
    # Renomear pra encaixar perfeitamente como LEGO com o CSV da Binance ('timestamp')
    sentimento_por_hora.rename(columns={'Hora_Alvo': 'timestamp', 'Score_IA': 'Score_Sentimento'}, inplace=True)
    
    # 5. FUNDIR OS DOIS MUNDOS
    print("Unindo a Linha do Tempo...")
    # 'left' garante que todos os fechamentos da Binance existam, e o NLP entra onde achar a hora
    df_final = pd.merge(df_precos, sentimento_por_hora, on='timestamp', how='left')
    
    # Remover campos "vazios". Se o mundo silenciar naquelas horas, o Humor é 0 (Neutro)
    df_final['Score_Sentimento'] = df_final['Score_Sentimento'].fillna(0.0)
    
    # 6. Exportação para a IA
    df_final.to_csv(arquivo_saida, index=False)
    
    noticias_impactantes = df_final[df_final['Score_Sentimento'] != 0.0]
    
    print("\n✅ MESCLAGEM CONCLUÍDA ABSOLUTAMENTE COM SUCESSO!")
    print(f"📊 Arquivo oficial da IA gerado em: {arquivo_saida}")
    print(f"🔗 A Neural Network vai treinar em {len(df_final)} horas, sendo {len(noticias_impactantes)} destas guiadas pela nova métrica de sentimento.")
    print("\nEXEMPLO DOS DADOS ALINHADOS (Apenas horas com notícias atreladas):")
    print(noticias_impactantes[['timestamp', 'close', 'Score_Sentimento']].head(10))

if __name__ == "__main__":
    main()
