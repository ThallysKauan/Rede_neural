import requests
import pandas as pd
from textblob import TextBlob
from datetime import datetime
import os
import time

# API Key da NewsData.io (Fornecida pelo usuário)
API_KEY = "pub_50489aa3cd1a491183533efaf3a18477"
TOKEN_PESQUISA = "bitcoin" 

url_base = f"https://newsdata.io/api/1/news?apikey={API_KEY}&q={TOKEN_PESQUISA}&language=en"

def main():
    print(f"Buscando histórico PROFUNDO de notícias sobre {TOKEN_PESQUISA}...")
    
    noticias_processadas = []
    
    # Vamos buscar 10 páginas (100 notícias). A conta free permite 200 requests/dia.
    # Você pode aumentar isso para 50 páginas se quiser criar um modelo bem mais denso.
    paginas_para_buscar = 10 
    proxima_pagina = ""

    try:
        for i in range(paginas_para_buscar):
            print(f"Buscando Lote {i+1}...")
            
            url_atual = url_base
            if proxima_pagina:
                url_atual += f"&page={proxima_pagina}"
                
            resposta = requests.get(url_atual)
            dados = resposta.json()

            if dados.get('status') == 'success':
                resultados = dados.get('results', [])
                if not resultados:
                    print("Nenhuma notícia nova encontrada.")
                    break

                for artigo in resultados:
                    titulo = artigo.get('title', '')
                    descricao = artigo.get('description', '') or ''
                    texto_completo = f"{titulo}. {descricao}"
                    data_pub = artigo.get('pubDate', '')

                    # Análise de Sentimento (O "cérebro" da IA NLP)
                    analise = TextBlob(texto_completo)
                    score_sentimento = analise.sentiment.polarity
                    
                    classificacao = "Neutro"
                    if score_sentimento > 0.1: classificacao = "Positivo (Comprar)"
                    if score_sentimento < -0.1: classificacao = "Negativo (Vender)"

                    noticias_processadas.append({
                        'Data_Publicacao': data_pub,
                        'Manchete': titulo,
                        'Score_IA': round(score_sentimento, 4),
                        'Classificacao': classificacao
                    })

                # Pega a chave da próxima página
                proxima_pagina = dados.get('nextPage')
                if not proxima_pagina:
                    print("Chegamos ao fim do histórico gratuito disponível no momento.")
                    break
                    
                # Pequena pausa para a API não nos bloquear por spam (Rate Limit)
                time.sleep(1)

            else:
                print(f"Aviso da API na página {i+1}:", dados)
                break

        # Define o caminho de salvamento para a pasta dados/
        pasta_dados = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dados")
        os.makedirs(pasta_dados, exist_ok=True)
        
        caminho_arquivo = os.path.join(pasta_dados, f"noticias_{TOKEN_PESQUISA}_treino.csv")
        df = pd.DataFrame(noticias_processadas)
        df.to_csv(caminho_arquivo, index=False)
        
        print(f"\n✅ SUCESSO! Coleta finalizada com {len(df)} notícias inseridas no banco.")
        print(f"✅ Arquivo atualizado em: {caminho_arquivo}\n")
        print("MAMOSTRA DO NOVO BANCO DE DADOS:")
        print(df.tail()) # Mostra as últimas linhas pra provar que tem mais data

    except Exception as e:
        print("Ocorreu um erro fatal ao rodar os Lotes da API NewsData:", e)

if __name__ == "__main__":
    main()
