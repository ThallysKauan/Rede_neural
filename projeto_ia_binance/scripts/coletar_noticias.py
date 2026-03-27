import requests
import pandas as pd
from textblob import TextBlob
from datetime import datetime
import os
import time

# API Key da NewsData.io (Fornecida pelo usuário)
API_KEY = "pub_50489aa3cd1a491183533efaf3a18477"
# Termos focados no mercado brasileiro
TOKEN_PESQUISA = "ibovespa OR b3 OR bovespa OR economia brasil" 

# Configurado para Brasil (PT)
url_base = f"https://newsdata.io/api/1/news?apikey={API_KEY}&q={TOKEN_PESQUISA}&language=pt&country=br"

def main():
    print(f"🇧🇷 BUSCANDO NOTÍCIAS DO MERCADO BRASILEIRO: {TOKEN_PESQUISA}...")
    
    noticias_processadas = []
    paginas_para_buscar = 5 # Reduzido para focar nas mais recentes e assertivas
    proxima_pagina = ""

    try:
        for i in range(paginas_para_buscar):
            print(f"   📡 Lote {i+1}...")
            
            url_atual = url_base
            if proxima_pagina:
                url_atual += f"&page={proxima_pagina}"
                
            resposta = requests.get(url_atual)
            dados = resposta.json()

            if dados.get('status') == 'success':
                resultados = dados.get('results', [])
                for artigo in resultados:
                    titulo = artigo.get('title', '')
                    descricao = artigo.get('description', '') or ''
                    texto_completo = f"{titulo}. {descricao}"
                    data_pub = artigo.get('pubDate', '')

                    # Tradução Simples de Sentimento (TextBlob é melhor em EN, mas dá uma base em PT)
                    # Para algo profissional, usaríamos um modelo específico de PT-BR.
                    analise = TextBlob(texto_completo)
                    score_sentimento = analise.sentiment.polarity
                    
                    classificacao = "Neutro"
                    if score_sentimento > 0.05: classificacao = "Positivo"
                    if score_sentimento < -0.05: classificacao = "Negativo"

                    noticias_processadas.append({
                        'Data_Publicacao': data_pub,
                        'Manchete': titulo,
                        'Score_IA': round(score_sentimento, 4),
                        'Classificacao': classificacao
                    })

                proxima_pagina = dados.get('nextPage')
                if not proxima_pagina: break
                time.sleep(1)
            else:
                print(f"   ⚠️ Aviso da API: {dados.get('results', {}).get('message', 'Erro desconhecido')}")
                break

        # Salvar na pasta dados
        caminho_arquivo = 'dados/noticias_ibov_treino.csv'
        os.makedirs('dados', exist_ok=True)
        
        df = pd.DataFrame(noticias_processadas)
        if not df.empty:
            df.to_csv(caminho_arquivo, index=False)
            print(f"\n✅ SUCESSO! {len(df)} notícias brasileiras salvas em: {caminho_arquivo}")
        else:
            print("\n❌ Nenhuma notícia encontrada para os critérios.")

    except Exception as e:
        print("Erro na coleta de notícias BR:", e)

if __name__ == "__main__":
    main()
