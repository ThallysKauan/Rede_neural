import pandas as pd
from textblob import TextBlob
import os
import time

try:
    from gnews import GNews
except ImportError:
    print("Aviso: Falta instalar a biblioteca gnews. Rode: pip install gnews")
    exit()

def main():
    print("====================================================")
    print("🕷️  Robô Scraper Autônomo - Módulo Histórico")
    print("💰  Sem restrição de API limitadora de pagamento")
    print("====================================================")
    
    TOKEN_PESQUISA = "bitcoin"
    
    # DEFINIÇÃO DO RECUO HISTÓRICO (Os 24 Meses Completos)
    anos = [2024, 2025, 2026]
    
    noticias_processadas = []

    print(f"\nAlvo detectado: {TOKEN_PESQUISA.upper()}. Iniciando varredura profunda no Google para 24 meses...")

    for ano in anos:
        for mes in range(1, 13):
            # Prevenir busca no futuro de 2026
            if ano == 2026 and mes > 3:
                break
                
            # O histórico de preços começou em Abril de 2024
            if ano == 2024 and mes < 4:
                continue

            print(f"🔎 Minerando arquivos encriptados do Google [{mes:02d}/{ano}] ...")
            
            # GNews permite definir a data de início e fim. Enganamos o banco de dados simulando buscas do passado.
            google_news = GNews(language='en', country='US', start_date=(ano, mes, 1), end_date=(ano, mes, 28), max_results=100)
            
            try:
                resultados = google_news.get_news(TOKEN_PESQUISA)
                
                for artigo in resultados:
                    titulo = artigo.get('title', '')
                    data_pub = artigo.get('published date', '')
                    
                    # Sentiment Analysis em tempo real local
                    analise = TextBlob(titulo)
                    score_sentimento = analise.sentiment.polarity
                    
                    classificacao = "Neutro"
                    if score_sentimento > 0.1: classificacao = "Positivo (Comprar)"
                    if score_sentimento < -0.1: classificacao = "Negativo (Vender)"

                    try:
                        # Corrige a data confusa do Google ('Tue, 28 Feb 2024...') para o padrão ('YYYY-MM-DD HH:MM:00') do nosso mesclar_dados
                        data_convertida = pd.to_datetime(data_pub).strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        data_convertida = data_pub

                    noticias_processadas.append({
                        'Data_Publicacao': data_convertida,
                        'Manchete': titulo,
                        'Score_IA': round(score_sentimento, 4),
                        'Classificacao': classificacao
                    })
                    
                print(f"   [+] Capturadas {len(resultados)} manchetes intocadas.")
                
                # PAUSA ANTI-BANIMENTO (O segredo do Scraper durável)
                time.sleep(4)
                
            except Exception as e:
                print(f"   [!] Bloqueio temporário ou erro ao buscar mês {mes}: {e}")

    # Salva o arquivo pra enviar pra IA
    if noticias_processadas:
        pasta_dados = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dados")
        os.makedirs(pasta_dados, exist_ok=True)
        caminho_arquivo = os.path.join(pasta_dados, f"noticias_historicas_{TOKEN_PESQUISA}.csv")
        
        df = pd.DataFrame(noticias_processadas)
        # Ordenamos da manchete mais antiga para a mais nova
        df = df.sort_values(by='Data_Publicacao') 
        df.to_csv(caminho_arquivo, index=False)
        
        print("\n🏆 MISSÃO CUMPRIDA! EXTENSÃO SCRAPING FINALIZADA.")
        print(f"Total de Manchetes blindadas contra APIs pagas: {len(df)}")
        print(f"Arquivo gerado em: {caminho_arquivo}")
        print("\nAMOSTRA DOS FRUTOS DA MINERAÇÃO PROFUNDA:")
        print(df.head())
    else:
        print("\nFalha: O Google bloqueou o bot ou nenhuma notícia encontrada.")

if __name__ == "__main__":
    main()
