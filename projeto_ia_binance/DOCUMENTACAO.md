# 📑 Relatório Técnico: Estrutura do Projeto IA de Trading

Este documento explica cada parte do seu projeto, o que cada arquivo faz e como eles se conectam. Seu projeto agora segue uma estrutura profissional de Ciência de Dados.

---

## 📁 Pastas (Diretórios)

1.  **`/scripts`**: É o "cérebro" do projeto. Contém todos os códigos em Python que realizam tarefas (coleta, treino, teste).
2.  **`/dados`**: O "depósito". Aqui ficam todos os arquivos `.csv` com o histórico de preços que baixamos da Binance.
3.  **`/modelos`**: O "cofre". Guarda os arquivos `.h5` (os cérebros treinados da sua IA).
4.  **`/graficos`**: A "galeria". Contém as imagens `.png` geradas pelas simulações em massa.

---

## 🐍 Scripts Python (`/scripts`)

### 🛠️ Coleta de Dados
- **`coletar.py`**: Baixa o histórico de **uma** moeda específica. Aceita argumentos como o nome da moeda e quantos meses baixar.
- **`coletar_em_massa.py`**: Um script automatizado que chama o `coletar.py` repetidas vezes para baixar os 24 meses das 10 maiores moedas do mercado de uma vez.

### 🧠 Treinamento (Aprendizado)
- **`treinar.py`**: Treina a IA para o **curto prazo** (prever o que acontece na próxima 1 hora). É excelente para seguir a tendência atual.
- **`treinar_24h.py`**: O modo **PRO**. Treina a IA para prever o preço de **amanhã** (24 horas à frente). Usa uma rede neural mais profunda (3 camadas).

### 📊 Simulação e Teste
- **`simular.py`**: Roda um teste em uma moeda e abre um gráfico na tela para você ver a linha azul (real) vs vermelha (IA).
- **`simular_lote.py`**: Roda o teste em **todas** as moedas da pasta `/dados` de uma vez. Ele não abre janelas; em vez disso, salva os gráficos como imagens na pasta `/graficos/` e imprime um ranking de qual moeda a IA acertou mais.
- **`relatorio_diario.py`**: O "Modo Simplificado". Mostra apenas se a IA acertou a **direção** (Subiu/Desceu) nos últimos 7 ou 30 dias. É a forma mais fácil de auditar se a IA está funcionando.

---

## 📄 Arquivos de Suporte

- **`COMO_USAR.md`**: Seu guia rápido de comandos. Se esquecer como rodar algo, olhe aqui primeiro.
- **`requirements.txt`**: A lista de "ingredientes" (bibliotecas como TensorFlow, Pandas, CCXT) que o Python precisa para rodar o projeto.
- **`dados/*.csv`**: Cada arquivo deste contém milhares de linhas com preço de abertura, máxima, mínima e fechamento de cada hora dos últimos 2 anos.
- **`modelos/*.h5`**: São os arquivos binários da IA. Se você deletar isso, terá que treinar a IA novamente do zero.

---

## 🔄 Fluxo de Trabalho Recomendado

1.  **Coletar**: `python scripts/coletar.py "MOEDA/USDT" 24`
2.  **Treinar**: `python scripts/treinar_24h.py "dados/moeda.csv" 20`
3.  **Auditar**: `python scripts/relatorio_diario.py modelos/modelo.h5 dados/moeda.csv 30`

Este projeto foi construído para ser **modular**: você pode adicionar novas moedas ou novos tipos de "videntes" (ex: prever 1 semana à frente) apenas criando novos scripts na pasta `/scripts` sem bagunçar o resto!
