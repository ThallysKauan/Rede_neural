# Projeto IA Binance: Guia de Início Rápido (v2.0)

Este projeto foi organizado para ser profissional e fácil de usar. Agora, as ferramentas (scripts), os dados (.csv) e os modelos (.h5) ficam em pastas separadas para melhor organização.

---

## 📂 Estrutura do Projeto

*   **scripts/**: Onde ficam os códigos Python (as ferramentas).
*   **dados/**: Onde o histórico de preços é salvo.
*   **modelos/**: Onde o "cérebro" treinado da IA é guardado.

---

## 🚀 Passo 1: Preparar o Ambiente

1. Entre na pasta do projeto e instale as bibliotecas:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Passo 2: Coletar Dados (Histórico)

Você pode baixar dados de qualquer moeda e qualquer período (em meses).
Exemplo: Baixar 24 meses de Ethereum (ETH):
```bash
python scripts/coletar.py "ETH/USDT" 24
```
*Isso criará o arquivo `dados/eth_usdt_24m.csv`.*

---

## 🧠 Passo 3: Treinar a sua IA

Treine a IA usando um arquivo de dados específico:
```bash
python scripts/treinar.py "dados/eth_usdt_24m.csv"
```
*O modelo será salvo automaticamente na pasta `modelos/`.*

---

## 📈 Passo 4: Simulação e Teste (Backtesting)

Para ver como a IA se comporta ao longo do tempo (simulação mês a mês):
```bash
python scripts/simular.py "modelos/modelo_eth_usdt_24m.h5" "dados/eth_usdt_24m.csv"
```
*Este comando abrirá um gráfico e mostrará a performance mensal no terminal.*

---

## ⚠️ Avisos Importantes

1.  **Isto é Educativo:** Use este projeto para aprender. Nunca invista dinheiro real baseado apenas em previsões de IA básica.
2.  **Organização:** Sempre rode os comandos da **raiz do projeto**, chamando os scripts pelo caminho `scripts/nome_do_arquivo.py`.

---
*Divirta-se com seu novo sistema de IA organizado!*
