# Previsão de Preços com Redes Neurais (Séries Temporais)

Prever preços (como ações, criptomoedas ou vendas) é um desafio de **Séries Temporais**, onde a ordem dos dados importa tanto quanto os valores.

---

## 1. A Abordagem: Redes LSTM

Para este tipo de problema, a abordagem mais comum e eficaz é o uso de redes **LSTM (Long Short-Term Memory)**.

### Por que LSTM?
Diferente das redes neurais comuns que "esquecem" tudo após processar um dado, as LSTMs possuem uma **memória interna**. Elas conseguem identificar padrões que aconteceram há 10, 50 ou 100 dias atrás e entender como isso afeta o preço de amanhã.

---

## 2. Preparação dos Dados (O Ponto Crítico)

Uma rede neural não entende "datas" ou "preços" brutos diretamente de forma eficiente. Você precisa transformá-los:


### A. Normalização
Os preços podem variar de R$ 10 a R$ 100.000. Redes neurais funcionam melhor com números entre **0 e 1**.
- Usamos ferramentas como o `MinMaxScaler` para achatar os valores.

### B. Janelamento (Windowing / Lookback)
A rede não recebe uma única linha. Ela recebe uma "janela" de tempo.
- **Exemplo:** Para prever o preço de amanhã, você entrega para a rede os preços dos últimos 30 dias.
- **Input:** [P1, P2, ..., P30] -> **Output:** [P31]

### C. Tratamento de Datas
Datas como "26/03/2026" são convertidas em informações numéricas úteis:
- Dia da semana (0-6)
- Mês (1-12)
- Se é feriado ou não (0 ou 1)

---

## 3. Estratégias Fixas vs. Redes Neurais

Muitas pessoas tentam criar robôs usando regras simples (ex: "Se o preço cair 2% e o indicador X for Y, então compre"). Isso costuma falhar drasticamente porque o mercado é dinâmico — o que funcionou em 2021 raramente funciona em 2024.

### Por que a Rede Neural pode ajudar?
Diferente de uma estratégia fixa, a rede neural **ajusta seus próprios pesos** (os valores que você mencionou) automaticamente. 
-   Se ela perceber que uma certa queda de preço não leva mais a uma subida, ela "diminui a importância" desse padrão.
-   Ela consegue encontrar relações sutis que um humano ou uma regra "if/else" nunca veriam.

### O Perigo do "Ajuste Automático" (Overfitting)
O fato de ela modificar os valores até chegar no objetivo é uma faca de dois gumes. 
-   **O Risco:** Se você der à rede poucos dados (poucos JSONs), ela vai encontrar um jeito de "decorar" exatamente como ganhar dinheiro *naqueles dados específicos*.
-   **O Resultado:** Ela vai parecer um gênio no computador, mas na hora que você ligar ela no gráfico real (que ela nunca viu), ela vai "errar drasticamente" novamente.

**Dica de Ouro:** Sempre teste sua rede neural em dados que ela **não viu** durante o treinamento (Dados de Teste). Se ela for bem nos dois, aí sim você tem algo sólido.

---

## 4. Fluxo de Trabalho Sugerido

1.  **Dados Históricos:** Baixe um CSV com colunas `Data` e `Preço Fechamento`.
2.  **Feature Engineering:** Adicione indicadores (Médias Móveis, RSI) se quiser ajudar a rede.
3.  **Split:** Use os primeiros 80% dos dados para treinar e os últimos 20% para testar (nunca misture a ordem temporal!).
4.  **Modelo:**
    ```python
    # Exemplo conceitual em Keras:
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(janela, 1)),
        LSTM(units=50),
        Dense(1) # Saída: O preço previsto
    ])
    ```

---

## 4. O "Pulo do Gato": Expectativa vs Realidade

> [!WARNING]
> **Cuidado:** O mercado financeiro é "ruidoso". Uma rede neural pode aprender perfeitamente o passado (Overfitting), mas falhar ao prever o futuro porque eventos externos (notícias, crises) não estão nos dados de preço.
> Use a rede como uma ferramenta de auxílio, não como uma bola de cristal definitiva.

---
### Próximos Passos:
Procure por tutoriais de **"Stock Price Prediction with LSTM and Python"** para ver o código em ação.
