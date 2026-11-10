# Resultados da Regressão Logística na Detecção de ESP

## 1. Configuração experimental

A regressão logística foi utilizada como modelo linear de referência para a
classificação de comportamentos associados a ESP/wallhack. O modelo foi
configurado com regularização L2, parâmetro `C = 1,0`, balanceamento automático
das classes e limiar de decisão igual a `0,50`. Antes do treinamento, valores
ausentes foram preenchidos pela mediana e todas as features foram padronizadas
com `StandardScaler`.

A avaliação utilizou Leave-One-Group-Out (LOGO), considerando cada partida como
um grupo independente. Em cada rodada, uma partida completa foi reservada para
teste e as demais foram utilizadas no treinamento. Dessa forma, janelas da
mesma partida nunca foram compartilhadas entre treino e teste.

Antes da classificação, foi aplicada uma porta determinística de ESP. Somente
janelas com alvo oculto e distância média superior a 300 metros foram enviadas
ao modelo. Arma, identificadores de jogador, nome da partida e respostas do
motor de regras não foram utilizados como features.

## 2. Avaliação principal com 17 partidas

O experimento principal reuniu 17 partidas e 5.618 janelas rotuladas. Como o
conjunto combina duas gerações de telemetria, foram utilizadas as 30 features
comuns a todas as partidas.

| Métrica | Resultado |
|---|---:|
| Verdadeiros negativos | 5.421 |
| Falsos positivos | 125 |
| Falsos negativos | 5 |
| Verdadeiros positivos | 67 |
| Precisão hacker | 34,9% |
| Recall hacker | 93,1% |
| F1-score hacker | 50,8% |
| Taxa de falso positivo | 2,25% |
| Partidas hacker detectadas | 8 de 9 |
| Partidas com algum falso positivo | 11 de 17 |

Das 72 janelas hacker, 67 foram corretamente identificadas. O recall de 93,1%
mostra que a regressão logística foi sensível à classe positiva, deixando de
detectar somente cinco janelas hacker. Entretanto, das 192 janelas classificadas
como hacker, 125 eram normais, resultando em precisão de 34,9%.

A porta determinística enviou 430 das 5.618 janelas ao júri, aproximadamente
7,7% do total. Assim, cerca de 92,3% das janelas foram descartadas antes da
classificação por não apresentarem simultaneamente alvo oculto e distância
superior a 300 metros.

A regressão detectou ações hacker em oito das nove partidas positivas. A única
partida hacker sem qualquer verdadeiro positivo foi `par5plyr2h`.

## 3. Estudo de ablação com seis partidas contextuais

O estudo de ablação utilizou as mesmas seis partidas, os mesmos rótulos e as
mesmas rodadas do LOGO. A única diferença foi o conjunto de features:

- 30 features comportamentais comuns;
- 37 features, incluindo contexto histórico do mesmo alvo.

| Métrica | Features comuns | Contexto rico |
|---|---:|---:|
| Verdadeiros negativos | 2.455 | **2.469** |
| Falsos positivos | 56 | **42** |
| Falsos negativos | **3** | 10 |
| Verdadeiros positivos | **40** | 33 |
| Precisão hacker | 41,7% | **44,0%** |
| Recall hacker | **93,0%** | 76,7% |
| F1-score hacker | **57,6%** | 55,9% |
| Taxa de falso positivo | 2,23% | **1,67%** |
| Partidas hacker detectadas | 2 de 2 | 2 de 2 |
| Partidas com algum falso positivo | 4 de 6 | 4 de 6 |

As features contextuais reduziram os falsos positivos de 56 para 42, uma
redução relativa de 25%. A precisão aumentou de 41,7% para 44,0% e a taxa de
falso positivo caiu de 2,23% para 1,67%.

Esse ganho de especificidade foi acompanhado por perda de sensibilidade. O
recall caiu de 93,0% para 76,7%, com sete verdadeiros positivos a menos. Como
consequência, o F1-score diminuiu levemente, de 57,6% para 55,9%.

Os dois experimentos enviaram as mesmas 202 janelas ao júri, aproximadamente
7,9% das 2.554 janelas. Portanto, a mudança observada foi causada pelas features
contextuais, e não pela porta determinística.

## 4. Comparação com o Random Forest

Para permitir uma comparação justa, os dois algoritmos utilizaram os mesmos
manifestos, as mesmas divisões LOGO, a mesma porta determinística e o mesmo
limiar de decisão.

### 4.1 Conjunto ampliado de 17 partidas

| Métrica | Regressão logística | Random Forest |
|---|---:|---:|
| Precisão hacker | 34,9% | **47,8%** |
| Recall hacker | **93,1%** | 75,0% |
| F1-score hacker | 50,8% | **58,4%** |
| Taxa de falso positivo | 2,25% | **1,06%** |
| Falsos positivos | 125 | **59** |
| Falsos negativos | **5** | 18 |
| Partidas hacker detectadas | **8 de 9** | 7 de 9 |

A regressão logística apresentou maior recall e detectou uma partida hacker a
mais. Em contrapartida, produziu mais que o dobro de falsos positivos. O Random
Forest apresentou melhor precisão, maior F1-score e menor taxa de falso
positivo, oferecendo um equilíbrio mais adequado entre detecção e confiabilidade.

### 4.2 Seis partidas com contexto rico

| Métrica | Regressão logística | Random Forest |
|---|---:|---:|
| Precisão hacker | 44,0% | **62,5%** |
| Recall hacker | **76,7%** | 69,8% |
| F1-score hacker | 55,9% | **65,9%** |
| Taxa de falso positivo | 1,67% | **0,72%** |
| Falsos positivos | 42 | **18** |
| Falsos negativos | **10** | 13 |

No conjunto contextual, o Random Forest novamente apresentou melhor precisão,
F1-score e controle de falsos positivos. A regressão logística manteve maior
recall, mas com mais que o dobro de acusações incorretas.

As features contextuais melhoraram o F1-score do Random Forest, mas não o da
regressão logística. Isso sugere que parte do valor dessas variáveis depende de
relações não lineares ou de interações entre features, que são representadas
com mais facilidade por um conjunto de árvores.

## 5. Discussão

A regressão logística demonstrou ser adequada como baseline linear e como
classificador de alta sensibilidade. Seu principal benefício foi o recall de
93,1% no conjunto ampliado, indicando baixa quantidade de ações hacker não
detectadas.

Entretanto, a baixa precisão mostra que o modelo se comportou de maneira mais
agressiva, encaminhando muitas janelas normais para revisão. Esse comportamento
pode ser aceitável em uma etapa inicial de triagem, desde que a decisão final
seja tomada por outro mecanismo ou por revisão humana. Ele não é adequado para
banimento automático.

O Random Forest apresentou melhor equilíbrio geral. Na avaliação com 17
partidas, reduziu os falsos positivos de 125 para 59 e elevou o F1-score de
50,8% para 58,4%. Dessa forma, entre os dois modelos, o Random Forest mostrou-se
mais apropriado como júri final, enquanto a regressão logística pode atuar como
baseline acadêmico ou detector mais sensível.

## 6. Limitações

- O conjunto ampliado contém apenas 17 partidas independentes.
- Somente nove partidas possuem janelas positivas.
- O estudo contextual contém seis partidas e apenas duas partidas hacker.
- As janelas são temporalmente correlacionadas, embora o LOGO impeça o
  compartilhamento de uma mesma partida entre treino e teste.
- O limiar de 0,50 não foi calibrado separadamente dentro de cada fold.
- Os resultados concentram-se na detecção de ESP/wallhack e não representam
  diretamente o desempenho para aimbot ou speedhack.

## 7. Conclusão

> Na validação Leave-One-Group-Out com 17 partidas, a regressão logística
> alcançou recall de 93,1%, precisão de 34,9%, F1-score de 50,8% e taxa de falso
> positivo de 2,25%. O modelo detectou ações hacker em oito das nove partidas
> positivas, demonstrando elevada sensibilidade. Entretanto, produziu 125 falsos
> positivos, mais que o dobro dos 59 observados no Random Forest. O Random Forest
> apresentou maior precisão, maior F1-score e menor taxa de falso positivo,
> enquanto a regressão logística apresentou maior recall. Os resultados indicam
> que a regressão logística constitui um baseline linear útil e um detector
> sensível, mas o Random Forest oferece melhor equilíbrio para atuar como júri de
> triagem no sistema anti-cheat proposto.
