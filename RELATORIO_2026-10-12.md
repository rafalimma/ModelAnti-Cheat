# Relatório de evolução do anti-cheat — 12/10/2026

## Objetivo da sessão

Avaliar o modelo com todas as partidas rotuladas e melhorar a distinção entre ESP/X-ray e comportamentos legítimos, principalmente em confrontos de sniper acima de 300 metros.

## Resultados obtidos

A validação Leave-One-Group-Out (LOGO) foi executada com 11 partidas e `min_samples_leaf=5`.

### Modelo anterior — 23 features

- Verdadeiros negativos: 2.982
- Falsos positivos: 53
- Falsos negativos: 3
- Verdadeiros positivos: 26
- F1 hacker: 0,481
- Taxa de falso positivo: 1,75%
- Janelas enviadas ao júri: 228 de 3.064

### Modelo com evidências de ESP — 30 features

- Verdadeiros negativos: 2.988
- Falsos positivos: 47
- Falsos negativos: 4
- Verdadeiros positivos: 25
- F1 hacker: 0,495
- Taxa de falso positivo: 1,55%

A porta determinística funcionou corretamente: somente janelas com alvo oculto e distância superior a 300 metros foram enviadas à árvore.

## Diagnóstico

A maior concentração de falsos positivos ocorreu em `par8plyr2h`. Nessa partida, os dois jogadores participavam de um confronto legítimo de sniper e conheciam a região onde o adversário estava. Por isso, ambos produziram telemetria semelhante a ESP ao mirar em posições ocultas.

Também verificamos que ADS oculto repetido, isoladamente, não distingue o hacker com segurança. Esse comportamento pode ocorrer legitimamente quando o jogador viu o adversário entrar atrás de uma construção ou terreno.

## Decisões tomadas

- Manter a porta determinística de ESP: alvo oculto e distância maior que 300 metros.
- Não usar arma, identidade do jogador ou resultados das regras como features da IA.
- Remover `duracao_rastreando_oculto_max` do treinamento, impedindo que perseguição simples seja a principal evidência.
- Usar ADS, precisão, duração e repetição como evidências comportamentais, mas não como prova automática.
- Acrescentar contexto do mesmo alvo para distinguir conhecimento legítimo de acompanhamento por X-ray.
- Continuar interpretando a saída como revisão prioritária, não banimento automático.

## Alterações implementadas

### `init.c`

- Passou a registrar a identidade técnica do alvo selecionado.
- Passou a registrar a posição da cabeça do alvo.
- O identificador será usado apenas para relacionar eventos do mesmo alvo e não entrará no modelo.

### `preprocessor_ens.py`

Foram adicionadas features relacionadas a:

- tempo desde a última visão direta do mesmo alvo;
- visão do alvo nos últimos 30 e 60 segundos;
- deslocamento do alvo desde a última visão;
- ADS oculto sem visão recente;
- acompanhamento de alvo que se moveu escondido;
- troca entre diferentes alvos ocultos;
- episódios repetidos de ADS oculto.

Também foi incluída a coluna `contexto_alvo_confiavel`, que impede o uso dessas features em conjuntos formados por logs antigos sem identidade do alvo.

### `trainer_arvore_decisao.py`

- A lista de features passou de 23 para 30 nos datasets atuais.
- Foram incluídas duração de ADS oculto, precisão e contagem de episódios.
- As novas features contextuais serão habilitadas somente quando todas as partidas do manifesto tiverem sido geradas pelo novo `init.c`.

## Próximos passos

1. Instalar o novo `init.c` e reiniciar o servidor.
2. Coletar novas partidas com confronto normal, X-ray sem visão anterior e X-ray acompanhando alvo em movimento atrás de obstáculos.
3. Incluir testes com ADS repetido e troca entre dois alvos ocultos.
4. Processar os novos logs e confirmar `contexto_alvo_confiavel = 1`.
5. Criar um manifesto separado contendo somente partidas do novo formato.
6. Deixar janelas de transição sem rótulo para não contaminar o ground truth.
7. Executar novamente o LOGO com `max_depth=4` e `min_samples_leaf=5`.
8. Comparar principalmente precisão, recall, F1, falsos positivos por partida e regras aprendidas pela árvore.

## Conclusão

O pipeline está funcionando e a validação mostrou uma limitação real: a mira oculta acima de 300 metros pode ocorrer em confrontos legítimos. A próxima evolução é avaliar se o jogador tinha informação recente sobre aquele mesmo alvo e se continuou seguindo sua posição real enquanto ele se movia escondido. Esse contexto deve tornar o júri mais confiável sem transformar uma regra isolada em acusação automática.
