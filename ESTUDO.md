# Guia de estudo deste projeto

Para cada parte do código: o que estudar, onde estudar e as perguntas que um entrevistador
pode fazer. Leia na ordem. Responda as perguntas em voz alta antes de olhar a resposta.

## 1. O problema e a métrica (`src/rotulos.py`, `src/metricas.py`)

**Conceitos:** classificação multiclasse, desbalanceamento, acurácia, precisão, revocação
(recall), F1, F1 macro, matriz de confusão.

**Onde estudar:** Google ML Crash Course, módulo "Classification" (gratuito, em português).
Depois a página `classification_report` da documentação do scikit-learn.

**Perguntas:**
- Por que a acurácia engana neste problema?
  > 59% do teste é "elogio". Um modelo que responde "elogio" para tudo acerta 59% sem saber nada.
- O que o F1 macro mede que a acurácia não mede?
  > A média do F1 de cada classe, com peso igual. Errar todas as avaliações de "atraso" derruba o
  > F1 macro mesmo que "atraso" seja só 4% da base.
- Por que existe um `f1_macro_reclamacoes`?
  > Separar elogio de reclamação é fácil (a nota quase entrega). O valor do modelo está em dizer
  > *qual* reclamação é, e essa métrica isola isso.

## 2. Dados e vazamento (`src/preparar.py`)

**Conceitos:** treino, validação e teste; vazamento de dados (data leakage); duplicatas;
amostragem estratificada.

**Onde estudar:** Google ML Crash Course, "Datasets, generalization, and overfitting".

**Perguntas:**
- Por que tirar duplicatas antes de separar o teste?
  > "Recebi no prazo" aparece centenas de vezes. Se a mesma frase está no treino e no teste,
  > o modelo é avaliado em algo que já decorou e a métrica fica otimista.
- Por que o teste é estratificado pela nota?
  > Para ter reclamações e elogios na mesma proporção da base real. Assim a métrica reflete o
  > uso de verdade.
- Para que servem as colunas `entregue` e `chegou_atrasado` se o modelo não as usa?
  > São uma verificação de fora: dá para ver se o que o modelo chama de "atraso" bate com
  > pedidos que de fato chegaram depois do prazo.

## 3. Rotulagem com LLM (notebook, seção 2)

**Conceitos:** rotulagem fraca (weak supervision), LLM como rotulador, destilação de
conhecimento (professor e aluno), quantização.

**Onde estudar:** post "Distilling step-by-step" do Google Research; a documentação do
vLLM (structured outputs); o capítulo de quantização do curso de LLM da Hugging Face.

**Perguntas:**
- Por que não rotular as 6.000 avaliações à mão?
  > Custo. 500 à mão viraram o teste, que precisa ser confiável. Para treinar, rótulos um pouco
  > ruidosos de um modelo grande funcionam, desde que se meça o quanto ele erra.
- Como você sabe se o professor é bom?
  > Rodando ele no teste rotulado à mão. A concordância dele é o teto prático do aluno: o aluno
  > aprende com os erros do professor também.
- O que é um modelo "4-bit"?
  > Os pesos são guardados com 4 bits em vez de 16. O modelo de 7B cai de ~15 GB para ~4 GB e
  > cabe numa GPU T4 de 16 GB com folga, com perda pequena de qualidade.
- Por que temperatura 0?
  > Para a resposta ser determinística: o mesmo texto sempre recebe o mesmo rótulo.
- O que é "saída restrita" (structured outputs)?
  > O vLLM só deixa o modelo gerar uma das 8 palavras da lista. Sem isso, ele às vezes responde
  > "Rótulo: atraso." ou inventa uma categoria, e seria preciso tratar esses casos.

## 4. Baseline clássico (`src/baseline_tfidf.py`)

**Conceitos:** TF-IDF, n-gramas de palavras e de caracteres, regressão logística,
`class_weight="balanced"`.

**Onde estudar:** documentação do scikit-learn, "Working with text data".

**Perguntas:**
- Por que começar por um modelo simples?
  > Para saber se o modelo de linguagem vale o custo. Se o TF-IDF chega perto, a resposta
  > certa para produção pode ser ele: roda em microssegundos e sem GPU.
- Para que servem os n-gramas de caracteres?
  > Pegam erros de digitação ("recebir", "nao") e variações de palavras que o vocabulário de
  > palavras inteiras perde.

## 5. LoRA (notebook, seção 5)

**Conceitos:** ajuste fino (fine-tuning), LoRA, camada de classificação, função de perda
(entropia cruzada), pesos por classe, taxa de aprendizado com aquecimento, padding.

**Onde estudar:** artigo original do LoRA (Hu et al., 2021, só a introdução e a figura 1);
documentação do PEFT na Hugging Face; o vídeo "LoRA explained" da IBM Technology.

**Perguntas:**
- O que o LoRA treina?
  > Em vez de mudar a matriz de pesos W inteira, aprende duas matrizes pequenas A e B com
  > W + B·A. Com r=16 isso é cerca de 1% dos parâmetros. O modelo original fica congelado.
- Por que um modelo de 0,5B e não o de 7B?
  > Custo e latência. O 7B rotula bem, mas é caro para rodar em cada avaliação em produção.
  > A pergunta do projeto é quanto do desempenho do 7B dá para manter num modelo 14 vezes menor.
- Por que padding à esquerda?
  > O classificador do Qwen lê o último token. Com padding à esquerda, o último token é
  > sempre texto de verdade, nunca preenchimento.
- Por que pesos por classe na perda?
  > Sem eles, o modelo aprende que chutar a classe mais comum minimiza a perda.
- Como você saberia se houve overfitting?
  > Perda de treino caindo e acurácia de validação parando de subir ou caindo. O
  > `historico_lora.json` (saída do notebook) guarda as duas por época.

## 6. Comparação final (`resultados/`)

**Perguntas:**
- Qual modelo você colocaria em produção, e por quê?
  > Depende do que o negócio precisa. O aluno com LoRA tem F1 de 0,635 a 4,6 ms; o TF-IDF, 0,571 a
  > 0,26 ms e sem GPU. Se um encaminhamento errado custa caro (um "não recebido" que vai para o
  > time errado), os 6 pontos valem a GPU. Se o volume é enorme e o erro é barato, o TF-IDF.
  > O 7B (0,698) só compensa se a qualidade for crítica e o volume, baixo.
- Por que o 0,5B sem treino acertou só 16%?
  > Modelos pequenos seguem mal instruções longas. Com 8 rótulos descritos no prompt, ele quase
  > nunca escolhe "elogio". O ajuste fino troca o prompt por exemplos, e aí o mesmo modelo chega
  > a 81% de acurácia.
- Como você sabe que o modelo aprendeu algo real, e não só a imitar o professor?
  > Pela conferência externa: avaliações que ele chama de "atraso" têm 36% de pedidos realmente
  > atrasados, contra 3% das que ele chama de "elogio". Ele nunca viu as datas de entrega.
- Onde o modelo mais erra?
  > "nao_recebido" virando "atraso" ou "atendimento_cobranca". A fronteira depende de a pessoa
  > dizer que o pedido ainda não chegou, e muitas reclamações misturam isso com falta de resposta.
- O que você faria com mais tempo?
  > Revisar os rótulos do professor nas classes onde ele mais erra, treinar mais exemplos
  > dessas classes e testar um modelo maior (1,5B) para ver se a curva compensa.
