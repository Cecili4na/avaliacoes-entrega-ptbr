# avaliacoes-entrega-ptbr

Classifica o **motivo** de avaliações de compras online em português: pedido não recebido, atraso,
pedido incompleto, produto errado, defeito, atendimento ou cobrança, elogio, outro.

A pergunta do projeto: **quanto do desempenho de um LLM de 7B dá para manter num modelo 14 vezes
menor**, treinado com os rótulos do grande? E vale a pena, comparado a um modelo clássico?

## Abordagens comparadas

| | Modelo | Treino |
|---|---|---|
| Baseline | TF-IDF (palavras e pedaços de palavras) + regressão logística | rótulos do professor |
| Zero-shot | Qwen2.5-0.5B-Instruct | nenhum, só o prompt |
| Ajuste fino | Qwen2.5-0.5B + LoRA (r=16, ~1% dos parâmetros) | rótulos do professor |
| Professor | Qwen2.5-7B-Instruct (AWQ, 4 bits) | nenhum, só o prompt |

## Dados

- **Fonte:** avaliações públicas da Olist (e-commerce brasileiro), licença CC BY-NC-SA 4.0.
  35.606 avaliações com comentário, sem duplicatas.
- **Teste:** 500 avaliações rotuladas à mão (`data/anotacao/`), estratificadas pela nota, para
  ter a proporção real de elogios e reclamações. Quase 60% são elogios.
- **Treino:** 6.000 avaliações, dois terços com nota de 1 a 3, rotuladas pelo professor.
- As regras de cada rótulo estão em `src/rotulos.py`. A fronteira mais delicada é entre
  `nao_recebido` e `atraso`: depende de a pessoa dizer ou não que o pedido ainda não chegou.

## Métrica principal

**F1 macro das reclamações** (as 7 classes sem `elogio`). Separar elogio de reclamação é fácil;
o valor está em dizer *qual* é o problema. A acurácia engana aqui: responder "elogio" para tudo
acerta 59%.

Há também uma **conferência externa**: a base tem a data prometida e a data real de entrega, que
o modelo nunca vê. Se ele aprendeu algo, avaliações classificadas como `atraso` devem ter muito
mais pedidos atrasados de verdade que as classificadas como `elogio`.

## Resultados

Teste com 500 avaliações rotuladas à mão, GPU T4 do Kaggle:

| Modelo | F1 reclamações | F1 macro | Acurácia | ms por avaliação |
|---|---|---|---|---|
| Professor: Qwen2.5-7B (AWQ) | **0,700** | 0,733 | 0,862 | 34,1 |
| Aluno: Qwen2.5-0.5B + LoRA | **0,599** | 0,644 | 0,810 | 4,3 |
| TF-IDF + regressão logística | 0,566 | 0,614 | 0,802 | 0,26 |
| Qwen2.5-0.5B sem treino | 0,241 | 0,212 | 0,160 | 6,1 |

O que a tabela diz:

- **O aluno mantém 86% do F1 do professor** (0,599 de 0,700), com 14 vezes menos parâmetros e
  8 vezes menos latência.
- **O modelo pequeno sem treino não serve**: acerta 16%, quando só responder "elogio" para
  tudo já daria 59%. Com o mesmo prompt do professor, ele quase nunca reconhece um elogio. O ajuste fino é o que torna o 0,5B útil.
- **O ganho sobre o TF-IDF é real, mas modesto**: 3,3 pontos de F1, com latência 17 vezes maior.
  Com rótulos ruidosos do professor, o modelo clássico já aprende boa parte do sinal. Para
  produção, a escolha depende do volume e de quanto custa um encaminhamento errado.

**Conferência externa.** Em 3.000 avaliações fora do treino, comparei o que o aluno previu com a
entrega real, que ele nunca viu:

| Previsto pelo aluno | Avaliações | Pedido chegou atrasado | Pedido consta como entregue |
|---|---|---|---|
| nao_recebido | 203 | 47% | 73% |
| atraso | 164 | 36% | 89% |
| atendimento_cobranca | 230 | 29% | 83% |
| elogio | 1.748 | 3% | 100% |

As avaliações que ele chama de `atraso` têm 12 vezes mais pedidos realmente atrasados que as
chamadas de `elogio`, e `nao_recebido` concentra os pedidos que nunca foram entregues.

**Onde ele erra.** São 95 erros em 500. O padrão dominante é `nao_recebido` virar `atraso` ou
`atendimento_cobranca` ("até hoje não recebi e ninguém responde"). É exatamente a fronteira mais
delicada da definição dos rótulos, e a próxima melhoria seria revisar os rótulos do professor
nessas três classes.

## Rodando

A parte que precisa de GPU está em `kaggle/avaliacoes-entrega-ptbr.ipynb`: abra no Kaggle com
GPU T4 x2 e internet ligada, e adicione os dados de `data/processed/` como dataset.

Localmente (preparação e baseline, segundos):

```bash
uv sync
mkdir -p data/raw && for f in olist_order_reviews_dataset.csv olist_orders_dataset.csv; do
  curl -L -o data/raw/$f https://huggingface.co/datasets/bulutttt/olist-raw-data/resolve/main/$f
done
uv run python src/preparar.py       # base sem duplicatas e separação do teste
uv run python src/montar_teste.py   # junta os rótulos manuais
```

O guia `ESTUDO.md` explica cada decisão e as perguntas que ela responde.
