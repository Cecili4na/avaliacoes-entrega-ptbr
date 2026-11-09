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

> Em execução no Kaggle (GPU T4). A tabela entra aqui com o notebook publicado.

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
