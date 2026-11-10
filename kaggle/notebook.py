# %% [markdown]
# # Qual é o problema da entrega? Destilando um LLM de 7B num classificador de 0,5B
#
# Avaliações de clientes chegam aos milhares, e cada tipo de reclamação vai para um time
# diferente: pedido que não chegou, atraso, item faltando, produto errado, defeito, atendimento.
# Este notebook constrói um classificador desse **motivo da reclamação** para avaliações em
# português e compara três abordagens em qualidade, latência e custo:
#
# 1. **TF-IDF + regressão logística**: o baseline clássico.
# 2. **Qwen2.5-0.5B-Instruct sem treino** (zero-shot): só o prompt.
# 3. **Qwen2.5-0.5B com LoRA**: ajuste fino com rótulos gerados por um modelo 14 vezes maior.
#
# **Dados:** avaliações públicas da Olist (e-commerce brasileiro, CC BY-NC-SA 4.0).
# O teste tem 500 avaliações rotuladas à mão. O treino é rotulado pelo
# **Qwen2.5-7B-Instruct** (o professor), cuja qualidade também é medida contra o teste.
#
# Código e guia de estudo: github.com/Cecili4na/avaliacoes-entrega-ptbr

# %% [markdown]
# ## 0. Ambiente
#
# Rode com **GPU T4 x2** (Settings → Accelerator). A instalação do vLLM leva uns 3 minutos e
# reinicia a sessão uma vez; depois disso, rode tudo de novo.

# %%
# Instala o vLLM e reinicia a sessão uma vez: o pip troca a versão do torch, e o torch antigo
# não pode continuar carregado na memória. Depois do reinício, "Run All" de novo pula esta parte.
import importlib.util
import os
import subprocess
import sys

if importlib.util.find_spec("vllm") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm", "peft>=0.13"], check=True)
    # o torchao que vem no Kaggle é antigo demais para o peft e quebra o get_peft_model; não usamos
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=True)
    print("Instalado. A sessão vai reiniciar: rode tudo de novo (Run All).")
    os._exit(0)

# %%
import gc
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline, make_union

ENTRADA = next(Path("/kaggle/input").rglob("teste.csv")).parent
SAIDA = Path("/kaggle/working")
SEMENTE = 42
torch.manual_seed(SEMENTE)
print(ENTRADA, torch.cuda.device_count(), "GPU(s)")

# %% [markdown]
# ## 1. Rótulos
#
# Um rótulo por avaliação: o motivo principal. A fronteira mais delicada é entre
# `nao_recebido` e `atraso`, e ela foi definida pelo que a pessoa diz: se afirma que nada
# chegou ainda, é `nao_recebido`; se fala de demora sem dizer isso, é `atraso`.

# %%
ROTULOS = {
    "nao_recebido": "No momento da avaliação, nada do pedido chegou: diz que não recebeu, que está esperando, que extraviou ou que consta entregue sem ter chegado.",
    "atraso": "Reclama de demora ou de passar do prazo sem dizer que ainda não recebeu (chegou demorado, ou só fala em atraso).",
    "incompleto": "Chegou só parte: faltou item, unidade, peça ou acessório, ou um dos produtos do pedido não veio.",
    "produto_errado": "Chegou algo diferente do comprado ou do anunciado: outro modelo, cor, tamanho, capacidade, réplica ou falsificado.",
    "defeito_qualidade": "O produto certo chegou, mas com defeito, quebrado, amassado, sujo, não funciona ou é de qualidade ruim.",
    "atendimento_cobranca": "Atendimento, falta de resposta ou de informação, cancelamento, falta de estoque, troca, estorno, nota fiscal ou cobrança.",
    "elogio": "Avaliação positiva, sem reclamação que pese mais que o elogio.",
    "outro": "Nenhum dos anteriores: embalagem, retirada nos Correios, tentativa de entrega, crítica genérica ou texto sem conteúdo.",
}
NOMES = list(ROTULOS)
ID = {n: i for i, n in enumerate(NOMES)}
RECLAMACOES = [n for n in NOMES if n != "elogio"]

teste = pd.read_csv(ENTRADA / "teste.csv")
pool = pd.read_csv(ENTRADA / "pool.csv")
print(len(teste), "no teste ·", len(pool), "no pool")
teste["rotulo"].value_counts()

# %% [markdown]
# **Atenção ao desbalanceamento:** quase 60% do teste é `elogio`. Por isso a métrica
# principal é o **F1 macro das reclamações** (sem `elogio`): ela mede se o modelo sabe dizer
# *qual* é o problema, que é a parte útil e difícil.

# %%
RESULTADOS = []


def registrar(nome, previsto, ms, extra=None):
    verdade = teste["rotulo"]
    r = {
        "modelo": nome,
        "acuracia": accuracy_score(verdade, previsto),
        "f1_macro": f1_score(verdade, previsto, labels=NOMES, average="macro", zero_division=0),
        "f1_macro_reclamacoes": f1_score(verdade, previsto, labels=RECLAMACOES, average="macro", zero_division=0),
        "ms_por_avaliacao": ms,
        **(extra or {}),
    }
    RESULTADOS.append(r)
    (SAIDA / f"{nome}.json").write_text(json.dumps(
        {"resumo": r, "por_classe": classification_report(verdade, previsto, labels=NOMES, zero_division=0, output_dict=True)},
        ensure_ascii=False, indent=2, default=float))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(verdade, previsto, labels=NOMES, ax=ax, xticks_rotation=45, colorbar=False, cmap="Blues")
    ax.set_title(f"{nome} · F1 reclamações {r['f1_macro_reclamacoes']:.2f}")
    fig.tight_layout()
    fig.savefig(SAIDA / f"confusao_{nome}.png", dpi=110)
    plt.show()
    return r

# %% [markdown]
# ## 2. O professor: Qwen2.5-7B-Instruct rotulando o treino
#
# O treino sai do pool: dois terços de avaliações com nota 1 a 3 (reclamações são raras e são
# o que interessa) e um terço com nota 4 ou 5.
#
# Com **saída restrita** (structured outputs), o modelo só pode responder um dos 8 rótulos.
# Isso elimina respostas inválidas sem precisar de pós-processamento.

# %%
N_TREINO = 6000
ruins = pool[pool["nota"] <= 3].sample(n=N_TREINO * 2 // 3, random_state=SEMENTE)
boas = pool[pool["nota"] >= 4].sample(n=N_TREINO - len(ruins), random_state=SEMENTE)
treino = pd.concat([ruins, boas]).sample(frac=1, random_state=SEMENTE).reset_index(drop=True)

SISTEMA = (
    "Você classifica avaliações de compras online feitas por clientes brasileiros.\n"
    "Escolha o motivo principal da avaliação entre os rótulos abaixo. Se houver mais de um problema, "
    "escolha o que a pessoa mais enfatiza.\n\n"
    + "\n".join(f"- {n}: {d}" for n, d in ROTULOS.items())
    + "\n\nResponda apenas com o nome do rótulo, exatamente como escrito acima."
)

# %%
from vllm import LLM, SamplingParams


def parametros_de_saida():
    """Restringe a resposta aos rótulos; o nome da classe mudou entre versões do vLLM."""
    try:
        from vllm.sampling_params import StructuredOutputsParams
        return SamplingParams(temperature=0, max_tokens=8, structured_outputs=StructuredOutputsParams(choice=NOMES))
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        return SamplingParams(temperature=0, max_tokens=8, guided_decoding=GuidedDecodingParams(choice=NOMES))


def classificar_com_llm(llm, textos):
    conversas = [[{"role": "system", "content": SISTEMA}, {"role": "user", "content": t}] for t in textos]
    inicio = time.perf_counter()
    saidas = llm.chat(conversas, parametros_de_saida(), use_tqdm=True)
    ms = (time.perf_counter() - inicio) * 1000 / len(textos)
    respostas = [s.outputs[0].text.strip() for s in saidas]
    return [r if r in NOMES else "outro" for r in respostas], ms


def liberar(llm):
    """Devolve a memória da GPU antes do próximo modelo."""
    from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    destroy_model_parallel()
    destroy_distributed_environment()
    del llm.llm_engine
    del llm
    gc.collect()
    torch.cuda.empty_cache()

# %%
# 7B em 4 bits ocupa ~5 GB: cabe numa T4 só, sem dividir entre as duas
professor = LLM("Qwen/Qwen2.5-7B-Instruct-AWQ", quantization="awq", dtype="half",
                max_model_len=1024, gpu_memory_utilization=0.85, seed=SEMENTE)

previsto, ms = classificar_com_llm(professor, teste["texto"].tolist())
registrar("qwen7b_professor", previsto, ms, {"parametros": "7,6 B (AWQ 4-bit)"})

treino["rotulo_professor"], _ = classificar_com_llm(professor, treino["texto"].tolist())
treino.to_csv(SAIDA / "treino_professor.csv", index=False)
liberar(professor)
treino["rotulo_professor"].value_counts()

# %% [markdown]
# A concordância do professor com o teste manual é o **teto prático** do aluno: o aluno
# aprende com os acertos e com os erros do professor.
#
# ## 3. Baseline clássico: TF-IDF + regressão logística

# %%
tfidf = make_pipeline(
    make_union(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode"),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True),
    ),
    LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"),
)
tfidf.fit(treino["texto"], treino["rotulo_professor"])
inicio = time.perf_counter()
previsto = tfidf.predict(teste["texto"])
registrar("tfidf_logreg", previsto, (time.perf_counter() - inicio) * 1000 / len(teste), {"parametros": "~1 MB"})

# %% [markdown]
# ## 4. Qwen2.5-0.5B-Instruct sem treino

# %%
pequeno = LLM("Qwen/Qwen2.5-0.5B-Instruct", dtype="half", max_model_len=1024,
              gpu_memory_utilization=0.5, seed=SEMENTE)
previsto, ms = classificar_com_llm(pequeno, teste["texto"].tolist())
registrar("qwen0.5b_zero_shot", previsto, ms, {"parametros": "0,5 B"})
liberar(pequeno)

# %% [markdown]
# ## 5. Qwen2.5-0.5B com LoRA
#
# O modelo ganha uma camada de classificação (8 saídas) lida no último token. Só as matrizes
# LoRA da atenção (r=16) e essa camada são treinadas, cerca de 1% dos parâmetros.
#
# - **Padding à esquerda:** o último token é sempre texto, nunca preenchimento.
# - **Peso por classe na perda:** sem isso, chutar a classe mais comum minimiza a perda.
# - **10% do treino separado para validação:** para acompanhar overfitting por época.

# %%
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

BASE, MAX_TOKENS, LOTE, EPOCAS, TAXA = "Qwen/Qwen2.5-0.5B", 128, 16, 3, 2e-4
dispositivo = "cuda"

treino["rotulo_id"] = treino["rotulo_professor"].map(ID)
validacao = treino.sample(frac=0.1, random_state=SEMENTE)
treino_lora = treino.drop(validacao.index)
teste["rotulo_id"] = teste["rotulo"].map(ID)

tokenizer = AutoTokenizer.from_pretrained(BASE)
tokenizer.padding_side = "left"


def carregador(df, embaralhar):
    pares = list(zip(df["texto"], df["rotulo_id"]))

    def juntar(lote):
        textos, rotulos = zip(*lote)
        enc = tokenizer(list(textos), truncation=True, max_length=MAX_TOKENS, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(rotulos)
        return enc

    return DataLoader(pares, batch_size=LOTE, shuffle=embaralhar, collate_fn=juntar)


modelo = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=len(NOMES), id2label=dict(enumerate(NOMES)), label2id=ID, torch_dtype=torch.float32)
modelo.config.pad_token_id = tokenizer.pad_token_id
modelo = get_peft_model(modelo, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    modules_to_save=["score"], task_type="SEQ_CLS")).to(dispositivo)
modelo.print_trainable_parameters()

contagem = treino_lora["rotulo_id"].value_counts().reindex(range(len(NOMES)), fill_value=1)
pesos = torch.tensor((len(treino_lora) / (len(NOMES) * contagem)).values, dtype=torch.float32).clamp(max=5.0)
perda_fn = torch.nn.CrossEntropyLoss(weight=pesos.to(dispositivo))


@torch.no_grad()
def prever(df):
    modelo.eval()
    saida = []
    for lote in carregador(df, False):
        lote = {k: v.to(dispositivo) for k, v in lote.items() if k != "labels"}
        with torch.autocast("cuda", dtype=torch.float16):
            saida += modelo(**lote).logits.argmax(-1).tolist()
    return np.array(saida)


dados = carregador(treino_lora, True)
otimizador = torch.optim.AdamW([p for p in modelo.parameters() if p.requires_grad], lr=TAXA, weight_decay=0.01)
agenda = get_cosine_schedule_with_warmup(otimizador, int(0.06 * EPOCAS * len(dados)), EPOCAS * len(dados))
escala = torch.amp.GradScaler()
historico = []

for epoca in range(EPOCAS):
    modelo.train()
    soma = 0.0
    for lote in dados:
        lote = {k: v.to(dispositivo) for k, v in lote.items()}
        rotulos = lote.pop("labels")
        with torch.autocast("cuda", dtype=torch.float16):
            perda = perda_fn(modelo(**lote).logits.float(), rotulos)
        escala.scale(perda).backward()
        escala.unscale_(otimizador)
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        escala.step(otimizador)
        escala.update()
        agenda.step()
        otimizador.zero_grad()
        soma += perda.item()
    acc_val = (prever(validacao) == validacao["rotulo_id"].values).mean()
    historico.append({"epoca": epoca + 1, "perda_treino": soma / len(dados), "acuracia_validacao": acc_val})
    print(historico[-1])

inicio = time.perf_counter()
previsto = [NOMES[i] for i in prever(teste)]
registrar("qwen0.5b_lora", previsto, (time.perf_counter() - inicio) * 1000 / len(teste), {"parametros": "0,5 B + 1% LoRA"})

modelo.save_pretrained(SAIDA / "qwen0.5b-lora-motivos")
tokenizer.save_pretrained(SAIDA / "qwen0.5b-lora-motivos")
(SAIDA / "historico_lora.json").write_text(json.dumps(historico, indent=2, default=float))

# %% [markdown]
# ## 6. Comparação

# %%
tabela = pd.DataFrame(RESULTADOS).set_index("modelo").round(3)
tabela.to_csv(SAIDA / "comparacao.csv")
tabela

# %% [markdown]
# ## 7. Conferência externa: o "atraso" bate com a data real de entrega?
#
# A base tem a data prometida e a data em que o pedido chegou. O modelo nunca viu essas
# colunas. Se ele aprendeu algo real, avaliações que ele chama de `atraso` devem ter muito
# mais pedidos atrasados de verdade do que as que ele chama de `elogio`.

# %%
amostra = pool[~pool["id"].isin(treino["id"])].sample(n=3000, random_state=SEMENTE)
for coluna in ["chegou_atrasado", "entregue"]:
    amostra[coluna] = amostra[coluna].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
amostra["previsto"] = [NOMES[i] for i in prever(amostra.assign(rotulo_id=0))]
(amostra.groupby("previsto")
        .agg(avaliacoes=("id", "size"), chegou_atrasado=("chegou_atrasado", "mean"), entregue=("entregue", "mean"))
        .round(2).sort_values("chegou_atrasado", ascending=False))

# %% [markdown]
# ## 8. Erros que valem olhar
#
# Onde o modelo com LoRA erra no teste, lado a lado com o rótulo manual.

# %%
teste["lora"] = previsto
erros = teste[teste["lora"] != teste["rotulo"]][["texto", "rotulo", "lora"]]
print(len(erros), "erros em", len(teste))
erros.head(25)

# %% [markdown]
# ## 9. Publicação no Hugging Face
#
# Publica o adaptador LoRA e os dados (teste rotulado à mão e treino rotulado pelo professor) como
# repositórios **privados** no Hugging Face. Revise lá e torne públicos quando estiverem bons.
#
# Antes de rodar: crie um token com permissão de escrita em huggingface.co/settings/tokens e
# guarde-o no Kaggle em **Add-ons → Secrets** com o nome `HF_TOKEN`.

# %%
from huggingface_hub import HfApi
from kaggle_secrets import UserSecretsClient

# lê tudo do disco: adicionar um segredo no Kaggle reinicia a sessão e apaga as variáveis
tabela = pd.read_csv(SAIDA / "comparacao.csv", index_col="modelo")
treino = pd.read_csv(SAIDA / "treino_professor.csv")
api = HfApi(token=UserSecretsClient().get_secret("HF_TOKEN"))
usuario = api.whoami()["name"]
REPO_MODELO = f"{usuario}/qwen2.5-0.5b-motivos-avaliacoes"
REPO_DADOS = f"{usuario}/avaliacoes-entrega-ptbr"
COLUNAS = ["acuracia", "f1_macro", "f1_macro_reclamacoes", "ms_por_avaliacao"]
tabela_md = "| modelo | " + " | ".join(COLUNAS) + " |\n|---" * (len(COLUNAS) + 1) + "|\n" + "\n".join(
    f"| {nome} | " + " | ".join(f"{linha[c]:.3f}" if c != "ms_por_avaliacao" else f"{linha[c]:.1f}" for c in COLUNAS) + " |"
    for nome, linha in tabela.iterrows())

cartao_modelo = f"""---
base_model: Qwen/Qwen2.5-0.5B
library_name: peft
language: [pt]
license: cc-by-nc-sa-4.0
pipeline_tag: text-classification
datasets: [{REPO_DADOS}]
tags: [lora, distillation, portuguese, e-commerce, reviews]
---

# Qwen2.5-0.5B + LoRA: motivo de reclamação em avaliações de compras

Classifica avaliações de compras online em português em 8 motivos: {", ".join(NOMES)}.

Treinado por **destilação**: o Qwen2.5-7B-Instruct rotulou {len(treino)} avaliações e este modelo,
14 vezes menor, aprendeu com esses rótulos. Avaliado em 500 avaliações rotuladas à mão.

## Resultados no teste (500 avaliações, rótulos manuais)

A métrica principal é o F1 macro das 7 classes de reclamação: acertar *qual* é o problema.

{tabela_md}

## Uso

```python
import torch
from peft import AutoPeftModelForSequenceClassification
from transformers import AutoTokenizer

repo = "{REPO_MODELO}"
tokenizer = AutoTokenizer.from_pretrained(repo)
tokenizer.padding_side = "left"  # o classificador lê o último token
modelo = AutoPeftModelForSequenceClassification.from_pretrained(repo).eval()
enc = tokenizer(["Comprei dois e só chegou um"], return_tensors="pt", padding=True)
with torch.no_grad():
    print(modelo.config.id2label[modelo(**enc).logits.argmax(-1).item()])
```

## Dados e licença

Avaliações públicas da Olist (CC BY-NC-SA 4.0); por isso o adaptador segue a mesma licença, sem
uso comercial. Código, notebook e decisões: github.com/Cecili4na/avaliacoes-entrega-ptbr
"""

cartao_dados = f"""---
language: [pt]
license: cc-by-nc-sa-4.0
task_categories: [text-classification]
pretty_name: Motivos de reclamação em avaliações de compras (pt-BR)
configs:
- config_name: default
  data_files:
  - split: train
    path: treino.csv
  - split: test
    path: teste.csv
---

# Motivos de reclamação em avaliações de compras (pt-BR)

Avaliações da Olist com o motivo principal: {", ".join(NOMES)}.

- **test**: 500 avaliações rotuladas à mão, estratificadas pela nota (quase 60% são elogios).
- **train**: {len(treino)} avaliações rotuladas pelo Qwen2.5-7B-Instruct (coluna `rotulo_professor`),
  dois terços com nota de 1 a 3. Rótulos de modelo, com ruído: veja a concordância do professor
  com o teste no cartão do modelo {REPO_MODELO}.

As colunas `entregue` e `chegou_atrasado` vêm dos dados do pedido e servem de conferência externa.

Fonte: Brazilian E-Commerce Public Dataset by Olist, CC BY-NC-SA 4.0.
Código: github.com/Cecili4na/avaliacoes-entrega-ptbr
"""

api.create_repo(REPO_MODELO, private=True, exist_ok=True)
api.upload_folder(repo_id=REPO_MODELO, folder_path=SAIDA / "qwen0.5b-lora-motivos")
for arquivo in ["comparacao.csv", "historico_lora.json", "confusao_qwen0.5b_lora.png"]:
    api.upload_file(repo_id=REPO_MODELO, path_or_fileobj=SAIDA / arquivo, path_in_repo=arquivo)
api.upload_file(repo_id=REPO_MODELO, path_or_fileobj=cartao_modelo.encode(), path_in_repo="README.md")

api.create_repo(REPO_DADOS, repo_type="dataset", private=True, exist_ok=True)
api.upload_file(repo_id=REPO_DADOS, repo_type="dataset", path_or_fileobj=ENTRADA / "teste.csv", path_in_repo="teste.csv")
api.upload_file(repo_id=REPO_DADOS, repo_type="dataset", path_or_fileobj=SAIDA / "treino_professor.csv", path_in_repo="treino.csv")
api.upload_file(repo_id=REPO_DADOS, repo_type="dataset", path_or_fileobj=cartao_dados.encode(), path_in_repo="README.md")
print(f"https://huggingface.co/{REPO_MODELO}\nhttps://huggingface.co/datasets/{REPO_DADOS}")
