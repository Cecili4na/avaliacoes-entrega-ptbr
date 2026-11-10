"""Demo do classificador no Hugging Face Spaces (CPU gratuita).

O adaptador LoRA é lido do repositório de modelo do mesmo usuário do Space. Se o modelo ainda
estiver privado, adicione HF_TOKEN nos segredos do Space.
"""

import os

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROTULOS = ["nao_recebido", "atraso", "incompleto", "produto_errado",
           "defeito_qualidade", "atendimento_cobranca", "elogio", "outro"]
LEGENDAS = {
    "nao_recebido": "Não recebido",
    "atraso": "Atraso",
    "incompleto": "Pedido incompleto",
    "produto_errado": "Produto errado",
    "defeito_qualidade": "Defeito ou qualidade",
    "atendimento_cobranca": "Atendimento ou cobrança",
    "elogio": "Elogio",
    "outro": "Outro",
}
REPO = os.environ.get("MODELO", f"{os.environ.get('SPACE_AUTHOR_NAME', 'Cecili4na')}/qwen2.5-0.5b-motivos-avaliacoes")

tokenizer = AutoTokenizer.from_pretrained(REPO)
tokenizer.padding_side = "left"
base = AutoModelForSequenceClassification.from_pretrained("Qwen/Qwen2.5-0.5B", num_labels=len(ROTULOS))
base.config.pad_token_id = tokenizer.pad_token_id
modelo = PeftModel.from_pretrained(base, REPO).eval()


@torch.no_grad()
def classificar(texto: str) -> dict[str, float]:
    if not texto.strip():
        return {}
    enc = tokenizer([texto], truncation=True, max_length=128, return_tensors="pt")
    probabilidades = modelo(**enc).logits.softmax(-1)[0].tolist()
    return {LEGENDAS[r]: p for r, p in zip(ROTULOS, probabilidades)}


gr.Interface(
    fn=classificar,
    inputs=gr.Textbox(label="Avaliação do cliente", lines=4),
    outputs=gr.Label(label="Motivo", num_top_classes=3),
    title="Qual é o problema da entrega?",
    description=(
        "Classifica o motivo de uma avaliação de compra online em português. Qwen2.5-0.5B com LoRA, "
        "treinado com rótulos de um modelo 14 vezes maior. F1 macro de 0,60 nas reclamações, "
        "contra 0,70 do modelo grande."
    ),
    examples=[
        ["Comprei dois kits e só chegou um, a nota fiscal veio com os dois."],
        ["Já passou 10 dias do prazo e até agora nada do meu pedido."],
        ["Chegou rápido, mas a tela veio trincada e não liga."],
        ["Pedi azul e veio preto, tamanho também errado."],
        ["Produto excelente, chegou antes do prazo. Recomendo!"],
    ],
).launch()
