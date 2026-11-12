"""Métricas comuns a todos os modelos, sempre contra o teste rotulado à mão."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)

from rotulos import NOMES

RAIZ = Path(__file__).resolve().parents[1]
RESULTADOS = RAIZ / "resultados"


def registrar(nome: str, verdade, previsto, ms_por_avaliacao: float, extra: dict | None = None) -> dict:
    """Salva métricas e matriz de confusão do modelo `nome` e devolve o resumo."""
    RESULTADOS.mkdir(exist_ok=True)
    resumo = {
        "modelo": nome,
        "acuracia": round(accuracy_score(verdade, previsto), 4),
        "f1_macro": round(f1_score(verdade, previsto, labels=NOMES, average="macro", zero_division=0), 4),
        # sem "elogio": mede só a parte difícil, separar um tipo de reclamação do outro
        "f1_macro_reclamacoes": round(
            f1_score(verdade, previsto, labels=[n for n in NOMES if n != "elogio"], average="macro", zero_division=0),
            4,
        ),
        "ms_por_avaliacao": round(ms_por_avaliacao, 1),
        **(extra or {}),
    }
    relatorio = classification_report(verdade, previsto, labels=NOMES, zero_division=0, output_dict=True)
    (RESULTADOS / f"{nome}.json").write_text(
        json.dumps({"resumo": resumo, "por_classe": relatorio}, ensure_ascii=False, indent=2)
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        verdade, previsto, labels=NOMES, ax=ax, xticks_rotation=45, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"{nome} · F1 macro {resumo['f1_macro']:.2f}")
    fig.tight_layout()
    fig.savefig(RESULTADOS / f"confusao_{nome}.png", dpi=120)
    plt.close(fig)

    print(json.dumps(resumo, ensure_ascii=False))
    return resumo
