"""Baseline clássico: TF-IDF de palavras e de pedaços de palavras + regressão logística.

É o ponto de partida honesto. Se um modelo de linguagem não ganhar disto com folga,
não vale o custo de rodá-lo.
"""

import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union

from metricas import registrar

RAIZ = Path(__file__).resolve().parents[1]
PROCESSADO = RAIZ / "data" / "processed"


def main() -> None:
    treino = pd.read_csv(PROCESSADO / "treino_professor.csv")
    teste = pd.read_csv(PROCESSADO / "teste.csv")

    modelo = make_pipeline(
        make_union(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode"),
            # pedaços de 3 a 5 letras pegam erros de digitação, comuns nas avaliações
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True),
        ),
        LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"),
    )
    modelo.fit(treino["texto"], treino["rotulo_professor"])

    inicio = time.perf_counter()
    previsto = modelo.predict(teste["texto"])
    ms = (time.perf_counter() - inicio) * 1000 / len(teste)
    registrar("tfidf_logreg", teste["rotulo"], previsto, ms, {"parametros": "~1 MB (vocabulário + pesos)"})


if __name__ == "__main__":
    main()
