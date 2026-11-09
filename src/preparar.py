"""Monta a base: avaliações com comentário, sem duplicatas, com sinais reais de entrega.

Os sinais de entrega (status do pedido e se chegou depois da data prometida) vêm da
tabela de pedidos e não entram no modelo. Servem só para conferir, de fora, se os rótulos
de atraso e de não recebimento fazem sentido.

Separação: 500 avaliações para o conjunto de teste (rotulado à mão), o resto para treino
e validação. A separação é por texto normalizado, para o mesmo comentário não cair nos dois.
"""

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
BRUTO = RAIZ / "data" / "raw"
SAIDA = RAIZ / "data" / "processed"
SEMENTE = 42
N_TESTE = 500


def normalizar(texto: str) -> str:
    return " ".join(texto.lower().split())


def main() -> None:
    avaliacoes = pd.read_csv(BRUTO / "olist_order_reviews_dataset.csv")
    pedidos = pd.read_csv(
        BRUTO / "olist_orders_dataset.csv",
        parse_dates=["order_delivered_customer_date", "order_estimated_delivery_date"],
    )

    df = avaliacoes.dropna(subset=["review_comment_message"]).copy()
    df["texto"] = (
        df["review_comment_title"].fillna("").str.strip() + ". " + df["review_comment_message"].str.strip()
    ).str.strip(". ").str.replace(r"\s+", " ", regex=True)
    df = df[df["texto"].str.len() >= 3]

    df["chave"] = df["texto"].map(normalizar)
    df = df.drop_duplicates("chave")

    pedidos["entregue"] = pedidos["order_status"].eq("delivered")
    pedidos["chegou_atrasado"] = (
        pedidos["order_delivered_customer_date"] > pedidos["order_estimated_delivery_date"]
    )
    df = df.merge(pedidos[["order_id", "entregue", "chegou_atrasado"]], on="order_id", how="left")

    df = df.rename(columns={"review_id": "id", "review_score": "nota"})
    df = df[["id", "texto", "nota", "entregue", "chegou_atrasado"]]

    # teste estratificado pela nota, para ter reclamações e elogios na proporção real
    teste = df.groupby("nota").sample(frac=N_TESTE / len(df), random_state=SEMENTE)
    resto = df.drop(teste.index).sample(frac=1, random_state=SEMENTE)

    SAIDA.mkdir(parents=True, exist_ok=True)
    teste.to_csv(SAIDA / "teste_sem_rotulo.csv", index=False)
    resto.to_csv(SAIDA / "pool.csv", index=False)
    print(f"total {len(df)} · teste {len(teste)} · pool {len(resto)}")
    print(teste["nota"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
