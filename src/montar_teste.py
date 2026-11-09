"""Junta os lotes de anotação manual ao conjunto de teste e confere se não falta nada."""

from pathlib import Path

import pandas as pd

from rotulos import NOMES

RAIZ = Path(__file__).resolve().parents[1]


def main() -> None:
    teste = pd.read_csv(RAIZ / "data" / "processed" / "teste_sem_rotulo.csv")
    rotulos = {}
    for lote in sorted((RAIZ / "data" / "anotacao").glob("lote*.txt")):
        for linha in lote.read_text().splitlines():
            i, rotulo = linha.split()
            assert rotulo in NOMES, f"{lote.name}: rótulo desconhecido {rotulo}"
            rotulos[int(i)] = rotulo
    faltando = set(range(len(teste))) - set(rotulos)
    assert not faltando, f"sem rótulo: {sorted(faltando)[:10]}"
    teste["rotulo"] = teste.index.map(rotulos)
    teste.to_csv(RAIZ / "data" / "processed" / "teste.csv", index=False)
    print(teste["rotulo"].value_counts().to_string())


if __name__ == "__main__":
    main()
