"""Conferências do teste rotulado à mão: se ele quebrar, todos os números do README perdem o sentido."""

from pathlib import Path

import pandas as pd
import pytest

from preparar import normalizar
from rotulos import ID, NOMES, descricao_para_prompt

DADOS = Path(__file__).resolve().parents[1] / "data" / "processed"


@pytest.fixture(scope="module")
def teste():
    return pd.read_csv(DADOS / "teste.csv")


def test_oito_rotulos_com_ids_na_ordem():
    assert len(NOMES) == 8
    assert [ID[n] for n in NOMES] == list(range(8))
    assert all(f"- {n}:" in descricao_para_prompt() for n in NOMES)


def test_teste_tem_500_avaliacoes_todas_com_rotulo_valido(teste):
    assert len(teste) == 500
    assert teste["id"].is_unique
    assert set(teste["rotulo"]) <= set(NOMES)


def test_proporcoes_citadas_no_readme(teste):
    # "responder elogio para tudo acerta 59%" e "só 207 reclamações no teste"
    assert round((teste["rotulo"] == "elogio").mean(), 2) == 0.59
    assert (teste["rotulo"] != "elogio").sum() == 207


def test_nenhum_texto_do_teste_aparece_no_treino(teste):
    pool = pd.read_csv(DADOS / "pool.csv", usecols=["id", "texto"])
    assert not set(teste["id"]) & set(pool["id"])
    assert not set(teste["texto"].map(normalizar)) & set(pool["texto"].map(normalizar))


def test_normalizar_ignora_caixa_e_espacos():
    assert normalizar("  Recebi   NO prazo ") == "recebi no prazo"
