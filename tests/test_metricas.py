import metricas


def test_f1_das_reclamacoes_ignora_elogio(tmp_path, monkeypatch):
    monkeypatch.setattr(metricas, "RESULTADOS", tmp_path)
    verdade = ["elogio", "elogio", "atraso", "nao_recebido"]
    # acerta os elogios e troca as duas reclamações entre si
    previsto = ["elogio", "elogio", "nao_recebido", "atraso"]
    resumo = metricas.registrar("t", verdade, previsto, ms_por_avaliacao=1.0)
    assert resumo["acuracia"] == 0.5
    assert resumo["f1_macro_reclamacoes"] == 0.0
    assert (tmp_path / "t.json").exists() and (tmp_path / "confusao_t.png").exists()
