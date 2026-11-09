"""Taxonomia de motivos: o que a pessoa está dizendo na avaliação.

Um rótulo por avaliação, o motivo principal. Quando há mais de um problema, vale o que
a pessoa mais enfatiza; empate fica com o que aparece primeiro na lista abaixo.
"""

ROTULOS = {
    "nao_recebido": (
        "No momento da avaliação, nada do pedido chegou: diz que não recebeu, que está esperando, "
        "que extraviou ou que consta entregue sem ter chegado."
    ),
    "atraso": (
        "Reclama de demora ou de passar do prazo sem dizer que ainda não recebeu "
        "(chegou demorado, ou só fala em atraso)."
    ),
    "incompleto": (
        "Chegou só parte: faltou item, unidade, peça ou acessório, ou um dos produtos do pedido não veio."
    ),
    "produto_errado": (
        "Chegou algo diferente do comprado ou do anunciado: outro modelo, cor, tamanho, capacidade, "
        "réplica ou falsificado."
    ),
    "defeito_qualidade": (
        "O produto certo chegou, mas com defeito, quebrado, amassado, sujo, não funciona "
        "ou é de qualidade ruim."
    ),
    "atendimento_cobranca": (
        "Atendimento, falta de resposta ou de informação, cancelamento, falta de estoque, troca, "
        "estorno, nota fiscal ou cobrança."
    ),
    "elogio": "Avaliação positiva, sem reclamação que pese mais que o elogio.",
    "outro": (
        "Nenhum dos anteriores: embalagem, retirada nos Correios, tentativa de entrega, "
        "crítica genérica ou texto sem conteúdo."
    ),
}

NOMES = list(ROTULOS)
ID = {nome: i for i, nome in enumerate(NOMES)}


def descricao_para_prompt() -> str:
    return "\n".join(f"- {nome}: {desc}" for nome, desc in ROTULOS.items())
