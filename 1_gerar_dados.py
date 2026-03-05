"""
ETAPA 1 — GERAÇÃO DE DADOS FICTÍCIOS
Problema: Prever churn (cancelamento) de clientes de uma telecom fictícia — TeleMax.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 5000

def gerar_dataset(n=N):
    ids = [f"CLI-{str(i).zfill(5)}" for i in range(1, n+1)]
    idade = np.random.randint(18, 75, n)
    tenure = np.random.randint(1, 72, n)

    contrato = np.random.choice(
        ["mensal", "anual", "bienal"],
        p=[0.55, 0.30, 0.15],
        size=n
    )
    contrato_enc = {"mensal": 0, "anual": 1, "bienal": 2}
    contrato_num = np.array([contrato_enc[c] for c in contrato])

    pagamento = np.random.choice(
        ["cartao_credito", "debito_automatico", "boleto"],
        p=[0.35, 0.40, 0.25],
        size=n
    )

    charge_base = np.random.normal(70, 20, n).clip(20, 150)
    num_produtos = np.random.randint(1, 6, n)
    charge_mensal = (charge_base + num_produtos * 8).round(2)
    total_cobrado = (charge_mensal * tenure * np.random.uniform(0.9, 1.1, n)).round(2)

    tem_internet = np.random.choice([0, 1], p=[0.15, 0.85], size=n)
    tem_fone = np.random.choice([0, 1], p=[0.10, 0.90], size=n)
    suporte_chamadas = np.random.poisson(lam=2, size=n)

    # Churn probabilístico baseado em features realistas
    logit = (
        -2.5
        + 0.02 * suporte_chamadas
        - 0.03 * tenure
        + 0.01 * charge_mensal
        - 0.8 * contrato_num
        + 0.3 * (pagamento == "boleto").astype(int)
        - 0.2 * num_produtos
        + np.random.normal(0, 0.3, n)
    )
    prob_churn = 1 / (1 + np.exp(-logit))
    churn = (np.random.uniform(0, 1, n) < prob_churn).astype(int)

    df = pd.DataFrame({
        "cliente_id": ids,
        "idade": idade,
        "tempo_cliente_meses": tenure,
        "contrato": contrato,
        "forma_pagamento": pagamento,
        "charge_mensal": charge_mensal,
        "total_cobrado": total_cobrado,
        "num_produtos": num_produtos,
        "tem_internet": tem_internet,
        "tem_fone": tem_fone,
        "chamadas_suporte": suporte_chamadas,
        "churn": churn,
    })

    return df

if __name__ == "__main__":
    df = gerar_dataset()
    df.to_csv("data/clientes_telemax.csv", index=False)
    print(f"✅ Dataset gerado: {len(df)} registros")
    print(f"   Taxa de churn: {df['churn'].mean():.1%}")
    print(f"   Colunas: {list(df.columns)}")
    print(df.head())
