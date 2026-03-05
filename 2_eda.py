"""
ETAPA 2 — ANÁLISE EXPLORATÓRIA (EDA)
"""

import pandas as pd
import numpy as np

df = pd.read_csv("data/clientes_telemax.csv")

print("=" * 60)
print("📊 ANÁLISE EXPLORATÓRIA — TeleMax Churn")
print("=" * 60)

print(f"\n🔢 Shape: {df.shape}")
print(f"\n📋 Tipos:\n{df.dtypes}")
print(f"\n🚨 Nulos:\n{df.isnull().sum()}")

print("\n📈 Estatísticas Descritivas:")
print(df.describe().round(2))

print(f"\n🎯 Distribuição do Churn:")
vc = df["churn"].value_counts()
print(f"   Não-churn (0): {vc[0]} ({vc[0]/len(df):.1%})")
print(f"   Churn (1):     {vc[1]} ({vc[1]/len(df):.1%})")

print("\n📌 Correlações com Churn:")
numericas = ["idade","tempo_cliente_meses","charge_mensal","total_cobrado",
             "num_produtos","tem_internet","tem_fone","chamadas_suporte"]
corr = df[numericas + ["churn"]].corr()["churn"].drop("churn").sort_values()
print(corr.round(3))

print("\n📦 Churn por tipo de contrato:")
print(df.groupby("contrato")["churn"].mean().round(3))

print("\n💳 Churn por forma de pagamento:")
print(df.groupby("forma_pagamento")["churn"].mean().round(3))

print("\n✅ EDA concluída!")
