"""
ETAPA 3 — PRÉ-PROCESSAMENTO + ETAPA 4 — TREINAMENTO
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from sklearn.inspection import permutation_importance
import joblib, warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 3. PRÉ-PROCESSAMENTO
# ──────────────────────────────────────────────
print("=" * 60)
print("⚙️  PRÉ-PROCESSAMENTO")
print("=" * 60)

df = pd.read_csv("data/clientes_telemax.csv")

# Engenharia de features
df["receita_por_produto"] = df["charge_mensal"] / df["num_produtos"]
df["razao_suporte_tenure"] = df["chamadas_suporte"] / (df["tempo_cliente_meses"] + 1)
df["cliente_novo"] = (df["tempo_cliente_meses"] <= 6).astype(int)

features_num = [
    "idade", "tempo_cliente_meses", "charge_mensal", "total_cobrado",
    "num_produtos", "tem_internet", "tem_fone", "chamadas_suporte",
    "receita_por_produto", "razao_suporte_tenure", "cliente_novo"
]
features_cat = ["contrato", "forma_pagamento"]
TARGET = "churn"

X = df[features_num + features_cat]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print(f"  Churn no treino: {y_train.mean():.1%}")

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), features_num),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), features_cat),
])

# ──────────────────────────────────────────────
# 4. TREINAMENTO — 3 modelos
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("🤖 TREINAMENTO DE MODELOS")
print("=" * 60)

modelos = {
    "Regressão Logística": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=42),
}

resultados = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for nome, modelo in modelos.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", modelo)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    ap  = average_precision_score(y_test, y_prob)

    resultados[nome] = {"pipe": pipe, "auc": auc, "ap": ap, "cv_mean": scores.mean(), "cv_std": scores.std()}
    print(f"\n  [{nome}]")
    print(f"    CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"    Test AUC: {auc:.4f} | Avg Precision: {ap:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Não-Churn","Churn"]))

# Melhor modelo
melhor_nome = max(resultados, key=lambda k: resultados[k]["auc"])
melhor_pipe  = resultados[melhor_nome]["pipe"]
print(f"\n🏆 Melhor modelo: {melhor_nome} (AUC={resultados[melhor_nome]['auc']:.4f})")

joblib.dump(melhor_pipe, "models/modelo_churn.pkl")
print("✅ Modelo salvo em models/modelo_churn.pkl")

# Feature importance (RF)
if "Random Forest" in melhor_nome:
    feat_names = features_num + list(
        melhor_pipe.named_steps["prep"]
            .named_transformers_["cat"]
            .get_feature_names_out(features_cat)
    )
    imp = melhor_pipe.named_steps["clf"].feature_importances_
    fi = pd.Series(imp, index=feat_names).sort_values(ascending=False).head(10)
    print("\n🔍 Top 10 Features Mais Importantes:")
    print(fi.round(4))
