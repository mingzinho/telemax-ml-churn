"""
ETAPA 5 — DEPLOY EM PRODUÇÃO COM FastAPI
Execute: uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Literal
import joblib
import pandas as pd
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("telemax-churn-api")

# ──────────────────────────────────────────────
# Carrega modelo
# ──────────────────────────────────────────────
try:
    modelo = joblib.load("models/modelo_churn.pkl")
    logger.info("✅ Modelo carregado com sucesso")
except Exception as e:
    logger.error(f"❌ Falha ao carregar modelo: {e}")
    raise

# ──────────────────────────────────────────────
# Schemas de entrada / saída
# ──────────────────────────────────────────────
class ClienteInput(BaseModel):
    cliente_id: str = Field(..., example="CLI-00001")
    idade: int = Field(..., ge=18, le=100, example=35)
    tempo_cliente_meses: int = Field(..., ge=1, example=24)
    contrato: Literal["mensal", "anual", "bienal"] = Field(..., example="mensal")
    forma_pagamento: Literal["cartao_credito", "debito_automatico", "boleto"] = Field(..., example="boleto")
    charge_mensal: float = Field(..., gt=0, example=85.50)
    total_cobrado: float = Field(..., gt=0, example=2052.00)
    num_produtos: int = Field(..., ge=1, le=10, example=2)
    tem_internet: int = Field(..., ge=0, le=1, example=1)
    tem_fone: int = Field(..., ge=0, le=1, example=1)
    chamadas_suporte: int = Field(..., ge=0, example=3)

class PredictionOutput(BaseModel):
    cliente_id: str
    churn_probabilidade: float
    churn_predicao: bool
    risco: Literal["BAIXO", "MÉDIO", "ALTO"]
    recomendacao: str
    tempo_ms: float

class HealthOutput(BaseModel):
    status: str
    modelo: str
    versao: str

# ──────────────────────────────────────────────
# Função de engenharia de features
# ──────────────────────────────────────────────
def preparar_features(dados: ClienteInput) -> pd.DataFrame:
    d = dados.dict()
    receita_por_produto = d["charge_mensal"] / d["num_produtos"]
    razao_suporte_tenure = d["chamadas_suporte"] / (d["tempo_cliente_meses"] + 1)
    cliente_novo = int(d["tempo_cliente_meses"] <= 6)

    return pd.DataFrame([{
        "idade": d["idade"],
        "tempo_cliente_meses": d["tempo_cliente_meses"],
        "charge_mensal": d["charge_mensal"],
        "total_cobrado": d["total_cobrado"],
        "num_produtos": d["num_produtos"],
        "tem_internet": d["tem_internet"],
        "tem_fone": d["tem_fone"],
        "chamadas_suporte": d["chamadas_suporte"],
        "receita_por_produto": receita_por_produto,
        "razao_suporte_tenure": razao_suporte_tenure,
        "cliente_novo": cliente_novo,
        "contrato": d["contrato"],
        "forma_pagamento": d["forma_pagamento"],
    }])

def classificar_risco(prob: float) -> tuple[str, str]:
    if prob < 0.30:
        return "BAIXO", "Cliente saudável. Manter relacionamento com programa de fidelidade."
    elif prob < 0.60:
        return "MÉDIO", "Atenção recomendada. Acionar equipe de retenção proativamente."
    else:
        return "ALTO", "⚠️ Risco crítico! Oferecer desconto ou upgrade imediatamente."

# ──────────────────────────────────────────────
# App FastAPI
# ──────────────────────────────────────────────
app = FastAPI(
    title="TeleMax Churn Prediction API",
    description="API de predição de churn de clientes para a TeleMax",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthOutput)
def health():
    return {"status": "ok", "modelo": "GradientBoostingClassifier", "versao": "1.0.0"}

@app.post("/predict", response_model=PredictionOutput)
def predict(cliente: ClienteInput):
    t0 = time.time()
    try:
        X = preparar_features(cliente)
        prob = float(modelo.predict_proba(X)[0, 1])
        predicao = prob >= 0.5
        risco, recomendacao = classificar_risco(prob)
        elapsed = (time.time() - t0) * 1000

        logger.info(f"Predição: {cliente.cliente_id} → prob={prob:.3f} risco={risco}")

        return PredictionOutput(
            cliente_id=cliente.cliente_id,
            churn_probabilidade=round(prob, 4),
            churn_predicao=predicao,
            risco=risco,
            recomendacao=recomendacao,
            tempo_ms=round(elapsed, 2),
        )
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=list[PredictionOutput])
def predict_batch(clientes: list[ClienteInput]):
    if len(clientes) > 1000:
        raise HTTPException(400, "Máximo 1000 clientes por batch")
    return [predict(c) for c in clientes]
