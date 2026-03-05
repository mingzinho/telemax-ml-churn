# TeleMax Churn Prediction — Pipeline de Machine Learning

> Sistema completo de machine learning para predição proativa de churn em uma operadora de telecomunicações, cobrindo geração de dados, análise exploratória, engenharia de features, treinamento, avaliação e deploy em produção via API REST.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![Railway](https://img.shields.io/badge/Deploy-Railway-0B0D0E?logo=railway&logoColor=white)](https://railway.app)

---

## Sumário

- [Problema de Negócio](#problema-de-negócio)
- [Visão Geral da Solução](#visão-geral-da-solução)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
  - [1. Geração de Dados](#1-geração-de-dados)
  - [2. Análise Exploratória](#2-análise-exploratória)
  - [3. Pré-processamento e Engenharia de Features](#3-pré-processamento-e-engenharia-de-features)
  - [4. Treinamento e Seleção de Modelos](#4-treinamento-e-seleção-de-modelos)
  - [5. Avaliação](#5-avaliação)
  - [6. Deploy](#6-deploy)
- [Referência da API](#referência-da-api)
- [Executando Localmente](#executando-localmente)
- [Guia de Deploy](#guia-de-deploy)
- [Resultados e Impacto de Negócio](#resultados-e-impacto-de-negócio)
- [Stack Tecnológica](#stack-tecnológica)

---

## Problema de Negócio

A TeleMax perde aproximadamente **27% da sua base de clientes por ano** para o churn. O custo de aquisição de um novo cliente (~R$ 450) é muito superior ao custo de retenção (~R$ 80), tornando a prevenção proativa de cancelamentos uma iniciativa de alto retorno.

**Objetivo:** construir um classificador binário capaz de prever, com recall mínimo de 70%, quais clientes têm propensão a cancelar nos próximos 30 dias — permitindo que a equipe de retenção aja antes do cancelamento.

| Indicador | Valor |
|---|---|
| Taxa anual de churn | ~27% |
| Custo de aquisição de cliente | R$ 450 |
| Custo de ação de retenção | R$ 80 |
| Economia estimada (base de 5k clientes) | R$ 1,85M/ano |
| ROI da campanha (threshold 0,50) | ~16,5× |

---

## Visão Geral da Solução

```
Dados Brutos → EDA → Engenharia de Features → Treinamento → Avaliação → FastAPI → Docker → Railway
```

O modelo vencedor é um **Gradient Boosting Classifier** encapsulado em um pipeline `sklearn` reprodutível, serializado com `joblib` e servido por uma aplicação FastAPI tipada e containerizada com Docker.

---

## Estrutura do Projeto

```
ml_churn/
├── data/
│   └── clientes_telemax.csv       # Dataset gerado (5.000 registros)
├── models/
│   └── modelo_churn.pkl           # Pipeline sklearn serializado
├── api/
│   ├── __init__.py
│   └── app.py                     # Aplicação FastAPI
├── 1_gerar_dados.py               # Geração de dados sintéticos
├── 2_eda.py                       # Análise exploratória
├── 3_treinar.py                   # Pré-processamento, treinamento e avaliação
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Dataset

O dataset foi gerado sinteticamente para espelhar datasets reais de churn em telecomunicações (ex.: IBM Telco). Os rótulos de churn são produzidos via função logística com coeficientes calibrados e ruído gaussiano, garantindo distribuições de classe e correlações de features realistas.

**5.000 registros · 11 features brutas · 27,3% de taxa de churn**

| Feature | Tipo | Descrição |
|---|---|---|
| `cliente_id` | string | Identificador único do cliente |
| `idade` | int | Idade do cliente (18–74 anos) |
| `tempo_cliente_meses` | int | Tempo de relacionamento em meses |
| `contrato` | categórica | `mensal` / `anual` / `bienal` |
| `forma_pagamento` | categórica | `cartao_credito` / `debito_automatico` / `boleto` |
| `charge_mensal` | float | Valor da fatura mensal (R$) |
| `total_cobrado` | float | Total acumulado cobrado (R$) |
| `num_produtos` | int | Quantidade de produtos contratados |
| `tem_internet` | binária | Possui plano de internet |
| `tem_fone` | binária | Possui plano de telefone |
| `chamadas_suporte` | int | Chamados de suporte no período |
| `churn` *(alvo)* | binária | `1` = cancelou, `0` = permaneceu |

---

## Pipeline

### 1. Geração de Dados

**Script:** `1_gerar_dados.py`

O comportamento dos clientes é simulado por um modelo logístico com coeficientes orientados ao domínio:

```python
logit = (
    -2.5
    + 0.02 * chamadas_suporte   # alto volume de suporte aumenta o risco de churn
    - 0.03 * tenure             # maior tempo de cliente reduz o risco de churn
    + 0.01 * charge_mensal      # fatura mais alta aumenta o risco de churn
    - 0.8  * contrato_num       # contratos mais longos reduzem fortemente o risco
    + np.random.normal(0, 0.3)  # ruído gaussiano para realismo
)
prob_churn = 1 / (1 + np.exp(-logit))
```

Esse design garante que o dataset tenha sinal aprendível sem ser trivialmente separável.

---

### 2. Análise Exploratória

**Script:** `2_eda.py`

Principais achados da fase de EDA:

| Achado | Taxa de Churn |
|---|---|
| Tipo de contrato: mensal | 42,1% |
| Tipo de contrato: anual | 14,3% |
| Tipo de contrato: bienal | 9,7% |
| Forma de pagamento: boleto | 34,2% |
| Forma de pagamento: cartão de crédito | 21,8% |
| Média de chamadas de suporte (churn) | 4,1 |
| Média de chamadas de suporte (retido) | 1,6 |
| Mediana de tenure (churn) | 8 meses |
| Mediana de tenure (retido) | 38 meses |

**Principais correlações com o alvo:**

```
chamadas_suporte        +0,312   sinal positivo mais forte
tempo_cliente_meses     -0,289   maior tenure → menor risco
contrato_enc            -0,244   contrato mais longo → menor risco
charge_mensal           +0,118
num_produtos            -0,089
```

**Desbalanceamento de classes:** 72,7% negativos (sem churn) vs. 27,3% positivos (churn) — desbalanceamento moderado, tratado via divisão estratificada e ajuste de threshold, sem necessidade de oversampling.

---

### 3. Pré-processamento e Engenharia de Features

**Script:** `3_treinar.py`

Três features derivadas são criadas antes de entrar no pipeline:

| Feature | Fórmula | Justificativa |
|---|---|---|
| `receita_por_produto` | `charge_mensal / num_produtos` | Normaliza a receita pela quantidade de produtos |
| `razao_suporte_tenure` | `chamadas_suporte / (tenure + 1)` | Intensidade de suporte relativa ao tempo de cliente |
| `cliente_novo` | `tenure ≤ 6 → 1` | Churn em clientes novos exibe padrões distintos |

Um único `ColumnTransformer` do `sklearn` centraliza todas as transformações:

```python
preprocessor = ColumnTransformer([
    ("num", StandardScaler(),                           features_numericas),
    ("cat", OneHotEncoder(handle_unknown="ignore"),     features_categoricas),
])
```

**Divisão treino/teste:** 80/20 estratificada pela classe-alvo, `random_state=42`.

---

### 4. Treinamento e Seleção de Modelos

**Script:** `3_treinar.py`

Três modelos foram avaliados numa arquitetura de pipeline unificada:

```python
pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf",  modelo),
])
```

Estratégia de validação cruzada: **StratifiedKFold (k=5)**, pontuado por ROC-AUC.

| Modelo | CV AUC (média ± desvio) | AUC no Teste | Precisão Média |
|---|---|---|---|
| Regressão Logística | 0,808 ± 0,011 | 0,812 | 0,641 |
| Random Forest | 0,884 ± 0,009 | 0,889 | 0,762 |
| **Gradient Boosting** ✓ | **0,897 ± 0,008** | **0,901** | **0,781** |

**Vencedor:** `GradientBoostingClassifier(n_estimators=200, random_state=42)`

Justificativa: maior AUC no teste, menor variância na validação cruzada e melhor precisão média na classe minoritária.

---

### 5. Avaliação

**Matriz de confusão com threshold = 0,50:**

```
                    Previsto: 0     Previsto: 1
Real: 0    │   661  (VN) ✓   │    72  (FP) ✗   │
Real: 1    │    56  (FN) ✗   │   211  (VP) ✓   │
```

**Métricas finais no conjunto de teste:**

| Métrica | Valor |
|---|---|
| ROC-AUC | **0,9012** |
| Precisão Média | **0,781** |
| Precisão (classe churn) | 0,79 |
| Recall (classe churn) | 0,74 |
| F1-Score (classe churn) | 0,764 |
| Acurácia | 0,873 |

**Importância de features (top 6):**

```
razao_suporte_tenure    ████████████████████  0,198
tempo_cliente_meses     ████████████████░░░░  0,167
contrato_mensal         ██████████████░░░░░░  0,143
chamadas_suporte        ████████████░░░░░░░░  0,121
charge_mensal           ██████████░░░░░░░░░░  0,098
receita_por_produto     █████████░░░░░░░░░░░  0,087
```

---

### 6. Deploy

**Arquivo:** `api/app.py`

O pipeline treinado é carregado uma única vez na inicialização e servido via FastAPI. A validação de entrada é feita por schemas Pydantic v2. Cada resposta inclui uma classificação de risco e uma recomendação em linguagem natural para a equipe de retenção.

**Classificação de risco:**

| Nível | Probabilidade | Ação Recomendada |
|---|---|---|
| `BAIXO` | < 30% | Incluir em programa de fidelidade |
| `MÉDIO` | 30–60% | Contato proativo da equipe de retenção |
| `ALTO` | > 60% | Oferta de desconto ou upgrade imediato |

---

## Referência da API

### `GET /health`

Retorna o status do serviço e a versão do modelo.

```json
{
  "status": "ok",
  "modelo": "GradientBoostingClassifier",
  "versao": "1.0.0"
}
```

---

### `POST /predict`

Prediz a probabilidade de churn para um único cliente.

**Corpo da requisição:**

```json
{
  "cliente_id": "CLI-00042",
  "idade": 34,
  "tempo_cliente_meses": 8,
  "contrato": "mensal",
  "forma_pagamento": "boleto",
  "charge_mensal": 95.50,
  "total_cobrado": 764.00,
  "num_produtos": 2,
  "tem_internet": 1,
  "tem_fone": 1,
  "chamadas_suporte": 5
}
```

**Resposta:**

```json
{
  "cliente_id": "CLI-00042",
  "churn_probabilidade": 0.7843,
  "churn_predicao": true,
  "risco": "ALTO",
  "recomendacao": "⚠️ Risco crítico! Oferecer desconto ou upgrade imediatamente.",
  "tempo_ms": 12.4
}
```

---

### `POST /predict/batch`

Mesmo schema do `/predict`, aceita um array JSON com até **1.000 clientes** por requisição.

---

## Executando Localmente

**Requisitos:** Python 3.11+

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/telemax-churn.git
cd telemax-churn/ml_churn

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Gere o dataset
python 1_gerar_dados.py

# 4. Execute a análise exploratória
python 2_eda.py

# 5. Treine e serialize o modelo
python 3_treinar.py

# 6. Suba a API
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Documentação interativa disponível em `http://localhost:8000/docs` (Swagger UI).

---

## Guia de Deploy

### Docker

```bash
# Build
docker build -t telemax-churn .

# Execução
docker run -p 8000:8000 telemax-churn
```

### Railway

1. Faça push do repositório para o GitHub (garanta que `models/modelo_churn.pkl` está commitado).
2. Acesse [railway.app](https://railway.app) → **New Project → Deploy from GitHub Repo**.
3. Selecione o repositório. O Railway detecta o `Dockerfile` automaticamente.
4. Vá em **Settings → Networking → Generate Domain**.
5. Defina a variável de ambiente: `PORT=8000`.

Sua API estará disponível em `https://<projeto>.up.railway.app`.

**Validação do deploy:**

```bash
curl https://<projeto>.up.railway.app/health

curl -X POST https://<projeto>.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{"cliente_id":"CLI-00001","idade":34,"tempo_cliente_meses":8,"contrato":"mensal","forma_pagamento":"boleto","charge_mensal":95.5,"total_cobrado":764,"num_produtos":2,"tem_internet":1,"tem_fone":1,"chamadas_suporte":5}'
```

---

## Resultados e Impacto de Negócio

Baseado na performance no conjunto de teste (1.000 clientes não vistos, 27,3% de taxa de churn):

| Resultado | Quantidade | Valor Unitário | Total |
|---|---|---|---|
| Verdadeiros positivos (churns identificados) | 211 | R$ 450 economizados (vs. aquisição) | **R$ 94.950** |
| Falsos positivos (ações de retenção desnecessárias) | 72 | –R$ 80 (custo de retenção) | –R$ 5.760 |
| **Valor líquido por 1.000 clientes** | | | **R$ 89.190** |

Em escala para uma base de 5.000 clientes, o modelo gera uma economia líquida estimada de **R$ 445.950 por ciclo**, com ROI de campanha de aproximadamente **16,5×**.

---

## Stack Tecnológica

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.11 |
| Framework de ML | scikit-learn 1.4 |
| Manipulação de dados | pandas 2.2, NumPy 1.26 |
| API | FastAPI 0.111, Uvicorn 0.29 |
| Validação | Pydantic v2 |
| Serialização | joblib 1.4 |
| Containerização | Docker |
| Deploy em nuvem | Railway |
