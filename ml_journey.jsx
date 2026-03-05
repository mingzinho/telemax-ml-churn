import { useState, useEffect, useRef } from "react";

const STEPS = [
  {
    id: 0,
    icon: "🎯",
    tag: "ETAPA 01",
    title: "Definição do Problema",
    subtitle: "Business Understanding",
    color: "#00d4ff",
    content: {
      desc: "A TeleMax — operadora fictícia de telecomunicações — perde ~27% dos clientes por ano. O objetivo é prever quais clientes têm maior probabilidade de cancelar o serviço (churn) nos próximos 30 dias, permitindo ações proativas de retenção.",
      metrics: [
        { label: "Custo de aquisição de cliente", value: "R$ 450" },
        { label: "Custo de retenção", value: "R$ 80" },
        { label: "Churn atual", value: "~27% ao ano" },
        { label: "Economia potencial (5k clientes)", value: "R$ 1.85M/ano" },
      ],
      code: `# Problema: Classificação Binária
# Target: churn = 1 (cancelou) | churn = 0 (ficou)
# Métrica principal: ROC-AUC
# Threshold de negócio: prob >= 0.50 → acionar retenção

objetivo = {
  "tipo": "classificação binária",
  "target": "churn",
  "metrica": "ROC-AUC + Precision-Recall",
  "baseline": "taxa histórica de churn = 27%"
}`,
    },
  },
  {
    id: 1,
    icon: "🗄️",
    tag: "ETAPA 02",
    title: "Geração de Dados",
    subtitle: "Data Collection · 5.000 registros",
    color: "#a78bfa",
    content: {
      desc: "Dataset fictício gerado com distribuições realistas. As features foram baseadas em datasets públicos de churn de telecom (ex: IBM Telco). O churn foi gerado via função logística com ruído para simular comportamento real.",
      features: [
        { name: "idade", type: "int", desc: "Idade do cliente (18-74 anos)" },
        { name: "tempo_cliente_meses", type: "int", desc: "Tempo de relacionamento" },
        { name: "contrato", type: "cat", desc: "mensal / anual / bienal" },
        { name: "forma_pagamento", type: "cat", desc: "Método de pagamento" },
        { name: "charge_mensal", type: "float", desc: "Valor da fatura mensal (R$)" },
        { name: "total_cobrado", type: "float", desc: "Total pago desde a adesão" },
        { name: "num_produtos", type: "int", desc: "Qtd de produtos contratados" },
        { name: "tem_internet", type: "bool", desc: "Possui plano de internet" },
        { name: "tem_fone", type: "bool", desc: "Possui plano de telefone" },
        { name: "chamadas_suporte", type: "int", desc: "Chamados abertos (mês)" },
        { name: "churn 🎯", type: "target", desc: "Cancelou? 0=Não / 1=Sim" },
      ],
      code: `import numpy as np
import pandas as pd

np.random.seed(42)
N = 5000

# Churn gerado via regressão logística + ruído
logit = (
  -2.5
  + 0.02 * chamadas_suporte   # + suporte → + churn
  - 0.03 * tenure             # + tempo   → - churn
  + 0.01 * charge_mensal      # + valor   → + churn
  - 0.8  * tipo_contrato      # contrato longo → - churn
  + np.random.normal(0, 0.3)  # ruído realista
)
prob_churn = 1 / (1 + np.exp(-logit))
churn = (random.uniform() < prob_churn).astype(int)`,
    },
  },
  {
    id: 2,
    icon: "🔍",
    tag: "ETAPA 03",
    title: "Análise Exploratória",
    subtitle: "EDA · Exploratory Data Analysis",
    color: "#f59e0b",
    content: {
      desc: "Investigamos distribuições, correlações e padrões para entender o dataset antes de qualquer modelagem.",
      insights: [
        { insight: "Clientes com contrato mensal churnam 3.2× mais que bienais", impact: "🔴 Alto" },
        { insight: "Média de chamadas de suporte: 4.1 (churn) vs 1.6 (não-churn)", impact: "🔴 Alto" },
        { insight: "Mediana de tenure: 8 meses (churn) vs 38 meses (não-churn)", impact: "🔴 Alto" },
        { insight: "Pagamento por boleto: churn 34% vs cartão: 22%", impact: "🟡 Médio" },
        { insight: "Dataset desbalanceado: 73% não-churn vs 27% churn", impact: "⚙️ Técnico" },
      ],
      stats: [
        { label: "Total de registros", value: "5.000" },
        { label: "Features brutas", value: "11" },
        { label: "Taxa de churn", value: "27.3%" },
        { label: "Valores nulos", value: "0" },
        { label: "Features categóricas", value: "2" },
        { label: "Features numéricas", value: "9" },
      ],
      code: `# Correlações com churn
correlacoes = df.corr()["churn"].sort_values()

# chamadas_suporte    +0.312  ← mais correlacionada
# tempo_cliente       -0.289  ← tenure protege
# tipo_contrato_enc   -0.244  ← contrato longo protege
# charge_mensal       +0.118
# num_produtos        -0.089

# Análise de balanceamento
df["churn"].value_counts(normalize=True)
# 0 (não-churn): 72.7%
# 1 (churn):     27.3%  ← desbalanceado!`,
    },
  },
  {
    id: 3,
    icon: "⚙️",
    tag: "ETAPA 04",
    title: "Pré-processamento",
    subtitle: "Feature Engineering · Pipeline",
    color: "#34d399",
    content: {
      desc: "Transformamos os dados brutos em representações numéricas úteis para os modelos. Criamos 3 novas features derivadas e montamos um pipeline sklearn reprodutível.",
      engineered: [
        { feature: "receita_por_produto", formula: "charge_mensal / num_produtos", motivo: "Eficiência da receita por produto" },
        { feature: "razao_suporte_tenure", formula: "chamadas_suporte / (tenure + 1)", motivo: "Intensidade de suporte normalizada" },
        { feature: "cliente_novo", formula: "tenure ≤ 6 meses → 1", motivo: "Clientes novos têm churn diferente" },
      ],
      code: `from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Engenharia de features
df["receita_por_produto"]   = df["charge_mensal"] / df["num_produtos"]
df["razao_suporte_tenure"]  = df["chamadas_suporte"] / (df["tenure"] + 1)
df["cliente_novo"]          = (df["tenure"] <= 6).astype(int)

# Pipeline sklearn (reproducível e deployável)
preprocessor = ColumnTransformer([
  ("num", StandardScaler(),    features_numericas),
  ("cat", OneHotEncoder(
    handle_unknown="ignore"), features_categoricas),
])

# Split estratificado (mantém proporção de churn)
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.20, stratify=y, random_state=42
)
# Train: 4.000 | Test: 1.000`,
    },
  },
  {
    id: 4,
    icon: "🤖",
    tag: "ETAPA 05",
    title: "Treinamento de Modelos",
    subtitle: "3 Algoritmos · Cross-Validation",
    color: "#f472b6",
    content: {
      desc: "Treinamos 3 modelos com complexidades diferentes, avaliando via cross-validation estratificada de 5 folds para garantir robustez dos resultados.",
      models: [
        {
          name: "Regressão Logística",
          auc: 0.812,
          ap: 0.641,
          cv: "0.808 ± 0.011",
          pros: "Interpretável, rápido",
          cons: "Assume linearidade",
          badge: "baseline",
        },
        {
          name: "Random Forest",
          auc: 0.889,
          ap: 0.762,
          cv: "0.884 ± 0.009",
          pros: "Robusto, feature importance",
          cons: "Menos interpretável",
          badge: "bom",
        },
        {
          name: "Gradient Boosting",
          auc: 0.901,
          ap: 0.781,
          cv: "0.897 ± 0.008",
          pros: "Melhor performance geral",
          cons: "Mais lento para treinar",
          badge: "🏆 melhor",
        },
      ],
      code: `from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

modelos = {
  "LogReg": LogisticRegression(max_iter=1000),
  "RF":     RandomForestClassifier(n_estimators=200),
  "GBM":    GradientBoostingClassifier(n_estimators=200),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for nome, modelo in modelos.items():
  pipe = Pipeline([("prep", preprocessor), ("clf", modelo)])
  scores = cross_val_score(pipe, X_train, y_train,
                           cv=cv, scoring="roc_auc")
  print(f"{nome}: {scores.mean():.4f} ± {scores.std():.4f}")`,
    },
  },
  {
    id: 5,
    icon: "📊",
    tag: "ETAPA 06",
    title: "Avaliação & Tuning",
    subtitle: "Métricas · Threshold · Importância",
    color: "#fb923c",
    content: {
      desc: "Avaliamos o modelo campeão no conjunto de teste. Para o problema de churn, Precision-Recall é tão importante quanto ROC-AUC — queremos identificar quem vai sair sem acionar muitos falsos alarmes.",
      finalMetrics: [
        { metric: "ROC-AUC", value: "0.9012", comment: "Excelente discriminação" },
        { metric: "Avg Precision", value: "0.781", comment: "Alta precisão no recall" },
        { metric: "Precision (churn)", value: "0.79", comment: "79% dos alertas são corretos" },
        { metric: "Recall (churn)", value: "0.74", comment: "Captura 74% dos churns reais" },
        { metric: "F1-Score", value: "0.764", comment: "Equilíbrio P/R" },
        { metric: "Accuracy", value: "0.873", comment: "Linha de base para contexto" },
      ],
      topFeatures: [
        { feature: "razao_suporte_tenure", imp: 0.198, bar: 100 },
        { feature: "tempo_cliente_meses", imp: 0.167, bar: 84 },
        { feature: "contrato_mensal", imp: 0.143, bar: 72 },
        { feature: "chamadas_suporte", imp: 0.121, bar: 61 },
        { feature: "charge_mensal", imp: 0.098, bar: 49 },
        { feature: "receita_por_produto", imp: 0.087, bar: 44 },
      ],
      code: `# Matriz de confusão (threshold = 0.50)
# ┌─────────────┬──────────────┬──────────────┐
# │             │ Pred: 0      │ Pred: 1      │
# ├─────────────┼──────────────┼──────────────┤
# │ Real: 0     │ 661 (TN) ✅  │  72 (FP) ⚠️  │
# │ Real: 1     │  56 (FN) ❌  │ 211 (TP) ✅  │
# └─────────────┴──────────────┴──────────────┘

# Impacto de negócio:
# - 211 churns capturados × R$450 custo acq. = R$94.950 economia
# - 72 falsos alarmes × R$80 retenção desnecessária = R$5.760
# ROI da campanha de retenção: ~16.5×`,
    },
  },
  {
    id: 6,
    icon: "🚀",
    tag: "ETAPA 07",
    title: "Deploy em Produção",
    subtitle: "FastAPI · Docker · REST API",
    color: "#00d4ff",
    content: {
      desc: "O modelo é servido como API REST via FastAPI, containerizado com Docker para deploy em qualquer ambiente cloud (AWS ECS, GCP Cloud Run, Azure Container Apps).",
      endpoints: [
        { method: "GET", path: "/health", desc: "Health check da API + versão do modelo" },
        { method: "POST", path: "/predict", desc: "Predição individual (JSON → JSON)" },
        { method: "POST", path: "/predict/batch", desc: "Predição em lote (até 1.000/req)" },
      ],
      riskLevels: [
        { level: "BAIXO", range: "0% – 30%", color: "#34d399", action: "Programa de fidelidade" },
        { level: "MÉDIO", range: "30% – 60%", color: "#f59e0b", action: "Acionar equipe de retenção" },
        { level: "ALTO", range: "60% – 100%", color: "#f87171", action: "Oferta especial imediata" },
      ],
      code: `# REQUEST: POST /predict
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

# RESPONSE: 200 OK
{
  "cliente_id": "CLI-00042",
  "churn_probabilidade": 0.7843,
  "churn_predicao": true,
  "risco": "ALTO",
  "recomendacao": "⚠️ Risco crítico! Oferecer desconto imediatamente.",
  "tempo_ms": 12.4
}`,
    },
  },
];

function CodeBlock({ code }) {
  const [copied, setCopied] = useState(false);
  const keywords = ["import", "from", "def", "class", "return", "for", "in", "if", "else", "True", "False", "None", "and", "or", "not", "lambda", "with", "as", "try", "except", "raise", "GET", "POST"];
  
  const highlight = (line) => {
    const isComment = line.trim().startsWith("#");
    if (isComment) return <span style={{ color: "#6b7280", fontStyle: "italic" }}>{line}</span>;
    
    const parts = [];
    let remaining = line;
    let key = 0;
    
    const patterns = [
      { re: /(".*?"|'.*?')/g, color: "#a78bfa" },
      { re: /\b(\d+\.?\d*)\b/g, color: "#fbbf24" },
    ];
    
    return <span style={{ color: "#e2e8f0" }}>{line}</span>;
  };
  
  return (
    <div style={{ position: "relative", marginTop: 12 }}>
      <div style={{
        background: "#0a0f1a",
        border: "1px solid #1e293b",
        borderRadius: 8,
        padding: "16px 18px",
        fontFamily: "'JetBrains Mono', 'Fira Code', 'Courier New', monospace",
        fontSize: 12,
        lineHeight: 1.7,
        overflowX: "auto",
      }}>
        {code.split("\n").map((line, i) => {
          const isComment = line.trim().startsWith("#");
          const isString = line.includes('"') || line.includes("'");
          const keywords = ["import", "from", "def", "class", "return", "for", "in", "if", "print", "True", "False", "None"];
          let color = "#c9d1d9";
          if (isComment) color = "#5c6f80";
          
          return (
            <div key={i} style={{ display: "flex" }}>
              <span style={{ color: "#2d3f52", userSelect: "none", marginRight: 16, minWidth: 20, textAlign: "right" }}>{i + 1}</span>
              <span style={{ color: isComment ? "#5c6f80" : "#c9d1d9", fontStyle: isComment ? "italic" : "normal" }}>
                {line.split(" ").map((word, wi) => {
                  const clean = word.replace(/[(){}:,\[\]]/g, "");
                  if (keywords.includes(clean)) return <span key={wi} style={{ color: "#79c0ff" }}>{word} </span>;
                  if (word.match(/^["'].*["']$/)) return <span key={wi} style={{ color: "#a78bfa" }}>{word} </span>;
                  if (word.match(/^\d+\.?\d*$/)) return <span key={wi} style={{ color: "#fbbf24" }}>{word} </span>;
                  return <span key={wi}>{word} </span>;
                })}
              </span>
            </div>
          );
        })}
      </div>
      <button
        onClick={() => { navigator.clipboard?.writeText(code); setCopied(true); setTimeout(() => setCopied(false), 2000); }}
        style={{
          position: "absolute", top: 8, right: 8,
          background: copied ? "#34d39922" : "#1e293b",
          border: `1px solid ${copied ? "#34d399" : "#334155"}`,
          borderRadius: 4, padding: "3px 10px",
          color: copied ? "#34d399" : "#64748b",
          fontSize: 10, cursor: "pointer", fontFamily: "monospace",
        }}
      >
        {copied ? "✓ copiado" : "copiar"}
      </button>
    </div>
  );
}

function ProgressBar({ value, color, max = 1 }) {
  const [width, setWidth] = useState(0);
  useEffect(() => { setTimeout(() => setWidth((value / max) * 100), 100); }, [value]);
  return (
    <div style={{ background: "#0f172a", borderRadius: 4, height: 6, overflow: "hidden" }}>
      <div style={{
        width: `${width}%`, height: "100%", background: color,
        borderRadius: 4, transition: "width 0.8s cubic-bezier(0.4,0,0.2,1)",
        boxShadow: `0 0 8px ${color}66`,
      }} />
    </div>
  );
}

export default function MLJourney() {
  const [activeStep, setActiveStep] = useState(0);
  const [mounted, setMounted] = useState(false);
  useEffect(() => { setMounted(true); }, []);

  const step = STEPS[activeStep];

  return (
    <div style={{
      fontFamily: "'Inter', system-ui, sans-serif",
      background: "#030712",
      minHeight: "100vh",
      color: "#e2e8f0",
      display: "flex",
      flexDirection: "column",
    }}>
      {/* Header */}
      <div style={{
        borderBottom: "1px solid #0f172a",
        padding: "20px 32px",
        display: "flex",
        alignItems: "center",
        gap: 16,
        background: "#04090f",
      }}>
        <div style={{
          width: 36, height: 36, borderRadius: 8,
          background: "linear-gradient(135deg, #00d4ff22, #a78bfa22)",
          border: "1px solid #1e293b",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 18,
        }}>⚡</div>
        <div>
          <div style={{ fontWeight: 700, fontSize: 15, letterSpacing: "0.02em" }}>TeleMax · ML Pipeline</div>
          <div style={{ fontSize: 11, color: "#475569", letterSpacing: "0.08em", textTransform: "uppercase" }}>
            Churn Prediction · End-to-End
          </div>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
          {["Python", "sklearn", "FastAPI", "Docker"].map(t => (
            <span key={t} style={{
              fontSize: 10, padding: "3px 8px", borderRadius: 4,
              background: "#0f172a", border: "1px solid #1e293b",
              color: "#64748b", letterSpacing: "0.05em",
            }}>{t}</span>
          ))}
        </div>
      </div>

      <div style={{ display: "flex", flex: 1 }}>
        {/* Sidebar */}
        <div style={{
          width: 220,
          borderRight: "1px solid #0f172a",
          padding: "20px 0",
          background: "#04090f",
          flexShrink: 0,
        }}>
          {STEPS.map((s, i) => (
            <button
              key={s.id}
              onClick={() => setActiveStep(i)}
              style={{
                width: "100%",
                padding: "11px 20px",
                display: "flex",
                alignItems: "center",
                gap: 12,
                cursor: "pointer",
                border: "none",
                background: activeStep === i ? `${s.color}12` : "transparent",
                borderLeft: activeStep === i ? `2px solid ${s.color}` : "2px solid transparent",
                textAlign: "left",
                transition: "all 0.15s",
              }}
            >
              <span style={{ fontSize: 16 }}>{s.icon}</span>
              <div>
                <div style={{ fontSize: 9, color: activeStep === i ? s.color : "#374151", letterSpacing: "0.1em", fontWeight: 700 }}>
                  {s.tag}
                </div>
                <div style={{ fontSize: 12, color: activeStep === i ? "#e2e8f0" : "#6b7280", fontWeight: 500, marginTop: 1 }}>
                  {s.title}
                </div>
              </div>
            </button>
          ))}

          {/* Overall progress */}
          <div style={{ padding: "20px", marginTop: 8, borderTop: "1px solid #0f172a" }}>
            <div style={{ fontSize: 10, color: "#374151", letterSpacing: "0.1em", marginBottom: 8 }}>
              PROGRESSO GERAL
            </div>
            <ProgressBar value={activeStep + 1} max={STEPS.length} color={step.color} />
            <div style={{ fontSize: 10, color: "#475569", marginTop: 6 }}>
              {activeStep + 1} / {STEPS.length} etapas
            </div>
          </div>
        </div>

        {/* Main content */}
        <div style={{ flex: 1, overflow: "auto", padding: "28px 32px" }}>
          {/* Step header */}
          <div style={{ marginBottom: 24 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
              <span style={{ fontSize: 28 }}>{step.icon}</span>
              <div>
                <div style={{ fontSize: 10, color: step.color, letterSpacing: "0.15em", fontWeight: 700, marginBottom: 2 }}>
                  {step.tag}
                </div>
                <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700, letterSpacing: "-0.02em" }}>{step.title}</h1>
              </div>
              <span style={{
                marginLeft: "auto", fontSize: 11, padding: "4px 12px",
                borderRadius: 20, border: `1px solid ${step.color}44`,
                color: step.color, background: `${step.color}11`,
              }}>{step.subtitle}</span>
            </div>
            <p style={{ color: "#94a3b8", fontSize: 14, lineHeight: 1.7, margin: 0, maxWidth: 700 }}>
              {step.content.desc}
            </p>
          </div>

          {/* Step-specific content */}

          {/* ETAPA 1 — Problema */}
          {step.id === 0 && (
            <div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 20 }}>
                {step.content.metrics.map((m, i) => (
                  <div key={i} style={{
                    background: "#04090f", border: "1px solid #0f172a",
                    borderRadius: 10, padding: "16px 20px",
                  }}>
                    <div style={{ fontSize: 11, color: "#475569", marginBottom: 4 }}>{m.label}</div>
                    <div style={{ fontSize: 20, fontWeight: 700, color: step.color }}>{m.value}</div>
                  </div>
                ))}
              </div>
              <CodeBlock code={step.content.code} />
            </div>
          )}

          {/* ETAPA 2 — Dados */}
          {step.id === 1 && (
            <div>
              <div style={{ background: "#04090f", border: "1px solid #0f172a", borderRadius: 10, marginBottom: 20, overflow: "hidden" }}>
                <div style={{ padding: "12px 20px", borderBottom: "1px solid #0f172a", fontSize: 11, color: "#475569", letterSpacing: "0.1em" }}>
                  SCHEMA DO DATASET · clientes_telemax.csv
                </div>
                {step.content.features.map((f, i) => (
                  <div key={i} style={{
                    display: "flex", alignItems: "center", gap: 16,
                    padding: "10px 20px",
                    borderBottom: i < step.content.features.length - 1 ? "1px solid #080d14" : "none",
                    background: i % 2 === 0 ? "transparent" : "#02060d",
                  }}>
                    <span style={{ fontFamily: "monospace", fontSize: 12, color: step.color, minWidth: 160 }}>{f.name}</span>
                    <span style={{
                      fontSize: 10, padding: "2px 7px", borderRadius: 4,
                      background: f.type === "target" ? "#a78bfa22" : "#0f172a",
                      border: `1px solid ${f.type === "target" ? "#a78bfa44" : "#1e293b"}`,
                      color: f.type === "target" ? "#a78bfa" : "#64748b",
                      minWidth: 44, textAlign: "center",
                    }}>{f.type}</span>
                    <span style={{ fontSize: 12, color: "#94a3b8" }}>{f.desc}</span>
                  </div>
                ))}
              </div>
              <CodeBlock code={step.content.code} />
            </div>
          )}

          {/* ETAPA 3 — EDA */}
          {step.id === 2 && (
            <div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, marginBottom: 20 }}>
                {step.content.stats.map((s, i) => (
                  <div key={i} style={{
                    background: "#04090f", border: "1px solid #0f172a",
                    borderRadius: 8, padding: "14px 16px",
                  }}>
                    <div style={{ fontSize: 10, color: "#374151", marginBottom: 4 }}>{s.label}</div>
                    <div style={{ fontSize: 18, fontWeight: 700, color: step.color }}>{s.value}</div>
                  </div>
                ))}
              </div>
              <div style={{ marginBottom: 20 }}>
                <div style={{ fontSize: 11, color: "#475569", letterSpacing: "0.1em", marginBottom: 10 }}>PRINCIPAIS INSIGHTS</div>
                {step.content.insights.map((ins, i) => (
                  <div key={i} style={{
                    display: "flex", alignItems: "center", gap: 12,
                    padding: "10px 14px", marginBottom: 6,
                    background: "#04090f", border: "1px solid #0f172a", borderRadius: 8,
                  }}>
                    <span style={{ fontSize: 12 }}>{ins.impact}</span>
                    <span style={{ fontSize: 13, color: "#cbd5e1" }}>{ins.insight}</span>
                  </div>
                ))}
              </div>
              <CodeBlock code={step.content.code} />
            </div>
          )}

          {/* ETAPA 4 — Pré-processamento */}
          {step.id === 3 && (
            <div>
              <div style={{ marginBottom: 20 }}>
                <div style={{ fontSize: 11, color: "#475569", letterSpacing: "0.1em", marginBottom: 10 }}>FEATURES CRIADAS (FEATURE ENGINEERING)</div>
                {step.content.engineered.map((f, i) => (
                  <div key={i} style={{
                    background: "#04090f", border: "1px solid #0f172a",
                    borderRadius: 8, padding: "14px 16px", marginBottom: 8,
                  }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 6 }}>
                      <span style={{ fontFamily: "monospace", color: step.color, fontSize: 13, fontWeight: 600 }}>{f.feature}</span>
                      <span style={{ fontSize: 10, color: "#475569" }}>=</span>
                      <span style={{ fontFamily: "monospace", color: "#a78bfa", fontSize: 12 }}>{f.formula}</span>
                    </div>
                    <div style={{ fontSize: 12, color: "#64748b" }}>💡 {f.motivo}</div>
                  </div>
                ))}
              </div>
              <CodeBlock code={step.content.code} />
            </div>
          )}

          {/* ETAPA 5 — Treinamento */}
          {step.id === 4 && (
            <div>
              <div style={{ marginBottom: 20 }}>
                {step.content.models.map((m, i) => (
                  <div key={i} style={{
                    background: "#04090f",
                    border: `1px solid ${m.badge === "🏆 melhor" ? step.color + "44" : "#0f172a"}`,
                    borderRadius: 10, padding: "16px 20px", marginBottom: 10,
                    boxShadow: m.badge === "🏆 melhor" ? `0 0 20px ${step.color}11` : "none",
                  }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
                      <span style={{ fontWeight: 600, fontSize: 14 }}>{m.name}</span>
                      <span style={{
                        fontSize: 10, padding: "2px 8px", borderRadius: 20,
                        background: m.badge === "🏆 melhor" ? `${step.color}22` : "#0f172a",
                        border: `1px solid ${m.badge === "🏆 melhor" ? step.color + "44" : "#1e293b"}`,
                        color: m.badge === "🏆 melhor" ? step.color : "#475569",
                      }}>{m.badge}</span>
                      <div style={{ marginLeft: "auto", display: "flex", gap: 16 }}>
                        <div style={{ textAlign: "right" }}>
                          <div style={{ fontSize: 9, color: "#374151" }}>TEST AUC</div>
                          <div style={{ fontSize: 16, fontWeight: 700, color: step.color }}>{m.auc}</div>
                        </div>
                        <div style={{ textAlign: "right" }}>
                          <div style={{ fontSize: 9, color: "#374151" }}>AVG PREC</div>
                          <div style={{ fontSize: 16, fontWeight: 700, color: "#94a3b8" }}>{m.ap}</div>
                        </div>
                      </div>
                    </div>
                    <div style={{ marginBottom: 10 }}>
                      <ProgressBar value={m.auc} color={step.color} />
                    </div>
                    <div style={{ display: "flex", gap: 16, fontSize: 11 }}>
                      <span style={{ color: "#34d399" }}>✓ {m.pros}</span>
                      <span style={{ color: "#6b7280" }}>✗ {m.cons}</span>
                      <span style={{ color: "#475569", marginLeft: "auto" }}>CV: {m.cv}</span>
                    </div>
                  </div>
                ))}
              </div>
              <CodeBlock code={step.content.code} />
            </div>
          )}

          {/* ETAPA 6 — Avaliação */}
          {step.id === 5 && (
            <div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 20 }}>
                <div>
                  <div style={{ fontSize: 11, color: "#475569", letterSpacing: "0.1em", marginBottom: 10 }}>MÉTRICAS FINAIS (TEST SET)</div>
                  {step.content.finalMetrics.map((m, i) => (
                    <div key={i} style={{
                      display: "flex", alignItems: "center", gap: 12,
                      padding: "8px 14px", marginBottom: 6,
                      background: "#04090f", border: "1px solid #0f172a", borderRadius: 8,
                    }}>
                      <span style={{ fontFamily: "monospace", color: step.color, fontSize: 13, fontWeight: 700, minWidth: 60 }}>{m.value}</span>
                      <div>
                        <div style={{ fontSize: 12, fontWeight: 600 }}>{m.metric}</div>
                        <div style={{ fontSize: 10, color: "#475569" }}>{m.comment}</div>
                      </div>
                    </div>
                  ))}
                </div>
                <div>
                  <div style={{ fontSize: 11, color: "#475569", letterSpacing: "0.1em", marginBottom: 10 }}>FEATURE IMPORTANCE (TOP 6)</div>
                  {step.content.topFeatures.map((f, i) => (
                    <div key={i} style={{ marginBottom: 8 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                        <span style={{ fontSize: 11, fontFamily: "monospace", color: "#94a3b8" }}>{f.feature}</span>
                        <span style={{ fontSize: 11, color: step.color }}>{f.imp}</span>
                      </div>
                      <ProgressBar value={f.bar} max={100} color={step.color} />
                    </div>
                  ))}
                </div>
              </div>
              <CodeBlock code={step.content.code} />
            </div>
          )}

          {/* ETAPA 7 — Deploy */}
          {step.id === 6 && (
            <div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 20 }}>
                <div>
                  <div style={{ fontSize: 11, color: "#475569", letterSpacing: "0.1em", marginBottom: 10 }}>ENDPOINTS DA API</div>
                  {step.content.endpoints.map((e, i) => (
                    <div key={i} style={{
                      display: "flex", gap: 12, alignItems: "flex-start",
                      padding: "10px 14px", marginBottom: 8,
                      background: "#04090f", border: "1px solid #0f172a", borderRadius: 8,
                    }}>
                      <span style={{
                        fontSize: 10, padding: "2px 8px", borderRadius: 4, fontWeight: 700,
                        background: e.method === "GET" ? "#34d39922" : "#00d4ff22",
                        border: `1px solid ${e.method === "GET" ? "#34d39944" : "#00d4ff44"}`,
                        color: e.method === "GET" ? "#34d399" : "#00d4ff",
                        minWidth: 36, textAlign: "center",
                      }}>{e.method}</span>
                      <div>
                        <div style={{ fontFamily: "monospace", fontSize: 12, color: "#e2e8f0", marginBottom: 2 }}>{e.path}</div>
                        <div style={{ fontSize: 11, color: "#475569" }}>{e.desc}</div>
                      </div>
                    </div>
                  ))}
                </div>
                <div>
                  <div style={{ fontSize: 11, color: "#475569", letterSpacing: "0.1em", marginBottom: 10 }}>CLASSIFICAÇÃO DE RISCO</div>
                  {step.content.riskLevels.map((r, i) => (
                    <div key={i} style={{
                      padding: "12px 14px", marginBottom: 8,
                      background: "#04090f", border: `1px solid ${r.color}22`, borderRadius: 8,
                      borderLeft: `3px solid ${r.color}`,
                    }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                        <span style={{ fontWeight: 700, color: r.color, fontSize: 13 }}>{r.level}</span>
                        <span style={{ fontSize: 11, color: "#475569", fontFamily: "monospace" }}>{r.range}</span>
                      </div>
                      <div style={{ fontSize: 12, color: "#64748b" }}>→ {r.action}</div>
                    </div>
                  ))}
                  <div style={{
                    padding: "12px 14px", background: "#04090f",
                    border: "1px solid #0f172a", borderRadius: 8, marginTop: 8,
                  }}>
                    <div style={{ fontSize: 11, color: "#475569", marginBottom: 6 }}>DEPLOY RÁPIDO</div>
                    <div style={{ fontFamily: "monospace", fontSize: 11, color: "#34d399", lineHeight: 1.8 }}>
                      $ docker build -t telemax-churn .<br/>
                      $ docker run -p 8000:8000 telemax-churn<br/>
                      <span style={{ color: "#475569" }}># → API pronta em localhost:8000</span>
                    </div>
                  </div>
                </div>
              </div>
              <CodeBlock code={step.content.code} />
            </div>
          )}

          {/* Navigation */}
          <div style={{ display: "flex", justifyContent: "space-between", marginTop: 28, paddingTop: 20, borderTop: "1px solid #0f172a" }}>
            <button
              onClick={() => setActiveStep(Math.max(0, activeStep - 1))}
              disabled={activeStep === 0}
              style={{
                padding: "10px 20px", borderRadius: 8, cursor: activeStep === 0 ? "not-allowed" : "pointer",
                background: "#0f172a", border: "1px solid #1e293b",
                color: activeStep === 0 ? "#2d3f52" : "#94a3b8", fontSize: 13,
              }}
            >← Anterior</button>

            <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
              {STEPS.map((s, i) => (
                <div key={i} onClick={() => setActiveStep(i)} style={{
                  width: i === activeStep ? 20 : 6,
                  height: 6, borderRadius: 3,
                  background: i === activeStep ? s.color : "#1e293b",
                  cursor: "pointer", transition: "all 0.3s",
                }} />
              ))}
            </div>

            <button
              onClick={() => setActiveStep(Math.min(STEPS.length - 1, activeStep + 1))}
              disabled={activeStep === STEPS.length - 1}
              style={{
                padding: "10px 20px", borderRadius: 8,
                cursor: activeStep === STEPS.length - 1 ? "not-allowed" : "pointer",
                background: activeStep === STEPS.length - 1 ? "#0f172a" : `${step.color}22`,
                border: `1px solid ${activeStep === STEPS.length - 1 ? "#1e293b" : step.color + "44"}`,
                color: activeStep === STEPS.length - 1 ? "#2d3f52" : step.color,
                fontSize: 13, fontWeight: 600,
              }}
            >Próxima →</button>
          </div>
        </div>
      </div>
    </div>
  );
}
