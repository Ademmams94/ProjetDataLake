# 🌊 Data Lake — Analyse de Sentiment Social (NLP)

Projet final **Data Lakes & Data Integration** — EFREI 2025-2026.
Un data lake de A à Z : ingestion (fichier + API) → raffinage en 3 zones → exposition via une API.

## 🎯 Ce que fait le projet

Ingère du texte social (posts Reddit en temps réel + un dataset HuggingFace),
le fait transiter par 3 zones de raffinage, calcule un **score de sentiment** (NLP),
et expose le tout via une **API Gateway**.

```
[Reddit API]  ┐
              ├─► RAW (S3/MinIO) ─► STAGING (Parquet) ─► CURATED (Elasticsearch) ─► API (FastAPI)
[Dataset HF]  ┘         brut            nettoyé              sentiment scoré
                         └──────────── orchestré par Apache Airflow ────────────┘
```

## 🏗️ Architecture

| Zone | Techno | Rôle |
|------|--------|------|
| **RAW** | MinIO (S3) | Données brutes stockées telles quelles (JSON) |
| **STAGING** | Parquet sur MinIO | Texte nettoyé, structuré, colonne compressée |
| **CURATED** | Elasticsearch | Documents scorés en sentiment, cherchables et agrégeables |

- **Orchestration** : Apache Airflow (scheduling de l'ingestion API)
- **API** : FastAPI (`/raw`, `/staging`, `/curated`, `/health`, `/stats`, + `/ingest` et `/ingest_fast`)
- **NLP** : VADER (score de sentiment)

## 🚀 Installation & lancement

> ⚠️ Prérequis : Docker Desktop, Python 3.11+

```bash
# 1. Configuration
cp .env.example .env        # puis remplir les identifiants Reddit

# 2. Lancer l'infra (MinIO + Elasticsearch)
docker compose up -d

# 3. Environnement Python
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

_(Sections détaillées ajoutées au fil des phases.)_

## 📁 Structure du repo

```
src/
  common/       # config, clients S3 & Elasticsearch, fonctions NLP
  ingestion/    # sources : Reddit (API) + HuggingFace (fichier)
  transform/    # pipelines raw→staging→curated
  api/          # API Gateway FastAPI
airflow/dags/   # orchestration
docs/           # documentation technique
```

## 👤 Auteur

Adem Mamouni — EFREI, spé Big Data & Machine Learning.
