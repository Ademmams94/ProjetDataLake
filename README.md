# 🌊 Data Lake — Analyse de Sentiment Social (NLP)

Projet final **Data Lakes & Data Integration** — EFREI 2025-2026.
Un data lake de A à Z : ingestion (fichier + API) → raffinage en 3 zones →
exposition via une API, le tout orchestré et conteneurisé. **Niveau avancé inclus.**

---

## 🎯 Ce que fait le projet

Ingère du texte social (posts **Hacker News** en temps réel + dataset **HuggingFace**),
le fait transiter par 3 zones de raffinage, calcule un **score de sentiment** via un
**Transformer RoBERTa** (Deep Learning, 75,6 % de précision), et expose le tout via une
**API Gateway**.

```
 [Hacker News API] ─┐
                    ├─► RAW (MinIO/S3) ─► STAGING (Parquet) ─► CURATED (Elasticsearch) ─► API (FastAPI)
 [Dataset HF] ──────┘      brut               nettoyé,             + sentiment (RoBERTa)
                                               schéma unifié
                           └──────────── orchestration : Apache Airflow ─────────────┘
```

---

## 🏗️ Stack technique

| Composant | Technologie | Rôle |
|---|---|---|
| Zone RAW | **MinIO** (S3) | données brutes (JSON) |
| Zone STAGING | **Parquet** sur MinIO | données nettoyées et unifiées |
| Zone CURATED | **Elasticsearch** | données enrichies, cherchables et agrégeables |
| Modèle NLP | **RoBERTa** (Transformers) | analyse de sentiment (VADER en baseline) |
| Orchestration | **Apache Airflow** | pipeline automatisé + scheduling horaire |
| API | **FastAPI** | exposition des zones + endpoints d'ingestion |
| Infra | **Docker Compose** | tout démarre en une commande |

Architecture détaillée : [`docs/architecture.md`](docs/architecture.md).
Résultats de performance : [`docs/performance.md`](docs/performance.md).

---

## 🚀 Installation & lancement

### Prérequis
- **Docker Desktop** (avec Docker Compose v2)
- ~6 Go de RAM disponibles, ~5 Go de disque (images + modèle)

### Démarrage (une commande)

```bash
# 1. Configuration (les valeurs par défaut fonctionnent telles quelles)
cp .env.example .env

# 2. Build + démarrage de toute la stack
docker compose up -d --build
```

Au premier lancement, le build des images et le téléchargement du modèle RoBERTa
(~500 Mo, mis en cache ensuite) prennent quelques minutes.

### Services accessibles

| Service | URL | Identifiants |
|---|---|---|
| API — doc interactive | http://localhost:8000/docs | — |
| Airflow | http://localhost:8080 | `admin` / `admin` |
| MinIO (console) | http://localhost:9001 | `minioadmin` / `minioadmin123` |
| Elasticsearch | http://localhost:9200 | — |

### Alimenter le lac

Le DAG Airflow tourne automatiquement toutes les heures. Pour le déclencher
immédiatement :

```bash
docker exec dl_airflow_scheduler airflow dags trigger datalake_pipeline
```

Ou, sans Airflow, en lançant les pipelines à la main (voir section suivante).

---

## 🕹️ Utilisation de l'API

```bash
# État des services
curl http://localhost:8000/health

# Volumétrie des 3 zones
curl http://localhost:8000/stats

# Données brutes (listing)
curl "http://localhost:8000/raw?source=hackernews&limit=5"

# Zone intermédiaire (dernier snapshot Parquet)
curl "http://localhost:8000/staging?source=huggingface&limit=5"

# Zone finale : recherche full-text + filtres
curl "http://localhost:8000/curated?q=python&sentiment=positive&limit=5"

# [Avancé] Ingestion à la demande
curl -X POST http://localhost:8000/ingest_fast \
  -H "Content-Type: application/json" \
  -d '{"data":{"texts":["I love this!","This is terrible."]}}'
```

Tous les endpoints sont documentés et testables sur http://localhost:8000/docs.

---

## 🔬 Développement local (hors conteneur)

Pour lancer les scripts de pipeline ou le benchmark directement sur l'hôte :

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate sous Unix)
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Pipelines individuels (l'infra Docker doit tourner)
python -m src.ingestion.hackernews_api     # source API   -> RAW
python -m src.ingestion.load_dataset       # source HF    -> RAW
python -m src.transform.raw_to_staging     # RAW          -> STAGING
python -m src.transform.staging_to_curated # STAGING      -> CURATED

# Benchmark du niveau avancé (génère docs/performance.md)
python scripts/benchmark.py --sizes 1 2 5 10 25 50 100
```

> ⚠️ **Windows** : `torch` doit être importé avant `pandas`/`pyarrow` (conflit de
> runtime OpenMP). Le module `src/common/torch_first.py` s'en charge ; il est
> importé en tête des points d'entrée concernés.

---

## ⚙️ Configuration

Toute la configuration passe par variables d'environnement (voir
[`.env.example`](.env.example)). Principales options :

| Variable | Défaut | Description |
|---|---|---|
| `SENTIMENT_ENGINE` | `transformer` | `transformer` (RoBERTa) ou `vader` |
| `HN_STORY_TYPE` | `topstories` | `topstories` \| `newstories` \| `beststories` |
| `HN_MAX_ITEMS` | `50` | posts récupérés par cycle |
| `HF_DATASET` | `cardiffnlp/tweet_eval` | dataset HuggingFace |
| `HF_SAMPLE_SIZE` | `500` | lignes chargées du dataset |

---

## 🛠️ Commandes utiles

```bash
docker compose ps              # état des services
docker compose logs -f api     # logs de l'API
docker compose stop            # arrêt (les données persistent)
docker compose down            # arrêt + suppression des conteneurs
docker compose down -v         # + suppression des données (reset complet)
```

---

## 📁 Structure du dépôt

```
src/
  common/       config, clients S3 & Elasticsearch, NLP
  ingestion/    sources : Hacker News (API) + HuggingFace (fichier)
  transform/    pipelines raw→staging→curated
  api/          API Gateway FastAPI (+ ingestion avancée)
airflow/dags/   DAG d'orchestration
docker/         Dockerfiles Airflow et API
scripts/        benchmark de performance
docs/           architecture.md, performance.md
```

---

## 👤 Auteur

**Adem Mamouni** — EFREI, spécialité Big Data & Machine Learning.
