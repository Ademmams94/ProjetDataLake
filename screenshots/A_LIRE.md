# Captures d'écran à déposer ici

Enregistre chaque capture dans CE dossier, avec **exactement** le nom indiqué
(format `.png`). Les captures manquantes seront simplement ignorées lors de la
génération du rapport — tu peux donc n'en faire qu'une partie.

Les ⭐ sont les indispensables.

## 1 — Infrastructure Docker

| Fichier | Contenu | Commande / URL |
|---|---|---|
| ⭐ `1.1-docker-ps.png` | Les 6 services `Up` / `healthy` | `docker compose ps` |
| `1.2-docker-images.png` | Les images construites | `docker images` |

## 2 — MinIO (zones RAW et STAGING) — http://localhost:9001

| Fichier | Contenu |
|---|---|
| `2.1-minio-buckets.png` | Les buckets `raw` et `staging` |
| ⭐ `2.2-minio-raw-sources.png` | Dans `raw` : les dossiers `hackernews/` et `huggingface/` (partitionnement par source) |
| `2.3-minio-raw-date.png` | Dans `raw/hackernews/` : les dossiers de dates (partitionnement par date) |
| `2.4-minio-staging-parquet.png` | Dans `staging` : les fichiers `.parquet` |

## 3 — Airflow (orchestration) — http://localhost:8080  (admin/admin)

| Fichier | Contenu |
|---|---|
| `3.1-airflow-dags.png` | Le DAG `datalake_pipeline` actif |
| ⭐ `3.2-airflow-graph.png` | Onglet **Graph** : les 2 ingestions en parallèle puis la convergence |
| `3.3-airflow-grid.png` | Onglet **Grid** : historique des runs, tous verts |
| ⭐ `3.4-airflow-report-log.png` | Log de la tâche `report` : le rapport XCom (volumes + précision 75,6 %) |

## 4 — API Gateway — http://localhost:8000/docs

| Fichier | Contenu |
|---|---|
| ⭐ `4.1-api-swagger.png` | La liste des 9 endpoints |
| `4.2-api-health.png` | `GET /health` exécuté (réponse `healthy`) |
| `4.3-api-stats.png` | `GET /stats` exécuté (volumétrie des 3 zones) |
| `4.4-api-curated.png` | `GET /curated` avec `q=AI` et `sentiment=negative` |
| ⭐ `4.5-api-ingest-fast.png` | `POST /ingest_fast` exécuté (sentiment calculé + `elapsed_seconds`) |

## 5 — Niveau avancé (performance)

| Fichier | Contenu | Commande |
|---|---|---|
| ⭐ `5.1-benchmark.png` | Le tableau de synthèse du benchmark | `.venv\Scripts\activate` puis `python scripts/benchmark.py --sizes 1 2 5 10 25 50 100` |
| `5.2-performance-md.png` | `docs/performance.md` en aperçu Markdown (VS Code : `Ctrl+Shift+V`) |

## 6 — Code et Git

| Fichier | Contenu | Commande |
|---|---|---|
| ⭐ `6.1-git-log.png` | L'historique des commits | `git log --oneline` |
| `6.2-arborescence.png` | L'explorateur VS Code déplié (src/, airflow/, docker/, docs/) |
| `6.3-github.png` | La page GitHub du dépôt (une fois pushé) |

## 7 — Bonus

| Fichier | Contenu | URL |
|---|---|---|
| `7.1-es-aggregation.png` | Agrégation Elasticsearch brute | `http://localhost:9200/curated_sentiment/_search?size=0&pretty` |
