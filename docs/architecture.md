# Architecture technique — Data Lake d'analyse de sentiment

Ce document décrit l'architecture de la solution, les choix techniques et leur
justification. Pour les instructions de build, voir le [README](../README.md).
Pour les résultats de performance, voir [performance.md](performance.md).

---

## 1. Vue d'ensemble

Le projet implémente un data lake de bout en bout : deux sources hétérogènes
sont ingérées, raffinées à travers trois zones, puis exposées via une API. Le
domaine applicatif est l'**analyse de sentiment** de texte social.

```
 SOURCES              ZONE RAW          ZONE STAGING        ZONE CURATED         API
 ───────              ────────          ───────────         ────────────         ───
 Hacker News  ─┐      MinIO (S3)        MinIO (S3)          Elasticsearch        FastAPI
 (API)         ├───►  JSON brut   ───►  Parquet       ───►  documents      ───►  /raw /staging
 HuggingFace  ─┘      tel quel          nettoyé,            + sentiment          /curated /health
 (fichier)            partitionné       schéma unifié       (RoBERTa)            /stats /ingest*

                      └──────────── orchestration : Apache Airflow ─────────────┘
```

Tout s'exécute en conteneurs Docker, décrits dans un unique `docker-compose.yml`.

---

## 2. Les trois zones du lake

Le sujet impose S3 ou Elasticsearch pour la zone raw, et laisse le reste libre.
Nous avons retenu l'architecture classique **raw / staging / curated** (dite
*medallion* : bronze / argent / or), chaque zone ayant un rôle précis.

### 2.1 RAW — la donnée brute (MinIO / S3)

- **Technologie** : MinIO, un stockage objet 100 % compatible S3, exécuté
  localement. Il satisfait la contrainte du sujet sans dépendre d'AWS.
- **Contenu** : la donnée telle que la source la fournit, au format JSON, avec
  une fine enveloppe de métadonnées d'ingestion (source, horodatage, volume).
- **Principe** : on ne transforme jamais la donnée brute. Elle est la **source
  de vérité** : toute zone en aval peut être reconstruite en la rejouant.
- **Organisation** : clés partitionnées par source et par date, p. ex.
  `hackernews/2026-07-10/20260710T120000_topstories.json`. Le partitionnement
  rend le lac navigable et permet le re-traitement ciblé d'une journée.

### 2.2 STAGING — la donnée nettoyée et unifiée (Parquet sur MinIO)

- **Technologie** : fichiers **Parquet** (format colonne, compressé) stockés
  dans un bucket MinIO dédié.
- **Rôle** : c'est l'étape d'**intégration**. Les deux sources, de structures
  très différentes, sont fondues dans un **schéma commun** unique :

  | Colonne | Description |
  |---|---|
  | `id` | identifiant unique (`hn_<id>`, `hf_<dataset>_<i>`) |
  | `source` | `hackernews` \| `huggingface` |
  | `author` | auteur si disponible |
  | `text` | texte d'origine |
  | `text_clean` | texte nettoyé (HTML, URLs, espaces) |
  | `true_label` | vérité terrain de sentiment (dataset HF uniquement) |
  | `created_at`, `ingested_at` | horodatages |

- **Pourquoi Parquet** : orienté colonne, il compresse bien et accélère les
  lectures analytiques (on peut ne lire qu'une colonne). C'est le format
  standard de la donnée analytique.

### 2.3 CURATED — la donnée enrichie et exploitable (Elasticsearch)

- **Technologie** : **Elasticsearch**. Choisi car le domaine est le texte :
  il offre la recherche *full-text* (index inversé) **et** des agrégations
  rapides (sentiment moyen par source, distribution, tendances) dans une même
  requête. Un simple SGBD relationnel serait moins adapté à ces deux besoins.
- **Contenu** : chaque document du staging, enrichi de deux champs calculés :
  `sentiment` (label) et `sentiment_score` (score continu dans [-1, 1]).
- **Idempotence** : chaque document est indexé avec son `id` métier comme `_id`
  Elasticsearch. Rejouer le pipeline **met à jour** les documents au lieu de
  créer des doublons.

---

## 3. Les deux sources de données

| | Source API | Source fichier |
|---|---|---|
| **Fournisseur** | Hacker News (API Firebase publique) | HuggingFace `cardiffnlp/tweet_eval` |
| **Nature** | posts tech en temps réel | tweets annotés en sentiment |
| **Authentification** | aucune | aucune |
| **Intérêt** | flux vivant, schedulé par Airflow | vérité terrain pour évaluer le modèle |

> **Note** : le choix initial pour la source API était Reddit, abandonné car la
> plateforme a fermé la création d'applications *legacy*. La couche d'ingestion
> ayant été conçue pour découpler la source du reste du pipeline, la bascule
> vers Hacker News n'a impacté qu'un seul fichier.

Le dataset HF conserve son **label d'origine** (`true_label`), ce qui permet
d'**évaluer automatiquement** la qualité du modèle à chaque exécution
(précision globale + rappel par classe).

---

## 4. Le modèle de sentiment (composante Deep Learning)

Deux moteurs sont implémentés, sélectionnables par la variable d'environnement
`SENTIMENT_ENGINE` :

| Moteur | Nature | Précision* | Usage |
|---|---|---|---|
| **RoBERTa** (défaut) | Transformer fine-tuné sur ~124M tweets | **75,6 %** | production |
| VADER | lexique + règles | 52,8 % | baseline de comparaison |

\* mesurée sur les 500 tweets annotés de `tweet_eval`, contre une baseline
naïve « tout neutre » à 43,4 %.

**Pourquoi RoBERTa** : VADER, purement lexical, plafonne sur du texte à 3
classes (il ne comprend ni contexte ni sarcasme). Le Transformer
`cardiffnlp/twitter-roberta-base-sentiment-latest`, entraîné précisément sur ce
type de données, apporte +23 points de précision et un rappel équilibré entre
les trois classes (0,77 / 0,76 / 0,75). Il matérialise la composante Deep
Learning valorisée par le sujet.

---

## 5. Orchestration (Apache Airflow)

Le pipeline complet est automatisé par un DAG Airflow (`datalake_pipeline`) :

```
   ingest_hackernews ─┐
                      ├─►  raw_to_staging  ─►  staging_to_curated  ─►  report
   ingest_dataset ────┘
```

- **Parallélisme** : les deux ingestions, indépendantes, s'exécutent en
  parallèle ; la transformation attend leur convergence (*fan-in*).
- **Scheduling** : le DAG tourne toutes les heures (`schedule=timedelta(hours=1)`),
  ce qui alimente le lac en continu depuis l'API — l'usage du scheduling
  recommandé par le sujet.
- **XCom** : la tâche `report` récupère les valeurs de retour des tâches amont
  (volumes ingérés, métriques de qualité) pour produire une synthèse de cycle.
- **Robustesse** : chaque tâche a `retries=2` et un `execution_timeout`, pour
  encaisser une panne réseau transitoire d'une API.

**Choix Airflow plutôt que DVC** : le sujet propose les deux ; Airflow apporte
en plus le scheduling temps réel et une interface de supervision, plus proche
d'un usage de production.

---

## 6. API Gateway (FastAPI)

FastAPI a été retenu pour son asynchronisme natif (utile à l'ingestion), sa
validation par Pydantic et sa documentation OpenAPI auto-générée (`/docs`).

| Endpoint | Rôle |
|---|---|
| `GET /health` | état des services ; renvoie 503 si un service est down |
| `GET /stats` | volumétrie des 3 zones (objets, octets, documents, sentiments) |
| `GET /raw` | listing des objets bruts (+ `/raw/object` pour le contenu) |
| `GET /staging` | lecture du dernier snapshot Parquet, filtrable |
| `GET /curated` | recherche full-text + filtres (source, sentiment) + pagination |
| `POST /ingest` | **[avancé]** ingestion synchrone d'un batch (implémentation de référence) |
| `POST /ingest_fast` | **[avancé]** même résultat, optimisé (voir performance.md) |

La **gestion d'erreurs** est explicite et suit les conventions HTTP : 404
(introuvable), 422 (valeur invalide), 503 (service injoignable), 500 (erreur
interne), avec à chaque fois un message actionnable.

---

## 7. Le niveau avancé : /ingest vs /ingest_fast

`/ingest_fast` produit un résultat **fonctionnellement identique** à `/ingest`
(labels rigoureusement égaux, scores à `1e-5` près) tout en étant nettement plus
rapide. Résultats : **+87 % (×7,9) sur un batch de 100**, objectif des 30 %
franchi dès un batch de 2. Le détail des optimisations (batching d'inférence,
indexation bulk, écritures S3 parallèles recouvrant le calcul, `inference_mode`,
`translog async`), l'analyse du cas limite N=1 (loi d'Amdahl) et les
optimisations testées puis rejetées (quantification int8) sont documentés dans
[performance.md](performance.md).

---

## 8. Choix d'infrastructure

- **Tout en Docker Compose** : une seule commande (`docker compose up`) démarre
  MinIO, Elasticsearch, Postgres, Airflow (webserver + scheduler) et l'API.
  Reproductibilité maximale, quel que soit le poste.
- **Images custom** pour Airflow et l'API : elles embarquent nos dépendances
  (PyTorch CPU, transformers…). PyTorch est installé via l'index CPU dédié pour
  éviter ~2,5 Go de librairies CUDA inutiles.
- **Cache de modèle partagé** : un volume `hf_cache` mutualisé entre Airflow et
  l'API évite de retélécharger RoBERTa (~500 Mo).
- **Configuration externalisée** : toute la config passe par variables
  d'environnement (`src/common/config.py` via pydantic-settings). Aucun secret
  ni aucune adresse en dur dans le code.

---

## 9. Structure du code

```
src/
  common/          code transverse
    config.py      configuration centralisée (variables d'environnement)
    s3_client.py   accès MinIO/S3 (zones raw + staging)
    es_client.py   accès Elasticsearch (zone curated)
    nlp.py         nettoyage de texte + moteurs de sentiment (VADER, RoBERTa)
    torch_first.py contournement d'un conflit de DLL Windows (torch avant pyarrow)
  ingestion/       les deux sources
    hackernews_api.py
    load_dataset.py
  transform/       les pipelines entre zones
    raw_to_staging.py
    staging_to_curated.py
  api/             l'API Gateway
    main.py        endpoints FastAPI
    ingest.py      pipeline d'ingestion à la demande (naive + fast)
airflow/dags/
  datalake_dag.py  le DAG d'orchestration
scripts/
  benchmark.py     tests de performance (génère docs/performance.md)
docker/
  Dockerfile.airflow, Dockerfile.api
```

---

## 10. Difficultés rencontrées et résolues

Ces points illustrent la démarche de mise au point du projet :

1. **Reddit a fermé son API legacy** → bascule vers Hacker News, absorbée par le
   découplage source/pipeline (un seul fichier modifié).
2. **Python 3.13 trop récent** → certaines libs sans wheel précompilé ; montée
   de `pyarrow` (18.1) et `numba` (0.61).
3. **Dataset HF non namespacé** → `tweet_eval` refusé par les versions récentes
   de `huggingface_hub` ; usage de `cardiffnlp/tweet_eval`.
4. **Conflit de DLL sous Windows** → PyTorch et PyArrow embarquent deux runtimes
   OpenMP ; PyTorch doit être importé en premier (`src/common/torch_first.py`).
5. **Permissions du volume de cache Airflow** → le volume appartenait à `root`
   alors qu'Airflow tourne en uid 50000 ; corrigé par un `chown` dans l'image.
6. **`bulk_index` plus lent qu'une indexation unitaire sur un batch de 1** →
   causé par un `refresh` séparé (2 allers-retours) ; corrigé en passant
   `refresh=True` dans la requête bulk.
