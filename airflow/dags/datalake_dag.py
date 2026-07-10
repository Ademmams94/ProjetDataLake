"""
DAG principal du Data Lake — alimentation automatisée de bout en bout.

Chaîne exécutée à chaque cycle :

    ingest_hackernews ─┐
                       ├─> raw_to_staging ─> staging_to_curated ─> report
    ingest_dataset ────┘

Les deux ingestions sont indépendantes : Airflow les exécute EN PARALLÈLE,
puis attend les deux avant de lancer la transformation (fan-in).

Note d'implémentation : les imports de notre code sont faits À L'INTÉRIEUR
des fonctions, pas au niveau du module. Airflow re-parse tous les fichiers
de DAG toutes les 30 secondes ; importer PyTorch à chaque parse mettrait le
scheduler à genoux. Règle d'or : un fichier de DAG doit se parser en < 1 s.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Paramètres appliqués par défaut à toutes les tâches.
DEFAULT_ARGS = {
    "owner": "adem",
    "retries": 2,                            # une API peut échouer ponctuellement
    "retry_delay": timedelta(minutes=1),
    "execution_timeout": timedelta(minutes=30),
}


# ---------------------------------------------------------------
#  Fonctions exécutées par les tâches
#  Chacune renvoie un dict, automatiquement poussé dans XCom par Airflow.
# ---------------------------------------------------------------

def task_ingest_hackernews() -> dict:
    """Source API : récupère les posts Hacker News -> zone RAW."""
    from src.ingestion.hackernews_api import ingest
    return ingest()


def task_ingest_dataset() -> dict:
    """Source fichier : charge le dataset HuggingFace -> zone RAW."""
    from src.ingestion.load_dataset import ingest
    return ingest()


def task_raw_to_staging() -> dict:
    """Nettoie et unifie les données brutes -> zone STAGING (Parquet)."""
    from src.transform.raw_to_staging import transform
    return transform()


def task_staging_to_curated() -> dict:
    """Score le sentiment (RoBERTa) -> zone CURATED (Elasticsearch)."""
    from src.transform.staging_to_curated import transform
    return transform()


def task_report(ti) -> None:
    """
    Démonstration de XCom : on récupère les valeurs renvoyées par les tâches
    précédentes pour construire un rapport de synthèse du cycle.

    `ti` (TaskInstance) est injecté automatiquement par Airflow.
    """
    hn = ti.xcom_pull(task_ids="ingest_hackernews") or {}
    hf = ti.xcom_pull(task_ids="ingest_dataset") or {}
    staging = ti.xcom_pull(task_ids="raw_to_staging") or {}
    curated = ti.xcom_pull(task_ids="staging_to_curated") or {}

    print("=" * 60)
    print("RAPPORT DU CYCLE D'INGESTION")
    print("=" * 60)
    print(f"  RAW     | Hacker News : {hn.get('count', 0)} posts")
    print(f"  RAW     | HuggingFace : {hf.get('count', 0)} textes")
    print(f"  STAGING | {staging.get('count', 0)} lignes unifiées "
          f"({staging.get('by_source', {})})")
    print(f"  CURATED | {curated.get('count', 0)} documents indexés")
    print(f"  CURATED | distribution : {curated.get('distribution', {})}")

    metrics = curated.get("metrics")
    if metrics:
        print(f"  QUALITÉ | moteur '{metrics['engine']}' -> "
              f"précision {metrics['accuracy']:.1%} "
              f"sur {metrics['evaluated_on']} textes annotés")
    print("=" * 60)


# ---------------------------------------------------------------
#  Définition du DAG
# ---------------------------------------------------------------

with DAG(
    dag_id="datalake_pipeline",
    description="Ingestion + raffinage RAW -> STAGING -> CURATED",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 1, 1),
    schedule=timedelta(hours=1),   # ingestion de l'API à intervalle régulier
    catchup=False,                 # pas de rattrapage des exécutions passées
    max_active_runs=1,             # jamais deux cycles en parallèle
    tags=["datalake", "nlp", "sentiment"],
    doc_md=__doc__,
) as dag:

    ingest_hackernews = PythonOperator(
        task_id="ingest_hackernews",
        python_callable=task_ingest_hackernews,
        doc_md="Source API — posts Hacker News vers la zone RAW.",
    )

    ingest_dataset = PythonOperator(
        task_id="ingest_dataset",
        python_callable=task_ingest_dataset,
        doc_md="Source fichier — dataset HuggingFace vers la zone RAW.",
    )

    raw_to_staging = PythonOperator(
        task_id="raw_to_staging",
        python_callable=task_raw_to_staging,
        doc_md="Nettoyage + unification des schémas, écriture Parquet.",
    )

    staging_to_curated = PythonOperator(
        task_id="staging_to_curated",
        python_callable=task_staging_to_curated,
        doc_md="Scoring de sentiment par Transformer, indexation Elasticsearch.",
    )

    report = PythonOperator(
        task_id="report",
        python_callable=task_report,
        doc_md="Synthèse du cycle, construite à partir des XCom.",
    )

    # Les 2 ingestions en parallèle, puis la chaîne de raffinage.
    [ingest_hackernews, ingest_dataset] >> raw_to_staging >> staging_to_curated >> report
