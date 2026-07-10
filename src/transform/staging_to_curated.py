"""
Transformation STAGING -> CURATED.

Lit le dernier snapshot Parquet de la zone STAGING, calcule un score de
sentiment sur chaque texte (VADER), puis indexe les documents enrichis
dans Elasticsearch (zone CURATED).

Bonus qualité : comme le dataset HuggingFace fournit une VÉRITÉ TERRAIN
(true_label), on mesure au passage la précision du modèle. Un pipeline qui
s'auto-évalue, c'est un pipeline qu'on peut surveiller dans le temps.

L'indexation est IDEMPOTENTE : chaque document est indexé avec son `id`
comme identifiant Elasticsearch. Relancer le pipeline met à jour les
documents existants au lieu de créer des doublons.

Lancement manuel :
    python -m src.transform.staging_to_curated
"""
from __future__ import annotations

# ⚠️ Doit précéder pandas/pyarrow : conflit de DLL OpenMP sous Windows.
from src.common import torch_first  # noqa: F401  (voir src/common/torch_first.py)

import io

import pandas as pd

from src.common import es_client, s3_client
from src.common.config import settings
from src.common.nlp import score_texts


def _latest_staging_key() -> str | None:
    """Renvoie la clé du snapshot Parquet le plus récent (tri lexicographique
    suffisant car les clés sont horodatées au format ISO)."""
    keys = sorted(s3_client.list_keys(settings.staging_bucket))
    return keys[-1] if keys else None


def _read_staging(key: str) -> pd.DataFrame:
    """Charge un Parquet depuis S3 en DataFrame."""
    raw = s3_client.get_bytes(settings.staging_bucket, key)
    return pd.read_parquet(io.BytesIO(raw))


def evaluate(df: pd.DataFrame) -> dict | None:
    """
    Compare les prédictions du modèle à la vérité terrain, sur les lignes
    qui en possèdent une (celles issues du dataset HuggingFace).

    On calcule la précision globale ET le rappel par classe : une précision
    globale peut masquer un modèle qui ignore complètement une classe rare.
    """
    labeled = df[df["true_label"].notna()]
    if labeled.empty:
        return None

    correct = (labeled["sentiment"] == labeled["true_label"])
    # `float(...)` : pandas renvoie des np.float64, non sérialisables en JSON
    # (nécessaire pour les XCom Airflow et les réponses de l'API).
    per_class = {
        cls: round(float(correct[labeled["true_label"] == cls].mean()), 4)
        for cls in sorted(labeled["true_label"].unique())
    }
    return {
        "engine": settings.sentiment_engine,
        "evaluated_on": int(len(labeled)),
        "accuracy": round(float(correct.mean()), 4),
        "recall_per_class": per_class,
    }


def transform() -> dict:
    """Score le sentiment du dernier staging et indexe dans Elasticsearch."""
    key = _latest_staging_key()
    if key is None:
        print("[curated] zone staging vide, rien à faire")
        return {"count": 0}

    df = _read_staging(key)

    # --- Cœur NLP : scoring par lots (le moteur est piloté par la config) ---
    scores = score_texts(df["text_clean"].tolist())
    df["sentiment"] = [s[0] for s in scores]
    df["sentiment_score"] = [s[1] for s in scores]

    # Pandas utilise NaN pour les valeurs manquantes ; Elasticsearch veut null.
    df = df.astype(object).where(pd.notna(df), None)

    docs = df.to_dict(orient="records")

    es_client.ensure_index()
    indexed = es_client.bulk_index(docs)

    metrics = evaluate(df)
    distribution = df["sentiment"].value_counts().to_dict()

    print(f"[curated] {indexed} documents indexés dans '{settings.es_index}' "
          f"(source : s3://{settings.staging_bucket}/{key})")
    print(f"[curated] moteur : {settings.sentiment_engine}")
    print(f"[curated] distribution des sentiments : {distribution}")
    if metrics:
        print(f"[curated] précision vs vérité terrain : "
              f"{metrics['accuracy']:.1%} sur {metrics['evaluated_on']} textes annotés")
        print(f"[curated] rappel par classe : {metrics['recall_per_class']}")

    return {"count": indexed, "distribution": distribution, "metrics": metrics}


if __name__ == "__main__":
    transform()
