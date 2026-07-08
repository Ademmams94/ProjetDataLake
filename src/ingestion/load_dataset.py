"""
Ingestion — SOURCE FICHIER : dataset HuggingFace.

Charge un échantillon d'un dataset texte (par défaut `tweet_eval/sentiment`)
et le dépose TEL QUEL dans la zone RAW (bucket S3 `raw`).

tweet_eval/sentiment : tweets annotés en sentiment
    label 0 = négatif, 1 = neutre, 2 = positif
On garde le label d'origine : il servira plus tard à ÉVALUER la qualité
de notre modèle de sentiment (comparaison prédiction vs vérité terrain).

Lancement manuel :
    python -m src.ingestion.load_dataset
"""
from __future__ import annotations

from datetime import datetime, timezone

from datasets import load_dataset

from src.common.config import settings
from src.common import s3_client


def ingest() -> dict:
    """Charge un échantillon du dataset et l'écrit dans RAW."""
    now = datetime.now(timezone.utc)

    # Chargement du dataset depuis HuggingFace (téléchargé + mis en cache local)
    ds = load_dataset(
        settings.hf_dataset,
        settings.hf_dataset_config,
        split=settings.hf_split,
    )

    # On ne garde qu'un échantillon pour rester léger
    n = min(settings.hf_sample_size, len(ds))
    sample = ds.select(range(n))

    # Conversion en liste de dictionnaires (donnée brute, non transformée)
    items = [dict(row) for row in sample]

    payload = {
        "source": "huggingface",
        "dataset": f"{settings.hf_dataset}/{settings.hf_dataset_config}",
        "split": settings.hf_split,
        "ingested_at": now.isoformat(),
        "count": len(items),
        "items": items,
    }

    key = f"huggingface/{settings.hf_dataset}/{now:%Y%m%dT%H%M%S}.json"
    s3_client.put_json(settings.raw_bucket, key, payload)

    print(f"[huggingface] {len(items)} lignes -> s3://{settings.raw_bucket}/{key}")
    return {"source": "huggingface", "key": key, "count": len(items)}


if __name__ == "__main__":
    ingest()
