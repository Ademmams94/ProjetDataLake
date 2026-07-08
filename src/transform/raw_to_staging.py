"""
Transformation RAW -> STAGING.

Lit tous les fichiers bruts de la zone RAW (Hacker News + HuggingFace),
les nettoie et les fond dans UN SCHÉMA COMMUN, puis écrit le résultat
en Parquet dans la zone STAGING.

Schéma commun (identique quelle que soit la source) :
    id, source, author, text, text_clean, true_label, created_at, ingested_at

Lancement manuel :
    python -m src.transform.raw_to_staging
"""
from __future__ import annotations

import io
from datetime import datetime, timezone

import pandas as pd

from src.common import s3_client
from src.common.config import settings
from src.common.nlp import clean_text

# tweet_eval : mapping des labels numériques -> texte
HF_LABELS = {0: "negative", 1: "neutral", 2: "positive"}


def _epoch_to_iso(ts: int | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def normalize_hackernews(payload: dict) -> list[dict]:
    """Transforme un fichier brut Hacker News en lignes du schéma commun."""
    rows = []
    for it in payload.get("items", []):
        # Le texte à analyser = titre + éventuel corps du post
        text = " ".join(filter(None, [it.get("title"), it.get("text")]))
        rows.append({
            "id": f"hn_{it.get('id')}",
            "source": "hackernews",
            "author": it.get("by"),
            "text": text,
            "text_clean": clean_text(text),
            "true_label": None,  # HN n'a pas de label de sentiment
            "created_at": _epoch_to_iso(it.get("time")),
            "ingested_at": payload.get("ingested_at"),
        })
    return rows


def normalize_huggingface(payload: dict) -> list[dict]:
    """Transforme un fichier brut HuggingFace en lignes du schéma commun."""
    rows = []
    dataset = payload.get("dataset", "hf")
    for i, it in enumerate(payload.get("items", [])):
        text = it.get("text", "")
        rows.append({
            "id": f"hf_{dataset}_{i}",
            "source": "huggingface",
            "author": None,
            "text": text,
            "text_clean": clean_text(text),
            "true_label": HF_LABELS.get(it.get("label")),  # vérité terrain
            "created_at": payload.get("ingested_at"),
            "ingested_at": payload.get("ingested_at"),
        })
    return rows


# Aiguillage : à chaque source son normaliseur
NORMALIZERS = {
    "hackernews/": normalize_hackernews,
    "huggingface/": normalize_huggingface,
}


def transform() -> dict:
    """Lit toute la zone RAW, normalise, écrit un snapshot Parquet en STAGING."""
    all_rows: list[dict] = []

    for key in s3_client.list_keys(settings.raw_bucket):
        # On choisit le normaliseur selon le préfixe de la clé
        normalizer = next(
            (fn for prefix, fn in NORMALIZERS.items() if key.startswith(prefix)),
            None,
        )
        if normalizer is None:
            print(f"[staging] clé ignorée (source inconnue) : {key}")
            continue
        payload = s3_client.get_json(settings.raw_bucket, key)
        all_rows.extend(normalizer(payload))

    if not all_rows:
        print("[staging] aucune donnée à transformer")
        return {"count": 0, "key": None}

    # Construction de la table + dédoublonnage sur l'id
    df = pd.DataFrame(all_rows).drop_duplicates(subset="id", keep="last")
    # On écarte les lignes sans texte exploitable après nettoyage
    df = df[df["text_clean"].str.len() > 0].reset_index(drop=True)

    # Écriture Parquet en mémoire puis upload vers S3 (bucket staging)
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", compression="snappy", index=False)

    now = datetime.now(timezone.utc)
    key = f"{now:%Y-%m-%d}/part-{now:%Y%m%dT%H%M%S}.parquet"
    s3_client.put_bytes(settings.staging_bucket, key, buf.getvalue(),
                        content_type="application/octet-stream")

    by_source = df["source"].value_counts().to_dict()
    print(f"[staging] {len(df)} lignes ({by_source}) "
          f"-> s3://{settings.staging_bucket}/{key}")
    return {"count": len(df), "key": key, "by_source": by_source}


if __name__ == "__main__":
    transform()
