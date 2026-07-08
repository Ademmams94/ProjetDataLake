"""
Ingestion — SOURCE API : Hacker News.

Récupère les posts populaires depuis l'API publique Hacker News et les
dépose TELS QUELS dans la zone RAW (bucket S3 `raw`).

Règle du data lake : on ne transforme rien ici. On capture la donnée brute
avec quelques métadonnées d'ingestion (source, horodatage) et c'est tout.
Le nettoyage viendra à la phase staging.

Lancement manuel :
    python -m src.ingestion.hackernews_api
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests

from src.common.config import settings
from src.common import s3_client


def fetch_story_ids(story_type: str, limit: int) -> list[int]:
    """Récupère la liste des IDs de posts (top / new / best)."""
    url = f"{settings.hn_base_url}/{story_type}.json"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    ids = resp.json() or []
    return ids[:limit]


def fetch_item(item_id: int) -> dict | None:
    """Récupère un post/commentaire par son ID. None si erreur/supprimé."""
    url = f"{settings.hn_base_url}/item/{item_id}.json"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def fetch_stories(story_type: str, limit: int) -> list[dict]:
    """
    Récupère `limit` posts en parallèle (les appels HTTP sont indépendants,
    donc un pool de threads accélère fortement l'ingestion).
    """
    ids = fetch_story_ids(story_type, limit)
    with ThreadPoolExecutor(max_workers=10) as pool:
        items = list(pool.map(fetch_item, ids))
    # On garde les items valides qui contiennent du texte exploitable
    return [it for it in items if it and (it.get("title") or it.get("text"))]


def ingest() -> dict:
    """
    Point d'entrée de l'ingestion : récupère les posts et les écrit dans RAW.
    Renvoie un petit résumé (utile pour les logs Airflow).
    """
    story_type = settings.hn_story_type
    now = datetime.now(timezone.utc)
    items = fetch_stories(story_type, settings.hn_max_items)

    # Enveloppe : la donnée brute + les métadonnées d'ingestion
    payload = {
        "source": "hackernews",
        "story_type": story_type,
        "ingested_at": now.isoformat(),
        "count": len(items),
        "items": items,
    }

    # Clé partitionnée par date : facilite l'exploration et le re-traitement
    key = f"hackernews/{now:%Y-%m-%d}/{now:%Y%m%dT%H%M%S}_{story_type}.json"
    s3_client.put_json(settings.raw_bucket, key, payload)

    print(f"[hackernews] {len(items)} posts -> s3://{settings.raw_bucket}/{key}")
    return {"source": "hackernews", "key": key, "count": len(items)}


if __name__ == "__main__":
    ingest()
