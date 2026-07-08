"""
Client Elasticsearch — gère la zone CURATED.

La zone curated stocke les documents enrichis (texte + score de sentiment).
Elasticsearch est parfait ici : recherche full-text ET agrégations
(ex: sentiment moyen par subreddit, évolution dans le temps) en une requête.
"""
from typing import Any, Iterable

from elasticsearch import Elasticsearch, helpers

from src.common.config import settings


def get_es_client() -> Elasticsearch:
    """Crée un client Elasticsearch."""
    return Elasticsearch(settings.es_host, request_timeout=30)


# Mapping = le "schéma" de l'index : on type explicitement les champs
# pour que les agrégations et la recherche fonctionnent correctement.
INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "id":            {"type": "keyword"},
            "source":        {"type": "keyword"},   # 'hackernews' ou 'huggingface'
            "author":        {"type": "keyword"},   # auteur du post (si dispo)
            "text":          {"type": "text"},       # analysé pour la recherche
            "text_clean":    {"type": "text"},
            "sentiment":     {"type": "keyword"},     # positive / neutral / negative
            "sentiment_score": {"type": "float"},     # score continu [-1, 1]
            "created_at":    {"type": "date"},
            "ingested_at":   {"type": "date"},
        }
    }
}


def ensure_index(index: str | None = None) -> None:
    """Crée l'index avec son mapping s'il n'existe pas (idempotent)."""
    es = get_es_client()
    index = index or settings.es_index
    if not es.indices.exists(index=index):
        es.indices.create(index=index, body=INDEX_MAPPING)


def bulk_index(docs: Iterable[dict[str, Any]], index: str | None = None) -> int:
    """
    Indexe une liste de documents en masse (bulk = bien plus rapide
    que document par document). Renvoie le nombre de docs indexés.
    """
    es = get_es_client()
    index = index or settings.es_index
    actions = (
        {"_index": index, "_id": doc.get("id"), "_source": doc}
        for doc in docs
    )
    success, _ = helpers.bulk(es, actions)
    es.indices.refresh(index=index)  # rend les docs immédiatement cherchables
    return success


def count_docs(index: str | None = None) -> int:
    """Compte les documents de l'index (pour l'endpoint /stats)."""
    es = get_es_client()
    index = index or settings.es_index
    if not es.indices.exists(index=index):
        return 0
    return es.count(index=index)["count"]


def search(query: dict[str, Any], index: str | None = None,
           size: int = 20) -> list[dict[str, Any]]:
    """Recherche simple, renvoie la liste des documents (_source)."""
    es = get_es_client()
    index = index or settings.es_index
    if not es.indices.exists(index=index):
        return []
    resp = es.search(index=index, query=query, size=size)
    return [hit["_source"] for hit in resp["hits"]["hits"]]
