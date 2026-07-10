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
            "sentiment":     {"type": "keyword"},     # prédiction du modèle
            "sentiment_score": {"type": "float"},     # score continu [-1, 1]
            "true_label":    {"type": "keyword"},     # vérité terrain (dataset HF)
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


def search_documents(q: str | None = None, source: str | None = None,
                     sentiment: str | None = None, limit: int = 20,
                     offset: int = 0) -> tuple[int, list[dict[str, Any]]]:
    """
    Recherche filtrée et paginée pour l'endpoint /curated.

    Construit une requête booléenne Elasticsearch :
    - `must`   : la recherche full-text (score de pertinence)
    - `filter` : les filtres exacts (pas de score, donc mis en cache par ES)
    Renvoie (nombre total de résultats, page de documents).
    """
    es = get_es_client()
    index = settings.es_index
    if not es.indices.exists(index=index):
        return 0, []

    must: list[dict] = []
    filters: list[dict] = []
    if q:
        must.append({"match": {"text": q}})
    if source:
        filters.append({"term": {"source": source}})
    if sentiment:
        filters.append({"term": {"sentiment": sentiment}})

    query = {"bool": {"must": must or [{"match_all": {}}], "filter": filters}}
    resp = es.search(index=index, query=query, size=limit, from_=offset,
                     track_total_hits=True)
    total = resp["hits"]["total"]["value"]
    return total, [hit["_source"] for hit in resp["hits"]["hits"]]


def cluster_health() -> dict[str, Any]:
    """État du cluster Elasticsearch (pour /health)."""
    return dict(get_es_client().cluster.health())


def index_stats() -> dict[str, Any]:
    """
    Statistiques de la zone CURATED (pour /stats) : volume, taille sur disque,
    et répartition des sentiments — le tout en une seule requête agrégée.
    """
    es = get_es_client()
    index = settings.es_index
    if not es.indices.exists(index=index):
        return {"documents": 0, "size_bytes": 0,
                "by_sentiment": {}, "by_source": {}, "avg_sentiment_score": None}

    stats = es.indices.stats(index=index)
    size_bytes = stats["indices"][index]["total"]["store"]["size_in_bytes"]

    resp = es.search(index=index, size=0, aggs={
        "by_sentiment": {"terms": {"field": "sentiment"}},
        "by_source": {"terms": {"field": "source"}},
        "avg_score": {"avg": {"field": "sentiment_score"}},
    })
    aggs = resp["aggregations"]
    return {
        "documents": resp["hits"]["total"]["value"],
        "size_bytes": size_bytes,
        "by_sentiment": {b["key"]: b["doc_count"]
                         for b in aggs["by_sentiment"]["buckets"]},
        "by_source": {b["key"]: b["doc_count"]
                      for b in aggs["by_source"]["buckets"]},
        "avg_sentiment_score": aggs["avg_score"]["value"],
    }
