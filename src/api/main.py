"""
API Gateway du Data Lake (FastAPI).

Offre une interface HTTP unique pour consulter les trois zones du lake
sans avoir à parler directement à MinIO ou Elasticsearch.

Endpoints exigés par le sujet :
    GET /health    état des services (MinIO, Elasticsearch)
    GET /stats     métriques de remplissage des buckets et de l'index
    GET /raw       données brutes (listing + contenu d'un objet)
    GET /staging   données intermédiaires (lecture du Parquet)
    GET /curated   données finales (recherche full-text + filtres)

Endpoints du niveau avancé (ajoutés en phase 7) :
    POST /ingest        ingestion synchrone d'un batch de textes
    POST /ingest_fast   même chose, optimisée

Documentation interactive générée automatiquement : http://localhost:8000/docs

Lancement :
    uvicorn src.api.main:app --reload
"""
from __future__ import annotations

# ⚠️ Doit précéder pandas/pyarrow : conflit de DLL OpenMP sous Windows.
from src.common import torch_first  # noqa: F401  (voir src/common/torch_first.py)

import io
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, field_validator

from src.api.ingest import ingest_fast, ingest_naive
from src.common import es_client, s3_client
from src.common.config import settings

app = FastAPI(
    title="Data Lake — Analyse de Sentiment Social",
    description="API Gateway exposant les zones RAW, STAGING et CURATED.",
    version="1.0.0",
)


@app.get("/", include_in_schema=False)
def root():
    """Redirige vers la documentation interactive."""
    return RedirectResponse(url="/docs")


# ===============================================================
#  /health — état des services
# ===============================================================

@app.get("/health", tags=["monitoring"])
def health():
    """
    Vérifie que chaque brique du lake répond.

    Renvoie 200 si tout va bien, 503 si au moins un service est injoignable :
    c'est la convention attendue par les sondes de supervision (Kubernetes,
    load balancers…), qui se fient au code HTTP et non au corps de la réponse.
    """
    services: dict[str, dict] = {}

    # --- MinIO ---
    try:
        s3_client.get_s3_client().list_buckets()
        services["minio"] = {"status": "up", "endpoint": settings.minio_endpoint}
    except Exception as exc:
        services["minio"] = {"status": "down", "error": str(exc)[:200]}

    # --- Elasticsearch ---
    try:
        h = es_client.cluster_health()
        services["elasticsearch"] = {
            "status": "up",
            "cluster_status": h.get("status"),   # green / yellow / red
        }
    except Exception as exc:
        services["elasticsearch"] = {"status": "down", "error": str(exc)[:200]}

    healthy = all(s["status"] == "up" for s in services.values())
    payload = {
        "status": "healthy" if healthy else "degraded",
        "services": services,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if not healthy:
        raise HTTPException(status_code=503, detail=payload)
    return payload


# ===============================================================
#  /stats — métriques de remplissage
# ===============================================================

@app.get("/stats", tags=["monitoring"])
def stats():
    """Volumétrie des trois zones : objets, octets, documents, sentiments."""
    try:
        raw = s3_client.bucket_summary(settings.raw_bucket)
        # Détail par source, déduit du préfixe des clés
        raw["by_source"] = {
            src: s3_client.count_objects(settings.raw_bucket, f"{src}/")
            for src in ("hackernews", "huggingface")
        }

        staging = s3_client.bucket_summary(settings.staging_bucket)
        curated = es_client.index_stats()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Impossible de collecter les statistiques : {exc}",
        ) from exc

    return {
        "raw": {"bucket": settings.raw_bucket, **raw},
        "staging": {"bucket": settings.staging_bucket, **staging},
        "curated": {"index": settings.es_index, **curated},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ===============================================================
#  /raw — zone des données brutes
# ===============================================================

@app.get("/raw", tags=["zones"])
def get_raw(
    source: str | None = Query(None, description="Filtre : hackernews | huggingface"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Liste les objets bruts du bucket RAW (métadonnées uniquement)."""
    prefix = f"{source}/" if source else ""
    try:
        objects = s3_client.list_objects(settings.raw_bucket, prefix)
    except Exception as exc:
        raise HTTPException(status_code=503,
                            detail=f"MinIO injoignable : {exc}") from exc

    if source and not objects:
        raise HTTPException(
            status_code=404,
            detail=f"Aucun objet brut pour la source '{source}'. "
                   f"Sources valides : hackernews, huggingface.",
        )
    return {
        "bucket": settings.raw_bucket,
        "total": len(objects),
        "limit": limit,
        "offset": offset,
        "objects": objects[offset:offset + limit],
    }


@app.get("/raw/object", tags=["zones"])
def get_raw_object(
    key: str = Query(..., description="Clé complète de l'objet, ex: hackernews/2026-07-10/xxx.json"),
    max_items: int = Query(10, ge=1, le=500,
                           description="Nombre d'items du payload à renvoyer"),
):
    """
    Renvoie le contenu d'un objet brut.

    Les fichiers bruts peuvent contenir des centaines d'items : on ne renvoie
    que les `max_items` premiers pour éviter une réponse de plusieurs Mo.
    """
    if not s3_client.object_exists(settings.raw_bucket, key):
        raise HTTPException(status_code=404,
                            detail=f"Objet introuvable dans la zone raw : '{key}'")
    try:
        payload = s3_client.get_json(settings.raw_bucket, key)
    except Exception as exc:
        raise HTTPException(status_code=500,
                            detail=f"Lecture impossible : {exc}") from exc

    items = payload.get("items", [])
    return {**payload, "count": len(items), "items": items[:max_items]}


# ===============================================================
#  /staging — zone intermédiaire (Parquet)
# ===============================================================

def _latest_staging_key() -> str:
    """Clé du snapshot Parquet le plus récent, ou 404 si la zone est vide."""
    keys = sorted(s3_client.list_keys(settings.staging_bucket))
    if not keys:
        raise HTTPException(
            status_code=404,
            detail="La zone staging est vide. Lancez le DAG Airflow "
                   "ou `python -m src.transform.raw_to_staging`.",
        )
    return keys[-1]


@app.get("/staging/files", tags=["zones"])
def get_staging_files():
    """Liste les snapshots Parquet disponibles en zone STAGING."""
    try:
        objects = s3_client.list_objects(settings.staging_bucket)
    except Exception as exc:
        raise HTTPException(status_code=503,
                            detail=f"MinIO injoignable : {exc}") from exc
    return {"bucket": settings.staging_bucket, "total": len(objects),
            "files": objects}


@app.get("/staging", tags=["zones"])
def get_staging(
    source: str | None = Query(None, description="Filtre : hackernews | huggingface"),
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    key: str | None = Query(None, description="Snapshot précis (défaut : le plus récent)"),
):
    """Lit les lignes nettoyées et unifiées du dernier snapshot Parquet."""
    key = key or _latest_staging_key()
    if not s3_client.object_exists(settings.staging_bucket, key):
        raise HTTPException(status_code=404,
                            detail=f"Snapshot introuvable : '{key}'")
    try:
        df = pd.read_parquet(
            io.BytesIO(s3_client.get_bytes(settings.staging_bucket, key))
        )
    except Exception as exc:
        raise HTTPException(status_code=500,
                            detail=f"Parquet illisible : {exc}") from exc

    if source:
        df = df[df["source"] == source]
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Aucune ligne pour la source '{source}' dans ce snapshot.",
            )

    page = df.iloc[offset:offset + limit]
    # Les NaN de pandas ne sont pas du JSON valide -> on les convertit en null
    records = page.astype(object).where(pd.notna(page), None).to_dict("records")
    return {"snapshot": key, "total": len(df), "limit": limit,
            "offset": offset, "rows": records}


# ===============================================================
#  /curated — zone finale (Elasticsearch)
# ===============================================================

@app.get("/curated", tags=["zones"])
def get_curated(
    q: str | None = Query(None, description="Recherche full-text dans le texte"),
    source: str | None = Query(None, description="Filtre : hackernews | huggingface"),
    sentiment: str | None = Query(None, description="Filtre : positive | neutral | negative"),
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """
    Interroge la zone CURATED : documents enrichis d'un score de sentiment.

    Combine recherche full-text (`q`) et filtres exacts (`source`, `sentiment`).
    """
    valid_sentiments = {"positive", "neutral", "negative"}
    if sentiment and sentiment not in valid_sentiments:
        raise HTTPException(
            status_code=422,
            detail=f"Sentiment invalide : '{sentiment}'. "
                   f"Valeurs acceptées : {sorted(valid_sentiments)}.",
        )
    try:
        total, docs = es_client.search_documents(q, source, sentiment, limit, offset)
    except Exception as exc:
        raise HTTPException(status_code=503,
                            detail=f"Elasticsearch injoignable : {exc}") from exc

    return {"index": settings.es_index, "total": total, "limit": limit,
            "offset": offset, "filters": {"q": q, "source": source,
                                          "sentiment": sentiment},
            "documents": docs}


# ===============================================================
#  NIVEAU AVANCÉ — /ingest et /ingest_fast
# ===============================================================

class IngestData(BaseModel):
    """Charge utile : la liste des textes à analyser."""
    texts: list[str] = Field(..., min_length=1, max_length=1000,
                             description="Textes à ingérer (1 à 1000)")

    @field_validator("texts")
    @classmethod
    def reject_blank_texts(cls, texts: list[str]) -> list[str]:
        """Un texte vide ne produit aucun sentiment exploitable : on refuse."""
        cleaned = [t for t in texts if t and t.strip()]
        if not cleaned:
            raise ValueError("Aucun texte exploitable : tous sont vides.")
        return cleaned


class IngestRequest(BaseModel):
    """Structure JSON imposée par le sujet : {"data": {"texts": [...]}}"""
    data: IngestData

    model_config = {
        "json_schema_extra": {
            "example": {"data": {"texts": ["Premier texte à analyser",
                                           "Deuxième texte à traiter"]}}
        }
    }


def _ingest_response(result: dict, include_documents: bool) -> dict:
    """Réponse commune aux deux endpoints (mêmes champs, comparables)."""
    payload = {
        "endpoint": result["endpoint"],
        "ingested": result["ingested"],
        "elapsed_seconds": result["elapsed_seconds"],
        "preview": result["documents"][:3],
    }
    if include_documents:
        payload["documents"] = result["documents"]
    return payload


@app.post("/ingest", tags=["ingestion"])
def post_ingest(
    request: IngestRequest,
    include_documents: bool = Query(False, description="Renvoyer tous les documents"),
):
    """
    Ingère un batch de textes à travers tout le pipeline (RAW -> CURATED).

    Implémentation de RÉFÉRENCE : traitement séquentiel, un texte à la fois.
    Sert de base de comparaison à `/ingest_fast`.
    """
    try:
        result = ingest_naive(request.data.texts)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Échec de l'ingestion (S3 ou Elasticsearch injoignable) : {exc}",
        ) from exc
    return _ingest_response(result, include_documents)


@app.post("/ingest_fast", tags=["ingestion"])
def post_ingest_fast(
    request: IngestRequest,
    include_documents: bool = Query(False, description="Renvoyer tous les documents"),
):
    """
    Version optimisée de `/ingest` — résultat identique, débit très supérieur.

    Optimisations : inférence par lots, indexation bulk Elasticsearch,
    écritures S3 parallèles recouvrant le calcul, `torch.inference_mode()`.
    Détail complet dans `src/api/ingest.py` et `docs/performance.md`.
    """
    try:
        result = ingest_fast(request.data.texts)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Échec de l'ingestion (S3 ou Elasticsearch injoignable) : {exc}",
        ) from exc
    return _ingest_response(result, include_documents)
