"""
Pipeline d'ingestion à la demande — niveau avancé du sujet.

Deux implémentations produisant EXACTEMENT le même résultat :

  * `ingest_naive` : implémentation directe, item par item. C'est ce qu'on
    écrit spontanément quand on ne pense pas encore à la performance.
  * `ingest_fast`  : même pipeline, optimisé. Objectif : > 30 % plus rapide.

Les deux traversent les 3 zones du lake :
    texte -> RAW (S3) -> nettoyage + sentiment -> CURATED (Elasticsearch)

--------------------------------------------------------------------
 OPTIMISATIONS PROPRES À `ingest_fast`
--------------------------------------------------------------------
 1. INFÉRENCE PAR LOTS (le gain dominant à grand batch)
    Le naïf appelle le Transformer une fois par texte : N passes avant.
    Le rapide empile les N textes dans un seul tenseur : une seule passe,
    exploitant les multiplications matricielles vectorisées de PyTorch.
    Gain croissant avec la taille du batch (mesuré : ×7 à N=100).

 2. INDEXATION BULK ELASTICSEARCH
    Le naïf fait N indexations HTTP (une par document).
    Le rapide envoie un seul appel `_bulk`. On paie une latence réseau
    au lieu de N.

 3. ÉCRITURES S3 PARALLÈLES
    Les N écritures dans RAW sont indépendantes et I/O-bound (on attend
    le réseau). Un pool de threads les recouvre au lieu de les enchaîner.

 4. RECOUVREMENT I/O / CALCUL
    Les écritures RAW ne dépendent pas du scoring : le rapide lance les
    écritures S3 en tâche de fond PENDANT que le CPU calcule le sentiment.
    Le temps d'écriture devient quasi gratuit.

--------------------------------------------------------------------
 OPTIMISATIONS PARTAGÉES PAR LES DEUX ENDPOINTS
--------------------------------------------------------------------
 Appliquées dans les couches communes, elles accélèrent AUSSI le naïf,
 donc ne faussent pas la comparaison :
   - `torch.inference_mode()` dans le scoring (+16 % sur l'inférence) ;
   - clients S3/Elasticsearch mis en cache (`lru_cache`) ;
   - `translog.durability: async` sur l'index (+21 % sur les écritures ES).

--------------------------------------------------------------------
 OPTIMISATION TESTÉE PUIS REJETÉE — quantification int8
--------------------------------------------------------------------
 `torch.quantization.quantize_dynamic` sur les couches Linear : mesurée
 2,5× PLUS LENTE sur ce CPU (noyaux int8 non optimisés) et seulement
 85,5 % d'accord de labels avec le modèle fp32. Rejetée, chiffres à
 l'appui — voir docs/performance.md.

--------------------------------------------------------------------
 POURQUOI PAS NUMBA ?
--------------------------------------------------------------------
 Le sujet suggère Numba, qui compile en code machine les boucles Python
 sur des tableaux NumPy. Ici, ce serait inutile : le goulot d'étranglement
 n'est pas une boucle numérique Python, mais (a) l'inférence d'un réseau
 de neurones — déjà en C++/ATen optimisé — et (b) les entrées/sorties
 réseau vers S3 et Elasticsearch. Optimiser ce qui n'est pas le goulot
 d'étranglement ne sert à rien : on cible le batching et l'I/O.
"""
from __future__ import annotations

# ⚠️ Doit précéder pandas/pyarrow : conflit de DLL OpenMP sous Windows.
from src.common import torch_first  # noqa: F401

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from src.common import es_client, s3_client
from src.common.config import settings
from src.common.nlp import clean_text, score_batch_transformer

# Nombre de threads pour les écritures S3 parallèles (charge I/O, pas CPU).
S3_WRITE_WORKERS = 16


def _doc_id(text: str) -> str:
    """
    Identifiant déterministe dérivé du texte.

    Conséquence : ré-ingérer le même texte MET À JOUR le document au lieu
    d'en créer un doublon. Le pipeline est donc idempotent.
    """
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return f"api_{digest}"


def _raw_key(doc_id: str, now: datetime) -> str:
    """Clé S3 de l'objet brut, partitionnée par date comme le reste du lake."""
    return f"api/{now:%Y-%m-%d}/{doc_id}.json"


def _build_document(text: str, label: str, score: float,
                    now: datetime) -> dict:
    """Construit un document au schéma commun de la zone curated."""
    return {
        "id": _doc_id(text),
        "source": "api",
        "author": None,
        "text": text,
        "text_clean": clean_text(text),
        "true_label": None,
        "sentiment": label,
        "sentiment_score": score,
        "created_at": now.isoformat(),
        "ingested_at": now.isoformat(),
    }


def _raw_payload(text: str, now: datetime) -> dict:
    """Enveloppe brute écrite dans RAW (donnée d'origine + métadonnées)."""
    return {
        "source": "api",
        "ingested_at": now.isoformat(),
        "count": 1,
        "items": [{"id": _doc_id(text), "text": text}],
    }


# ===============================================================
#  Implémentation NAÏVE — référence de performance
# ===============================================================

def ingest_naive(texts: list[str]) -> dict:
    """
    Traite les textes UN PAR UN, de bout en bout.

    Pour chaque texte : une écriture S3, une inférence du modèle,
    une indexation Elasticsearch. Simple à lire, lent à l'échelle.
    """
    started = time.perf_counter()
    now = datetime.now(timezone.utc)
    es = es_client.get_es_client()
    es_client.ensure_index()

    documents = []
    for text in texts:
        doc_id = _doc_id(text)

        # 1. Zone RAW : une écriture réseau par texte
        s3_client.put_json(settings.raw_bucket, _raw_key(doc_id, now),
                           _raw_payload(text, now))

        # 2. Sentiment : une passe du Transformer par texte (batch de 1)
        cleaned = clean_text(text)
        label, score = score_batch_transformer([cleaned], batch_size=1)[0]

        # 3. Zone CURATED : une requête HTTP d'indexation par texte
        doc = _build_document(text, label, score, now)
        es.index(index=settings.es_index, id=doc_id, document=doc)
        documents.append(doc)

    # Rend les documents immédiatement cherchables (même coût des deux côtés)
    es.indices.refresh(index=settings.es_index)

    return {
        "endpoint": "ingest",
        "ingested": len(documents),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "documents": documents,
    }


# ===============================================================
#  Implémentation OPTIMISÉE
# ===============================================================

def ingest_fast(texts: list[str]) -> dict:
    """
    Même pipeline, même résultat, mais pensé pour le débit.

    Voir l'en-tête du module pour le détail des 5 optimisations.
    """
    started = time.perf_counter()
    now = datetime.now(timezone.utc)
    es_client.ensure_index()

    # --- Nettoyage de tous les textes en une passe ---
    cleaned = [clean_text(t) for t in texts]

    # --- Optim. 3 + 4 : les écritures RAW partent en tâche de fond,
    #     en parallèle, PENDANT que le modèle calcule le sentiment.
    pool = ThreadPoolExecutor(max_workers=S3_WRITE_WORKERS)
    raw_futures = [
        pool.submit(s3_client.put_json, settings.raw_bucket,
                    _raw_key(_doc_id(t), now), _raw_payload(t, now))
        for t in texts
    ]

    # --- Optim. 1 : une seule inférence vectorisée pour tout le batch ---
    # (le `torch.inference_mode()` est appliqué dans score_batch_transformer,
    #  donc les deux endpoints en profitent : ce n'est pas un biais du fast.)
    scores = score_batch_transformer(cleaned,
                                     batch_size=settings.sentiment_batch_size)

    documents = [
        _build_document(text, label, score, now)
        for text, (label, score) in zip(texts, scores)
    ]

    # --- Optim. 2 : une seule requête _bulk au lieu de N indexations ---
    es_client.bulk_index(documents)

    # On s'assure que les écritures RAW sont bien terminées avant de répondre
    # (et on propage toute exception survenue dans un thread).
    for future in raw_futures:
        future.result()
    pool.shutdown(wait=True)

    return {
        "endpoint": "ingest_fast",
        "ingested": len(documents),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "documents": documents,
    }
