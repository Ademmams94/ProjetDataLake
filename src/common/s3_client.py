"""
Client S3 pour MinIO — gère les zones RAW et STAGING.

MinIO expose une API 100 % compatible S3, donc on utilise boto3
(le SDK AWS officiel). Le seul changement vs le vrai AWS : on pointe
`endpoint_url` vers notre MinIO local au lieu d'Amazon.

Ce module offre des helpers simples : écrire/lire du JSON, écrire/lire
des bytes (pour le Parquet), et lister les objets d'un bucket.
"""
import io
import json
from typing import Any, Iterator

import boto3
from botocore.client import Config

from src.common.config import settings


def get_s3_client():
    """Crée un client boto3 configuré pour MinIO."""
    return boto3.client(
        "s3",
        endpoint_url=settings.minio_endpoint,
        aws_access_key_id=settings.minio_root_user,
        aws_secret_access_key=settings.minio_root_password,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",  # valeur factice, requise par boto3
    )


def ensure_bucket(bucket: str) -> None:
    """Crée le bucket s'il n'existe pas déjà (idempotent)."""
    s3 = get_s3_client()
    existing = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
    if bucket not in existing:
        s3.create_bucket(Bucket=bucket)


# ---------- JSON (un objet = un fichier .json dans le bucket) ----------

def put_json(bucket: str, key: str, obj: Any) -> None:
    """Écrit un objet Python en JSON dans S3."""
    body = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    get_s3_client().put_object(Bucket=bucket, Key=key, Body=body,
                               ContentType="application/json")


def get_json(bucket: str, key: str) -> Any:
    """Lit un fichier JSON depuis S3 et le renvoie en objet Python."""
    resp = get_s3_client().get_object(Bucket=bucket, Key=key)
    return json.loads(resp["Body"].read().decode("utf-8"))


# ---------- Bytes bruts (utilisé pour le Parquet en zone staging) ----------

def put_bytes(bucket: str, key: str, data: bytes,
              content_type: str = "application/octet-stream") -> None:
    """Écrit des bytes bruts dans S3."""
    get_s3_client().put_object(Bucket=bucket, Key=key,
                               Body=io.BytesIO(data), ContentType=content_type)


def get_bytes(bucket: str, key: str) -> bytes:
    """Lit des bytes bruts depuis S3."""
    resp = get_s3_client().get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()


# ---------- Exploration ----------

def list_keys(bucket: str, prefix: str = "") -> Iterator[str]:
    """Liste toutes les clés (chemins) d'un bucket, avec pagination."""
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"]


def count_objects(bucket: str, prefix: str = "") -> int:
    """Compte les objets d'un bucket (pour l'endpoint /stats)."""
    return sum(1 for _ in list_keys(bucket, prefix))
