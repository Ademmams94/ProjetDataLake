"""
Configuration centralisée du Data Lake.

Toutes les valeurs sont lues depuis les variables d'environnement
(fichier .env en local). On ne met JAMAIS de secret en dur dans le code.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- MinIO / S3 (zones RAW + STAGING) ---
    minio_endpoint: str = "http://localhost:9000"
    minio_root_user: str = "minioadmin"
    minio_root_password: str = "minioadmin123"
    raw_bucket: str = "raw"
    staging_bucket: str = "staging"

    # --- Elasticsearch (zone CURATED) ---
    es_host: str = "http://localhost:9200"
    es_index: str = "curated_sentiment"

    # --- API Reddit (source API) ---
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "datalake-projet-efrei/0.1"
    reddit_subreddits: str = "technology,france,artificial"

    # --- Dataset fichier (source HuggingFace) ---
    hf_dataset: str = "tweet_eval"
    hf_dataset_config: str = "sentiment"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def subreddit_list(self) -> list[str]:
        """Transforme 'technology,france' -> ['technology', 'france']."""
        return [s.strip() for s in self.reddit_subreddits.split(",") if s.strip()]


# Instance unique importée partout : `from src.common.config import settings`
settings = Settings()
