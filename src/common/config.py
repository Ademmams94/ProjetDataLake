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

    # --- API Hacker News (source API, sans authentification) ---
    hn_base_url: str = "https://hacker-news.firebaseio.com/v0"
    hn_story_type: str = "topstories"   # topstories | newstories | beststories
    hn_max_items: int = 50

    # --- Dataset fichier (source HuggingFace) ---
    hf_dataset: str = "cardiffnlp/tweet_eval"
    hf_dataset_config: str = "sentiment"
    hf_split: str = "train"
    hf_sample_size: int = 500        # nb de lignes chargées (échantillon)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def subreddit_list(self) -> list[str]:
        """Transforme 'technology,france' -> ['technology', 'france']."""
        return [s.strip() for s in self.reddit_subreddits.split(",") if s.strip()]


# Instance unique importée partout : `from src.common.config import settings`
settings = Settings()
