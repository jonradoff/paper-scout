"""Configuration via pydantic-settings. All values from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # MongoDB
    mongodb_uri: str = "mongodb://localhost:27017"
    db_name: str = "paper_scout"

    # Semantic Scholar
    s2_api_key: str = ""

    # Fetch settings
    fetch_window_hours: int = 24
    arxiv_categories: list[str] = [
        "cs.AI", "cs.CL", "cs.LG", "cs.MA", "cs.CR", "cs.SE",
    ]
    arxiv_max_results: int = 200

    # Scoring weights
    weight_embedding: float = 0.45
    weight_keyword: float = 0.15
    weight_citation: float = 0.20
    weight_community: float = 0.10
    weight_recency: float = 0.10
    candidate_threshold: float = 0.35

    # Keyword match sub-weights
    keyword_high_weight: float = 1.0
    keyword_medium_weight: float = 0.6
    keyword_low_weight: float = 0.3
    keyword_title_boost: float = 1.5

    # Scheduling (cron-based: run once daily at this hour/minute in this timezone)
    schedule_cron_hour: int = 5
    schedule_cron_minute: int = 0
    schedule_timezone: str = "America/New_York"

    # Model cache
    model_cache_dir: str = "/data/models"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Paper TTL
    paper_ttl_days: int = 30


settings = Settings()
