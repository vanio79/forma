"""Configuration management for Forma."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Extraction LLM configuration (for entity/relationship/fact extraction)
    # Used internally by Forma's RAG pipeline, not exposed as endpoint
    extractor_base_url: str = ""  # Empty = use upstream from database
    extractor_api_key: str = ""  # Empty = no auth for extraction endpoint
    extractor_model_name: str = ""  # Model for extraction tasks (required)
    extractor_timeout: float = 120.0  # 2 minutes for extraction (may need more time)
    extractor_send_reasoning_params: bool = (
        False  # Send reasoning_effort/enable_thinking params (not supported by all APIs)
    )

    # GrafitoDB configuration (SQLite-backed graph + vector database)
    grafitodb_path: str = "./grafito_data/forma.db"  # SQLite database file path
    grafitodb_embedding_model: str = "all-MiniLM-L6-v2"  # SentenceTransformer model for embeddings
    grafitodb_vector_dim: int = 384  # Embedding dimension (depends on model)
    grafitodb_model_cache_path: str = "./models"  # Local cache for embedding model

    # Forma database configuration (system data + upstreams + request history)
    forma_db_path: str = "./data/forma.db"  # SQLite database for system data
    history_max_records: int = 100  # Maximum request history records to keep
    history_enabled: bool = True  # Enable request history for web UI


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
