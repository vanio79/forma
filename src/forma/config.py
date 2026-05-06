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

    # Upstream API configuration (for chat/completions)
    upstream_base_url: str = "https://api.openai.com/v1"
    upstream_api_key: str = ""
    upstream_timeout: float = 300.0  # 5 minutes for long requests

    # Extraction LLM configuration (for entity/relationship/fact extraction)
    # Used internally by Forma's RAG pipeline, not exposed as endpoint
    extractor_base_url: str = ""  # Empty = use upstream_base_url for extraction
    extractor_api_key: str = ""  # Empty = no auth for extraction endpoint
    extractor_model_name: str = ""  # Model for extraction tasks
    extractor_timeout: float = 120.0  # 2 minutes for extraction (may need more time)
    extractor_send_reasoning_params: bool = (
        False  # Send reasoning_effort/enable_thinking params (not supported by all APIs)
    )

    # GrafitoDB configuration (SQLite-backed graph + vector database)
    # Single file database - minimal file descriptors
    grafitodb_path: str = "./grafito_data/forma.db"  # SQLite database file path
    grafitodb_embedding_model: str = "all-MiniLM-L6-v2"  # SentenceTransformer model for embeddings
    grafitodb_vector_dim: int = 384  # Embedding dimension (depends on model)
    grafitodb_model_cache_path: str = "./models"  # Local cache for embedding model

    # Request tracker configuration (for web UI)
    tracker_db_path: str = "./tracker_data/forma_tracker.db"  # SQLite tracker database
    tracker_max_records: int = 100  # Maximum request records to keep
    tracker_enabled: bool = True  # Enable request tracking for web UI

    # Model mapping (optional: map local model names to upstream models)
    # Format: "local_name:upstream_name,local_name2:upstream_name2"
    model_mapping: str = ""

    def get_model_mapping(self) -> dict[str, str]:
        """Parse model mapping string into dictionary."""
        if not self.model_mapping:
            return {}
        mapping = {}
        for pair in self.model_mapping.split(","):
            if ":" in pair:
                local, upstream = pair.strip().split(":", 1)
                mapping[local.strip()] = upstream.strip()
        return mapping


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
