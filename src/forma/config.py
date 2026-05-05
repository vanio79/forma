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

    # Embedding API configuration (separate endpoint)
    # If set, embeddings requests go here instead of upstream_base_url
    embedding_base_url: str = ""  # Empty = use upstream_base_url for embeddings
    embedding_api_key: str = ""  # Empty = no auth for embedding endpoint
    embedding_model_name: str = ""  # Default model for embeddings if not specified
    embedding_timeout: float = 60.0  # 1 minute for embedding requests

    # Extraction LLM configuration (for entity/relationship/fact extraction)
    # Used internally by Forma's RAG pipeline, not exposed as endpoint
    extractor_base_url: str = ""  # Empty = use upstream_base_url for extraction
    extractor_api_key: str = ""  # Empty = no auth for extraction endpoint
    extractor_model_name: str = ""  # Model for extraction tasks
    extractor_timeout: float = 120.0  # 2 minutes for extraction (may need more time)

    # ChromaDB configuration (for storing extracted facts and recipes)
    # Note: port is only used when persist_directory is empty (HttpClient mode)
    # Default port 8001 to avoid conflict with Forma server port 8000
    chromadb_host: str = "localhost"
    chromadb_port: int = 8001
    chromadb_persist_directory: str = (
        ""  # Empty = use in-memory (ephemeral), or path for persistent
    )

    # CogDB configuration (for storing entities and relationships as graph)
    cogdb_home: str = "forma_graph"  # Graph database name
    cogdb_path_prefix: str = "./cog_data"  # Storage directory for CogDB

    # File descriptor limits to prevent exhaustion under load
    chromadb_max_file_handles: int = 256  # Max open files for ChromaDB
    cogdb_index_capacity: int = 50000  # Index hash table size (lower = fewer indices)
    cogdb_l2_cache_size: int = 50000  # L2 cache entries (lower = fewer cached handles)

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
