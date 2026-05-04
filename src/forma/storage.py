"""ChromaDB storage for extracted facts and recipes."""

import logging
from datetime import datetime
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)

# Collection names
FACTS_COLLECTION = "facts_v1"
RECIPES_COLLECTION = "recipes_v1"


class Storage:
    """Manages ChromaDB storage for extracted facts and recipes."""

    def __init__(
        self, host: str = "localhost", port: int = 8000, persist_directory: str = ""
    ) -> None:
        """Initialize ChromaDB client and collections."""
        self.client = self._create_client(host, port, persist_directory)
        self.facts_collection = self._get_or_create_collection(FACTS_COLLECTION)
        self.recipes_collection = self._get_or_create_collection(RECIPES_COLLECTION)
        logger.info(
            f"Storage initialized - facts: {FACTS_COLLECTION}, recipes: {RECIPES_COLLECTION}"
        )

    def _create_client(self, host: str, port: int, persist_directory: str) -> chromadb.ClientAPI:
        """Create ChromaDB client."""
        if persist_directory:
            # Persistent storage
            logger.info(f"Using persistent ChromaDB at: {persist_directory}")
            return chromadb.PersistentClient(path=persist_directory)
        else:
            # Ephemeral (in-memory) - requires running ChromaDB server
            logger.info(f"Connecting to ChromaDB server at {host}:{port}")
            return chromadb.HttpClient(host=host, port=port)

    def _get_or_create_collection(self, name: str) -> chromadb.Collection:
        """Get or create a collection with cosine similarity."""
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def store_facts(self, facts: list[dict[str, Any]]) -> int:
        """
        Store facts in ChromaDB.

        Each fact is stored with:
        - document: the statement text
        - metadata: confidence, timestamp, source_type
        - id: auto-generated UUID

        Returns number of facts stored.
        """
        if not facts:
            return 0

        timestamp = datetime.utcnow().isoformat()
        documents = []
        metadatas = []
        ids = []

        for i, fact in enumerate(facts):
            statement = fact.get("statement", "")
            confidence = fact.get("confidence", 0.9)

            if not statement.strip():
                continue

            # Generate unique ID using timestamp and index
            fact_id = f"fact_{timestamp}_{i}"

            documents.append(statement)
            metadatas.append(
                {
                    "confidence": confidence,
                    "timestamp": timestamp,
                    "source_type": "fact",
                }
            )
            ids.append(fact_id)

        if documents:
            self.facts_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )
            logger.info(f"Stored {len(documents)} facts to {FACTS_COLLECTION}")

        return len(documents)

    def store_recipes(self, recipes: list[dict[str, Any]]) -> int:
        """
        Store recipes in ChromaDB.

        Each recipe is stored with:
        - document: the description text
        - metadata: confidence, timestamp, source_type
        - id: auto-generated UUID

        Returns number of recipes stored.
        """
        if not recipes:
            return 0

        timestamp = datetime.utcnow().isoformat()
        documents = []
        metadatas = []
        ids = []

        for i, recipe in enumerate(recipes):
            description = recipe.get("description", "")
            confidence = recipe.get("confidence", 0.9)

            if not description.strip():
                continue

            # Generate unique ID using timestamp and index
            recipe_id = f"recipe_{timestamp}_{i}"

            documents.append(description)
            metadatas.append(
                {
                    "confidence": confidence,
                    "timestamp": timestamp,
                    "source_type": "recipe",
                }
            )
            ids.append(recipe_id)

        if documents:
            self.recipes_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )
            logger.info(f"Stored {len(documents)} recipes to {RECIPES_COLLECTION}")

        return len(documents)

    def store_extraction(
        self, facts: list[dict[str, Any]], recipes: list[dict[str, Any]]
    ) -> tuple[int, int]:
        """
        Store both facts and recipes from an extraction.

        Returns tuple of (facts_count, recipes_count).
        """
        facts_count = self.store_facts(facts)
        recipes_count = self.store_recipes(recipes)
        return (facts_count, recipes_count)

    def query_facts(self, query: str, n_results: int = 10) -> list[dict[str, Any]]:
        """
        Query facts by semantic similarity.

        Returns list of results with document, metadata, and distance.
        """
        results = self.facts_collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        # Format results
        formatted = []
        if results.get("documents") and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                formatted.append(
                    {
                        "document": doc,
                        "metadata": results.get("metadatas", [[]])[0][i]
                        if results.get("metadatas")
                        else {},
                        "distance": results.get("distances", [[]])[0][i]
                        if results.get("distances")
                        else None,
                        "id": results.get("ids", [[]])[0][i] if results.get("ids") else None,
                    }
                )

        return formatted

    def query_recipes(self, query: str, n_results: int = 10) -> list[dict[str, Any]]:
        """
        Query recipes by semantic similarity.

        Returns list of results with document, metadata, and distance.
        """
        results = self.recipes_collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        # Format results
        formatted = []
        if results.get("documents") and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                formatted.append(
                    {
                        "document": doc,
                        "metadata": results.get("metadatas", [[]])[0][i]
                        if results.get("metadatas")
                        else {},
                        "distance": results.get("distances", [[]])[0][i]
                        if results.get("distances")
                        else None,
                        "id": results.get("ids", [[]])[0][i] if results.get("ids") else None,
                    }
                )

        return formatted

    def get_collection_stats(self) -> dict[str, int]:
        """Get count of documents in each collection."""
        return {
            "facts": self.facts_collection.count(),
            "recipes": self.recipes_collection.count(),
        }
