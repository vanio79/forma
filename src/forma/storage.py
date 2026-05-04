"""Storage for extracted data: ChromaDB for facts/recipes, CogDB for entities/relationships."""

import logging
import os
from datetime import datetime
from typing import Any

import chromadb
from cog import config as cog_config
from cog.torque import Graph

logger = logging.getLogger(__name__)

# ChromaDB collection names
FACTS_COLLECTION = "facts_v1"
RECIPES_COLLECTION = "recipes_v1"


class Storage:
    """Manages ChromaDB for facts/recipes and CogDB for entities/relationships."""

    def __init__(
        self,
        chromadb_host: str = "localhost",
        chromadb_port: int = 8000,
        chromadb_persist_directory: str = "",
        cogdb_home: str = "forma_graph",
        cogdb_path_prefix: str = "./cog_data",
    ) -> None:
        """Initialize ChromaDB and CogDB."""
        # Initialize ChromaDB for facts and recipes
        self.chroma_client = self._create_chroma_client(
            chromadb_host, chromadb_port, chromadb_persist_directory
        )
        self.facts_collection = self._get_or_create_collection(FACTS_COLLECTION)
        self.recipes_collection = self._get_or_create_collection(RECIPES_COLLECTION)
        logger.info(
            f"ChromaDB initialized - facts: {FACTS_COLLECTION}, recipes: {RECIPES_COLLECTION}"
        )

        # Initialize CogDB for entities and relationships
        self.graph = self._create_cogdb_graph(cogdb_home, cogdb_path_prefix)
        logger.info(f"CogDB initialized - graph: {cogdb_home}")

    def _create_chroma_client(
        self, host: str, port: int, persist_directory: str
    ) -> chromadb.ClientAPI:
        """Create ChromaDB client."""
        if persist_directory:
            logger.info(f"Using persistent ChromaDB at: {persist_directory}")
            return chromadb.PersistentClient(path=persist_directory)
        else:
            logger.info(f"Connecting to ChromaDB server at {host}:{port}")
            return chromadb.HttpClient(host=host, port=port)

    def _get_or_create_collection(self, name: str) -> chromadb.Collection:
        """Get or create a ChromaDB collection with cosine similarity."""
        return self.chroma_client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def _create_cogdb_graph(self, graph_name: str, path_prefix: str) -> Graph:
        """Create CogDB graph for entities and relationships."""
        # Ensure storage directory exists
        os.makedirs(path_prefix, exist_ok=True)

        # Configure CogDB storage location
        cog_config.COG_PATH_PREFIX = path_prefix
        cog_config.COG_HOME = graph_name

        return Graph(graph_name)

    def store_entities(self, entities: list[dict[str, Any]]) -> int:
        """
        Store entities in CogDB as nodes with type property.

        Each entity is stored as:
        - entity_name -> type -> entity_type (triple)
        - entity_name -> confidence -> confidence_value (property)

        Returns number of entities stored.
        """
        if not entities:
            return 0

        timestamp = datetime.utcnow().isoformat()
        count = 0

        for entity in entities:
            name = entity.get("name", "")
            entity_type = entity.get("type", "other")
            confidence = entity.get("confidence", 0.9)

            if not name.strip():
                continue

            # Store entity with type
            self.graph.put(name, "type", entity_type)
            # Store confidence as property
            self.graph.put(name, "confidence", str(confidence))
            # Store timestamp
            self.graph.put(name, "extracted_at", timestamp)

            count += 1

        if count > 0:
            logger.info(f"Stored {count} entities to CogDB")

        return count

    def store_relationships(self, relationships: list[dict[str, Any]]) -> int:
        """
        Store relationships in CogDB as triples.

        Each relationship is stored as:
        - subject -> predicate -> object (triple)
        - Plus confidence and timestamp as properties on subject

        Returns number of relationships stored.
        """
        if not relationships:
            return 0

        timestamp = datetime.utcnow().isoformat()
        count = 0

        for rel in relationships:
            subject = rel.get("subject", "")
            predicate = rel.get("predicate", "")
            obj = rel.get("object", "")
            confidence = rel.get("confidence", 0.9)

            if not subject.strip() or not predicate.strip() or not obj.strip():
                continue

            # Store relationship triple
            self.graph.put(subject, predicate, obj)
            # Store confidence on the relationship edge (via subject)
            self.graph.put(f"{subject}_{predicate}_{obj}", "confidence", str(confidence))
            # Store timestamp
            self.graph.put(f"{subject}_{predicate}_{obj}", "extracted_at", timestamp)

            count += 1

        if count > 0:
            logger.info(f"Stored {count} relationships to CogDB")

        return count

    def store_facts(self, facts: list[dict[str, Any]]) -> int:
        """
        Store facts in ChromaDB.

        Each fact is stored with:
        - document: the statement text
        - metadata: confidence, timestamp, source_type
        - id: auto-generated

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
            logger.info(f"Stored {len(documents)} facts to ChromaDB")

        return len(documents)

    def store_recipes(self, recipes: list[dict[str, Any]]) -> int:
        """
        Store recipes in ChromaDB.

        Each recipe is stored with:
        - document: the description text
        - metadata: confidence, timestamp, source_type
        - id: auto-generated

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
            logger.info(f"Stored {len(documents)} recipes to ChromaDB")

        return len(documents)

    def store_extraction(
        self,
        entities: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
        facts: list[dict[str, Any]],
        recipes: list[dict[str, Any]],
    ) -> tuple[int, int, int, int]:
        """
        Store all extracted data.

        Returns tuple of (entities_count, relationships_count, facts_count, recipes_count).
        """
        entities_count = self.store_entities(entities)
        relationships_count = self.store_relationships(relationships)
        facts_count = self.store_facts(facts)
        recipes_count = self.store_recipes(recipes)
        return (entities_count, relationships_count, facts_count, recipes_count)

    def query_facts(self, query: str, n_results: int = 10) -> list[dict[str, Any]]:
        """
        Query facts by semantic similarity.

        Returns list of results with document, metadata, and distance.
        """
        results = self.facts_collection.query(
            query_texts=[query],
            n_results=n_results,
        )

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

    def query_entity(self, name: str) -> dict[str, Any] | None:
        """
        Query an entity by name from CogDB.

        Returns entity with type and properties, or None if not found.
        """
        try:
            result = self.graph.v(name).out("type").all()
            if result.get("result") and len(result["result"]) > 0:
                entity_type = result["result"][0].get("id")
                confidence_result = self.graph.v(name).out("confidence").all()
                confidence = (
                    float(confidence_result["result"][0].get("id", "0.9"))
                    if confidence_result.get("result")
                    else 0.9
                )
                timestamp_result = self.graph.v(name).out("extracted_at").all()
                timestamp = (
                    timestamp_result["result"][0].get("id")
                    if timestamp_result.get("result")
                    else None
                )
                return {
                    "name": name,
                    "type": entity_type,
                    "confidence": confidence,
                    "extracted_at": timestamp,
                }
        except Exception:
            pass
        return None

    def query_relationships(
        self, subject: str = "", predicate: str = "", n_results: int = 20
    ) -> list[dict[str, Any]]:
        """
        Query relationships from CogDB.

        If subject is provided, returns outgoing relationships from that subject.
        Otherwise returns all relationships.

        Returns list of relationships.
        """
        relationships = []
        try:
            if subject:
                # Get outgoing edges from subject
                result = self.graph.v(subject).out().all("e")
                edges = result.get("result", [])
                for edge in edges:
                    edge_labels = edge.get("edges", [])
                    for pred in edge_labels:
                        if pred in ["type", "confidence", "extracted_at"]:
                            continue
                        # Get the object for this predicate
                        obj_result = self.graph.v(subject).out(pred).all()
                        if obj_result.get("result"):
                            obj = obj_result["result"][0].get("id")
                            relationships.append(
                                {
                                    "subject": subject,
                                    "predicate": pred,
                                    "object": obj,
                                }
                            )
            else:
                # Scan all vertices and find relationships
                scan_result = self.graph.scan(n_results, "v")
                vertices = scan_result.get("result", [])
                for vertex in vertices:
                    v_name = vertex.get("id")
                    # Skip metadata vertices:
                    # - timestamps (start with year)
                    # - confidence values (numeric strings)
                    # - type values
                    # - relationship tracking vertices (contain underscores with relationship info)
                    if (
                        v_name.startswith("202")  # timestamps
                        or v_name
                        in ["0.9", "1.0", "person", "organization", "other", "concept", "object"]
                        or "_" in v_name
                        and "->"
                        not in v_name  # relationship tracking vertices like "Bob_is a_TechCorp"
                    ):
                        continue
                    # Check if this vertex has a type (meaning it's an entity)
                    type_result = self.graph.v(v_name).out("type").all()
                    if not type_result.get("result"):
                        continue
                    # Query relationships from this entity
                    rels = self.query_relationships(subject=v_name, n_results=5)
                    relationships.extend(rels)
        except Exception as e:
            logger.debug(f"Query relationships error: {e}")

        return relationships

    def get_stats(self) -> dict[str, Any]:
        """Get stats from both storage systems."""
        # ChromaDB stats
        chroma_stats = {
            "facts": self.facts_collection.count(),
            "recipes": self.recipes_collection.count(),
        }

        # CogDB stats (count unique vertices)
        try:
            scan_result = self.graph.scan(1000, "v")
            entity_count = len(scan_result.get("result", []))
        except Exception:
            entity_count = 0

        return {
            "chromadb": chroma_stats,
            "cogdb": {"entities": entity_count},
        }

    def clear_all(self) -> dict[str, Any]:
        """
        Clear all data from ChromaDB and CogDB.

        Returns stats showing what was cleared.
        """
        stats_before = self.get_stats()

        # Clear ChromaDB collections
        try:
            # Delete all documents from facts collection
            facts_ids = self.facts_collection.get()["ids"]
            if facts_ids:
                self.facts_collection.delete(ids=facts_ids)

            # Delete all documents from recipes collection
            recipes_ids = self.recipes_collection.get()["ids"]
            if recipes_ids:
                self.recipes_collection.delete(ids=recipes_ids)

            logger.info("ChromaDB collections cleared")
        except Exception as e:
            logger.error(f"Failed to clear ChromaDB: {e}")

        # Clear CogDB graph
        try:
            # Use truncate to clear the entire graph
            self.graph.truncate()
            logger.info("CogDB graph cleared")
        except Exception as e:
            logger.error(f"Failed to clear CogDB: {e}")

        stats_after = self.get_stats()

        return {
            "before": stats_before,
            "after": stats_after,
            "cleared": {
                "facts": stats_before["chromadb"]["facts"] - stats_after["chromadb"]["facts"],
                "recipes": stats_before["chromadb"]["recipes"] - stats_after["chromadb"]["recipes"],
                "entities": stats_before["cogdb"]["entities"] - stats_after["cogdb"]["entities"],
            },
        }
