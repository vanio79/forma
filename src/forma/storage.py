"""Storage for extracted data: ChromaDB for facts/recipes, CogDB for entities/relationships."""

import logging
import math
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, cast

import chromadb
from cog import config as cog_config
from cog.torque import Graph

logger = logging.getLogger(__name__)

# ChromaDB collection names
FACTS_COLLECTION = "facts_v1"
RECIPES_COLLECTION = "recipes_v1"

# Token estimation: roughly 4 characters per token
CHARS_PER_TOKEN = 4


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

    def _calculate_chroma_score(
        self, confidence: float, distance: float, timestamp: str, decay_days: float = 30.0
    ) -> float:
        """
        Calculate composite score for ChromaDB results (facts, recipes).

        Score = confidence * similarity * time_factor

        - similarity = 1 - distance (cosine distance, lower = more similar)
        - time_factor = exponential decay based on age (newer = higher)
        """
        try:
            similarity = 1.0 - distance
            # Calculate age in hours
            if timestamp:
                ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                now = datetime.now(UTC)
                age_hours = (now - ts).total_seconds() / 3600
            else:
                age_hours = 0
            # Exponential decay: half-life = decay_days
            time_factor = math.exp(-age_hours / (decay_days * 24))
            return confidence * similarity * time_factor
        except Exception:
            return confidence * (1.0 - distance)

    def _calculate_cog_score(
        self, confidence: float, timestamp: str, decay_days: float = 30.0
    ) -> float:
        """
        Calculate composite score for CogDB results (relationships).

        Score = confidence * time_factor

        - time_factor = exponential decay based on age (newer = higher)
        """
        try:
            # Calculate age in hours
            if timestamp:
                ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                now = datetime.now(UTC)
                age_hours = (now - ts).total_seconds() / 3600
            else:
                age_hours = 0
            # Exponential decay: half-life = decay_days
            time_factor = math.exp(-age_hours / (decay_days * 24))
            return confidence * time_factor
        except Exception:
            return confidence

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text length."""
        return len(text) // CHARS_PER_TOKEN

    def _format_relationship_for_context(self, rel: dict[str, Any]) -> str:
        """Format a relationship for context string."""
        return f"- {rel['subject']} → {rel['predicate']} → {rel['object']}"

    def _format_fact_for_context(self, fact: dict[str, Any]) -> str:
        """Format a fact for context string."""
        return f"- {fact['statement']}"

    def _format_recipe_for_context(self, recipe: dict[str, Any]) -> str:
        """Format a recipe for context string, truncating if too long."""
        desc = recipe["description"]
        if len(desc) > 200:
            desc = desc[:200] + "..."
        return f"- {desc}"

    def _create_chroma_client(self, host: str, port: int, persist_directory: str) -> Any:
        """Create ChromaDB client."""
        if persist_directory:
            logger.info(f"Using persistent ChromaDB at: {persist_directory}")
            return chromadb.PersistentClient(path=persist_directory)
        # Try server mode first, fall back to in-memory ephemeral
        try:
            logger.info(f"Connecting to ChromaDB server at {host}:{port}")
            return chromadb.HttpClient(host=host, port=port)
        except Exception:
            logger.warning(
                f"Could not connect to ChromaDB server at {host}:{port}, "
                "falling back to in-memory ephemeral client"
            )
            return chromadb.EphemeralClient()

    def _get_or_create_collection(self, name: str) -> Any:
        """Get or create a ChromaDB collection with cosine similarity."""
        return cast(
            Any,
            self.chroma_client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            ),
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
        - Plus confidence and timestamp as properties

        Deduplicates by checking if relationship already exists.
        Updates timestamp and keeps higher confidence if duplicate.

        Returns number of relationships stored/updated.
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

            rel_key = f"{subject}_{predicate}_{obj}"

            # Check if relationship already exists
            existing_result = self.graph.v(subject).out(predicate).all()
            existing_obj = None
            if existing_result.get("result"):
                existing_obj = existing_result["result"][0].get("id")

            if existing_obj == obj:
                # Relationship exists - update timestamp and keep higher confidence
                conf_result = self.graph.v(rel_key).out("confidence").all()
                existing_conf = (
                    float(conf_result.get("result", [{}])[0].get("id", "0.9"))
                    if conf_result.get("result")
                    else 0.9
                )
                # Keep higher confidence
                final_conf = max(existing_conf, confidence)
                self.graph.put(rel_key, "confidence", str(final_conf))
                self.graph.put(rel_key, "extracted_at", timestamp)
            else:
                # New relationship - store it
                self.graph.put(subject, predicate, obj)
                self.graph.put(rel_key, "confidence", str(confidence))
                self.graph.put(rel_key, "extracted_at", timestamp)
                count += 1

        if count > 0:
            logger.info(f"Stored {count} new relationships to CogDB")

        return count

    def store_facts(self, facts: list[dict[str, Any]]) -> int:
        """
        Store facts in ChromaDB.

        Each fact is stored with:
        - document: the statement text
        - metadata: confidence, timestamp, source_type
        - id: auto-generated

        Deduplicates by checking for similar existing facts (distance < 0.05).
        Returns number of facts stored.
        """
        if not facts:
            return 0

        timestamp = datetime.utcnow().isoformat()
        documents = []
        metadatas = []
        ids = []
        duplicate_threshold = 0.05  # Very similar = duplicate

        for i, fact in enumerate(facts):
            statement = fact.get("statement", "")
            confidence = fact.get("confidence", 0.9)

            # Skip empty statements and N/A values
            if not statement.strip() or statement.strip().upper() == "N/A":
                continue

            # Check for near-duplicate existing facts
            try:
                existing = self.facts_collection.query(
                    query_texts=[statement],
                    n_results=1,
                )
                distances = existing.get("distances")
                if distances and distances[0]:
                    min_distance = distances[0][0]
                    if min_distance < duplicate_threshold:
                        # Near-duplicate found - skip
                        logger.debug(f"Skipping duplicate fact: {statement[:50]}...")
                        continue
            except Exception:
                pass  # If query fails, proceed to add

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
                metadatas=cast(list[Mapping[str, Any]], metadatas),
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

        Deduplicates by checking for similar existing recipes (distance < 0.05).
        Returns number of recipes stored.
        """
        if not recipes:
            return 0

        timestamp = datetime.utcnow().isoformat()
        documents = []
        metadatas = []
        ids = []
        duplicate_threshold = 0.05  # Very similar = duplicate

        for i, recipe in enumerate(recipes):
            description = recipe.get("description", "")
            confidence = recipe.get("confidence", 0.9)

            # Skip empty descriptions and N/A values
            if not description.strip() or description.strip().upper() == "N/A":
                continue

            # Check for near-duplicate existing recipes
            try:
                existing = self.recipes_collection.query(
                    query_texts=[description],
                    n_results=1,
                )
                distances = existing.get("distances")
                if distances and distances[0]:
                    min_distance = distances[0][0]
                    if min_distance < duplicate_threshold:
                        # Near-duplicate found - skip
                        logger.debug(f"Skipping duplicate recipe: {description[:50]}...")
                        continue
            except Exception:
                pass  # If query fails, proceed to add

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
                metadatas=cast(list[Mapping[str, Any]], metadatas),
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

        formatted: list[dict[str, Any]] = []
        documents = results.get("documents")
        metadatas = results.get("metadatas")
        distances = results.get("distances")
        ids = results.get("ids")
        if documents and documents[0]:
            for i, doc in enumerate(documents[0]):
                formatted.append(
                    {
                        "document": doc,
                        "metadata": metadatas[0][i] if metadatas else {},
                        "distance": distances[0][i] if distances else None,
                        "id": ids[0][i] if ids else None,
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

        formatted: list[dict[str, Any]] = []
        documents = results.get("documents")
        metadatas = results.get("metadatas")
        distances = results.get("distances")
        ids = results.get("ids")
        if documents and documents[0]:
            for i, doc in enumerate(documents[0]):
                formatted.append(
                    {
                        "document": doc,
                        "metadata": metadatas[0][i] if metadatas else {},
                        "distance": distances[0][i] if distances else None,
                        "id": ids[0][i] if ids else None,
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

        If subject is provided, returns both:
        - Outgoing relationships (entity as subject)
        - Incoming relationships (entity as object)
        Otherwise returns all relationships.

        Returns list of relationships.
        """
        relationships = []
        try:
            if subject:
                # Get outgoing edges from subject (entity -> predicate -> object)
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

                # Get incoming edges by scanning all entities
                # Find relationships where entity appears as object
                scan_result = self.graph.scan(n_results, "v")
                vertices = scan_result.get("result", [])
                for vertex in vertices:
                    v_name = vertex.get("id")
                    # Skip metadata vertices
                    if (
                        v_name.startswith("202")  # timestamps
                        or v_name
                        in ["0.9", "1.0", "person", "organization", "other", "concept", "object"]
                        or "_" in v_name
                        and "->" not in v_name
                    ):
                        continue
                    # Check if this vertex has outgoing relationships to our subject
                    try:
                        out_result = self.graph.v(v_name).out().all("e")
                        out_edges = out_result.get("result", [])
                        for edge in out_edges:
                            edge_labels = edge.get("edges", [])
                            for pred in edge_labels:
                                if pred in ["type", "confidence", "extracted_at"]:
                                    continue
                                # Check if this edge points to our subject
                                obj_result = self.graph.v(v_name).out(pred).all()
                                if obj_result.get("result"):
                                    for obj_item in obj_result["result"]:
                                        obj = obj_item.get("id")
                                        if obj == subject:
                                            relationships.append(
                                                {
                                                    "subject": v_name,
                                                    "predicate": pred,
                                                    "object": subject,
                                                }
                                            )
                    except Exception:
                        continue
            else:
                # Scan all vertices and find relationships
                scan_result = self.graph.scan(n_results, "v")
                vertices = scan_result.get("result", [])
                for vertex in vertices:
                    v_name = vertex.get("id")
                    # Skip metadata vertices
                    if (
                        v_name.startswith("202")
                        or v_name
                        in ["0.9", "1.0", "person", "organization", "other", "concept", "object"]
                        or "_" in v_name
                        and "->" not in v_name
                    ):
                        continue
                    type_result = self.graph.v(v_name).out("type").all()
                    if not type_result.get("result"):
                        continue
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
            scan_result = self.graph.scan(5000, "v")
            entity_count = len(scan_result.get("result", []))
        except Exception:
            entity_count = 0

        return {
            "chromadb": chroma_stats,
            "cogdb": {"entities": entity_count},
        }

    def close(self) -> None:
        """Close all storage connections and release file handles."""
        # Close ChromaDB client
        try:
            if hasattr(self.chroma_client, "close"):
                self.chroma_client.close()
                logger.info("ChromaDB client closed")
        except Exception as e:
            logger.error(f"Failed to close ChromaDB: {e}")

        # Close CogDB graph
        try:
            if hasattr(self.graph, "close"):
                self.graph.close()
                logger.info("CogDB graph closed")
        except Exception as e:
            logger.error(f"Failed to close CogDB: {e}")

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

    def retrieve_context(
        self,
        entities_queries: list[str],
        fact_query: str | None,
        recipe_query: str | None,
        token_budget: int = 1500,
        min_confidence: float = 0.5,
        max_distance: float = 0.7,
        query_limit: int = 100,
        decay_days: float = 30.0,
    ) -> dict[str, Any]:
        """
        Retrieve context from storage based on queries with composite scoring and token budget.

        Composite scoring:
        - ChromaDB (facts, recipes): confidence * similarity * time_factor
        - CogDB (relationships): confidence * time_factor

        Token budget: stops adding items when estimated token count reaches budget.

        Returns dict with 'relationships', 'facts', 'recipes', 'tokens_used', 'scores' lists.
        """
        all_items: list[dict[str, Any]] = []  # Pool of all scored items

        # Query CogDB for entity relationships
        for entity in entities_queries:
            try:
                entity_rels = self.query_relationships(subject=entity, n_results=query_limit)
                for rel in entity_rels:
                    rel_key = f"{rel['subject']}_{rel['predicate']}_{rel['object']}"
                    ts_result = self.graph.v(rel_key).out("extracted_at").all()
                    timestamp = (
                        ts_result.get("result", [{}])[0].get("id", "")
                        if ts_result.get("result")
                        else ""
                    )
                    conf_result = self.graph.v(rel_key).out("confidence").all()
                    confidence = (
                        float(conf_result.get("result", [{}])[0].get("id", "0.9"))
                        if conf_result.get("result")
                        else 0.9
                    )

                    if confidence >= min_confidence:
                        score = self._calculate_cog_score(confidence, timestamp, decay_days)
                        formatted = self._format_relationship_for_context(rel)
                        all_items.append(
                            {
                                "type": "relationship",
                                "data": {
                                    "subject": rel["subject"],
                                    "predicate": rel["predicate"],
                                    "object": rel["object"],
                                    "confidence": confidence,
                                    "timestamp": timestamp,
                                    "source": entity,
                                },
                                "score": score,
                                "formatted": formatted,
                                "tokens": self._estimate_tokens(formatted),
                            }
                        )
            except Exception as e:
                logger.debug(f"Query entity relationships error for {entity}: {e}")

        # Query ChromaDB for facts
        if fact_query:
            try:
                fact_results = self.query_facts(fact_query, n_results=query_limit)
                for fact in fact_results:
                    confidence = fact.get("metadata", {}).get("confidence", 0.9)
                    distance = fact.get("distance", 1.0) or 1.0
                    timestamp = fact.get("metadata", {}).get("timestamp", "")

                    if confidence >= min_confidence and distance <= max_distance:
                        score = self._calculate_chroma_score(
                            confidence, distance, timestamp, decay_days
                        )
                        formatted = self._format_fact_for_context({"statement": fact["document"]})
                        all_items.append(
                            {
                                "type": "fact",
                                "data": {
                                    "statement": fact["document"],
                                    "confidence": confidence,
                                    "timestamp": timestamp,
                                    "distance": distance,
                                },
                                "score": score,
                                "formatted": formatted,
                                "tokens": self._estimate_tokens(formatted),
                            }
                        )
            except Exception as e:
                logger.debug(f"Query facts error: {e}")

        # Query ChromaDB for recipes
        if recipe_query:
            try:
                recipe_results = self.query_recipes(recipe_query, n_results=query_limit)
                for recipe in recipe_results:
                    confidence = recipe.get("metadata", {}).get("confidence", 0.9)
                    distance = recipe.get("distance", 1.0) or 1.0
                    timestamp = recipe.get("metadata", {}).get("timestamp", "")

                    if confidence >= min_confidence and distance <= max_distance:
                        score = self._calculate_chroma_score(
                            confidence, distance, timestamp, decay_days
                        )
                        formatted = self._format_recipe_for_context(
                            {"description": recipe["document"]}
                        )
                        all_items.append(
                            {
                                "type": "recipe",
                                "data": {
                                    "description": recipe["document"],
                                    "confidence": confidence,
                                    "timestamp": timestamp,
                                    "distance": distance,
                                },
                                "score": score,
                                "formatted": formatted,
                                "tokens": self._estimate_tokens(formatted),
                            }
                        )
            except Exception as e:
                logger.debug(f"Query recipes error: {e}")

        # Deduplicate items (keep highest score for duplicates)
        seen_keys: dict[str, int] = {}  # key -> index in all_items
        for i, item in enumerate(all_items):
            # Create unique key for each type
            if item["type"] == "relationship":
                key = (
                    f"rel:{item['data']['subject']}|"
                    f"{item['data']['predicate']}|"
                    f"{item['data']['object']}"
                )
            elif item["type"] == "fact":
                key = f"fact:{item['data']['statement']}"
            elif item["type"] == "recipe":
                key = f"recipe:{item['data']['description'][:100]}"  # First 100 chars for key
            else:
                key = f"{item['type']}:{i}"

            if key in seen_keys:
                # Keep the one with higher score
                existing_idx = seen_keys[key]
                if item["score"] > all_items[existing_idx]["score"]:
                    all_items[existing_idx] = item
            else:
                seen_keys[key] = i

        # Remove duplicates by keeping only items that are in seen_keys
        deduped_items = [all_items[i] for i in sorted(seen_keys.values())]

        # Sort deduplicated items by composite score DESC
        deduped_items.sort(key=lambda x: x["score"], reverse=True)

        # Build context within token budget
        relationships = []
        facts = []
        recipes = []
        scores: dict[str, list[float]] = {"relationships": [], "facts": [], "recipes": []}
        tokens_used = 0

        # Reserve tokens for headers
        header_tokens = self._estimate_tokens(
            "Relevant context from memory:\n"
            "Known relationships:\n"
            "Known facts:\n"
            "Known procedures:\n\n"
        )
        tokens_used += header_tokens

        for item in deduped_items:
            if tokens_used + item["tokens"] > token_budget:
                break

            tokens_used += item["tokens"]
            if item["type"] == "relationship":
                relationships.append(item["data"])
                scores["relationships"].append(item["score"])
            elif item["type"] == "fact":
                facts.append(item["data"])
                scores["facts"].append(item["score"])
            elif item["type"] == "recipe":
                recipes.append(item["data"])
                scores["recipes"].append(item["score"])

        logger.info(
            f"Retrieved {len(relationships)} relationships, "
            f"{len(facts)} facts, {len(recipes)} recipes "
            f"using {tokens_used}/{token_budget} tokens"
        )

        return {
            "relationships": relationships,
            "facts": facts,
            "recipes": recipes,
            "tokens_used": tokens_used,
            "scores": scores,
        }

    def format_context_for_prompt(self, context: dict[str, Any]) -> str:
        """
        Format retrieved context for prompt augmentation.

        Returns a formatted string suitable for prepending to user message.
        """
        lines = []

        if context.get("relationships"):
            lines.append("Known relationships:")
            for rel in context["relationships"]:
                lines.append(f"- {rel['subject']} → {rel['predicate']} → {rel['object']}")

        if context.get("facts"):
            lines.append("Known facts:")
            for fact in context["facts"]:
                lines.append(f"- {fact['statement']}")

        if context.get("recipes"):
            lines.append("Known procedures:")
            for recipe in context["recipes"]:
                desc = recipe["description"]
                if len(desc) > 200:
                    desc = desc[:200] + "..."
                lines.append(f"- {desc}")

        if lines:
            return "Relevant context from memory:\n" + "\n".join(lines) + "\n\n"
        return ""
