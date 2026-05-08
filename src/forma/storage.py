"""Storage for extracted data: GrafitoDB (SQLite-backed graph + vector database)."""

import logging
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from grafito import GrafitoDatabase
from grafito.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def _escape_cypher_string(value: str) -> str:
    """Escape a string for safe use in Cypher queries.

    Prevents injection by escaping single quotes and backslashes.
    """
    return value.replace("\\", "\\\\").replace("'", "\\'")

# Vector index names for semantic search
FACTS_VECTOR_INDEX = "facts_index"
RECIPES_VECTOR_INDEX = "recipes_index"

# Token estimation: roughly 4 characters per token
CHARS_PER_TOKEN = 4


class Storage:
    """Manages GrafitoDB for relationships, facts, and recipes."""

    def __init__(
        self,
        grafitodb_path: str = "./grafito_data/forma.db",
        grafitodb_embedding_model: str = "all-MiniLM-L6-v2",
        grafitodb_vector_dim: int = 384,
        grafitodb_model_cache_path: str = "./models",
    ) -> None:
        """Initialize GrafitoDB with configurable settings."""
        # Ensure storage directory exists
        db_dir = os.path.dirname(grafitodb_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        # Initialize GrafitoDB
        self.db = GrafitoDatabase(db_path=grafitodb_path)
        logger.info(f"GrafitoDB initialized at: {grafitodb_path}")

        # Ensure model is cached locally, then load from cache
        model_path = self._ensure_model_cached(
            grafitodb_embedding_model, grafitodb_model_cache_path
        )

        # Initialize embedding function for vector search
        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name=model_path)
        self.db.register_embedding_function("default", self.embedding_function)

        # Create vector indexes for facts and recipes
        self._create_vector_indexes(grafitodb_vector_dim)

    def _ensure_model_cached(self, model_name: str, cache_path: str) -> str:
        """
        Ensure the embedding model is cached locally.

        Downloads the model to the cache directory if not present,
        then returns the local path for loading.

        Args:
            model_name: Name of the SentenceTransformer model (e.g., 'all-MiniLM-L6-v2')
            cache_path: Root directory for model cache

        Returns:
            Local path to the cached model
        """
        # Build the expected local model path
        local_model_path = os.path.join(cache_path, model_name)

        # Check if model already exists locally
        if os.path.isdir(local_model_path):
            # Check for required model files
            required_files = ["config.json", "pytorch_model.bin"]
            # Also check for model.safetensors as alternative to pytorch_model.bin
            has_model_file = any(
                os.path.isfile(os.path.join(local_model_path, f))
                for f in ["pytorch_model.bin", "model.safetensors"]
            )

            if os.path.isfile(os.path.join(local_model_path, "config.json")) and has_model_file:
                logger.info(f"Loading embedding model from local cache: {local_model_path}")
                return local_model_path
            else:
                logger.info(f"Incomplete model cache found, re-downloading to: {local_model_path}")

        # Model not cached locally, download and save
        logger.info(
            f"Downloading embedding model '{model_name}' to local cache: {local_model_path}"
        )

        # Ensure cache directory exists
        os.makedirs(cache_path, exist_ok=True)

        try:
            # Download the model using SentenceTransformer
            model = SentenceTransformer(model_name)

            # Save to local cache
            model.save(local_model_path)
            logger.info(f"Embedding model saved to local cache: {local_model_path}")

            return local_model_path
        except Exception as e:
            logger.error(f"Failed to download/save model '{model_name}': {e}")
            logger.warning(f"Falling back to online model loading (will download each restart)")
            return model_name

    def _create_vector_indexes(self, dim: int) -> None:
        """Create vector indexes for facts and recipes semantic search.

        Note: store_embeddings=True is required to persist embeddings to the database,
        so they can be loaded on server restart.
        """
        try:
            self.db.create_vector_index(
                name=FACTS_VECTOR_INDEX,
                dim=dim,
                embedding_function=self.embedding_function,
                options={"store_embeddings": True},
                if_not_exists=True,
            )
            logger.info(
                f"Created vector index: {FACTS_VECTOR_INDEX} (dim={dim}, store_embeddings=True)"
            )

            self.db.create_vector_index(
                name=RECIPES_VECTOR_INDEX,
                dim=dim,
                embedding_function=self.embedding_function,
                options={"store_embeddings": True},
                if_not_exists=True,
            )
            logger.info(
                f"Created vector index: {RECIPES_VECTOR_INDEX} (dim={dim}, store_embeddings=True)"
            )
        except Exception as e:
            logger.warning(f"Could not create vector indexes: {e}")

    def _calculate_score(
        self, confidence: float, distance: float | None, timestamp: str, decay_days: float = 30.0
    ) -> float:
        """
        Calculate composite score for results.

        Score = confidence * similarity * time_factor

        - similarity = 1 - distance (if distance provided, else 1.0)
        - time_factor = exponential decay based on age (newer = higher)
        """
        try:
            similarity = 1.0 - (distance if distance is not None else 0.0)
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
            return confidence * (1.0 - (distance if distance is not None else 0.0))

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text length."""
        return len(text) // CHARS_PER_TOKEN

    def _embed_text(self, text: str) -> list[float]:
        """Generate embedding vector for text using the registered embedding function."""
        return self.embedding_function([text])[0]

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

    def store_relationships(self, relationships: list[dict[str, Any]]) -> int:
        """
        Store relationships in GrafitoDB as edges between entity nodes.

        Each relationship is stored as:
        - source_node (Entity with name=subject) -> target_node (Entity with name=object)
        - relationship type = predicate
        - properties: {confidence, extracted_at}

        Returns number of relationships stored.
        """
        if not relationships:
            return 0

        timestamp = datetime.now(UTC).isoformat()
        count = 0

        for rel in relationships:
            subject = rel.get("subject", "")
            predicate = rel.get("predicate", "")
            obj = rel.get("object", "")
            confidence = rel.get("confidence", 0.9)

            if not subject.strip() or not predicate.strip() or not obj.strip():
                continue

            # Skip metadata predicates
            if predicate in ["type", "confidence", "extracted_at"]:
                continue

            try:
                # Find source and target nodes
                source_nodes = self.db.match_nodes(
                    labels=["Entity"],
                    properties={"name": subject},
                    limit=1,
                )
                target_nodes = self.db.match_nodes(
                    labels=["Entity"],
                    properties={"name": obj},
                    limit=1,
                )

                # Create missing entity nodes if needed
                if not source_nodes:
                    source_node = self.db.create_node(
                        labels=["Entity"],
                        properties={
                            "name": subject,
                            "confidence": 0.5,
                            "extracted_at": timestamp,
                        },
                    )
                else:
                    source_node = source_nodes[0]

                if not target_nodes:
                    target_node = self.db.create_node(
                        labels=["Entity"],
                        properties={
                            "name": obj,
                            "confidence": 0.5,
                            "extracted_at": timestamp,
                        },
                    )
                else:
                    target_node = target_nodes[0]

                # Check if relationship already exists
                existing_rels = self.db.match_relationships(
                    source_id=source_node.id,
                    target_id=target_node.id,
                    rel_type=predicate,
                )

                if existing_rels:
                    # Update existing relationship
                    existing_rel = existing_rels[0]
                    self.db.update_relationship_properties(
                        existing_rel.id,
                        {
                            "confidence": max(
                                float(existing_rel.properties.get("confidence", 0.9)), confidence
                            ),
                            "extracted_at": timestamp,
                        },
                    )
                else:
                    # Create new relationship
                    self.db.create_relationship(
                        source_id=source_node.id,
                        target_id=target_node.id,
                        rel_type=predicate,
                        properties={"confidence": confidence, "extracted_at": timestamp},
                    )
                    count += 1
            except Exception as e:
                logger.debug(f"Error storing relationship '{subject}→{predicate}→{obj}': {e}")

        if count > 0:
            logger.info(f"Stored {count} relationships to GrafitoDB")

        return count

    def store_facts(self, facts: list[dict[str, Any]]) -> int:
        """
        Store facts in GrafitoDB as nodes with embeddings.

        Each fact is stored as a node with:
        - labels: ["Fact"]
        - properties: {statement, confidence, timestamp}
        - embedding: for semantic search

        Deduplicates by checking for similar existing facts.
        Returns number of facts stored.
        """
        if not facts:
            return 0

        timestamp = datetime.now(UTC).isoformat()
        documents = []
        duplicate_threshold = 0.05  # Very similar = duplicate

        for fact in facts:
            statement = fact.get("statement", "")
            confidence = fact.get("confidence", 0.9)

            # Skip empty statements and N/A values
            if not statement.strip() or statement.strip().upper() == "N/A":
                continue

            # Check for near-duplicate existing facts using semantic search
            try:
                existing = self.db.semantic_search(
                    vector=statement,  # Will be embedded by registered function
                    k=1,
                    index=FACTS_VECTOR_INDEX,
                    filter_labels=["Fact"],
                )
                if existing:
                    score = existing[0].get("score", 0.0)
                    # score is similarity (higher = more similar), convert to distance
                    distance = 1.0 - score
                    if distance < duplicate_threshold:
                        logger.debug(f"Skipping duplicate fact: {statement[:50]}...")
                        continue
            except Exception:
                pass  # If search fails, proceed to add

            documents.append(
                {
                    "statement": statement,
                    "confidence": confidence,
                    "timestamp": timestamp,
                }
            )

        # Add facts with embeddings
        for doc in documents:
            try:
                node = self.db.create_node(
                    labels=["Fact"],
                    properties={
                        "statement": doc["statement"],
                        "confidence": doc["confidence"],
                        "timestamp": doc["timestamp"],
                    },
                )
                # Compute embedding and upsert for semantic search
                embedding = self._embed_text(doc["statement"])
                self.db.upsert_embedding(
                    node_id=node.id,
                    vector=embedding,
                    index=FACTS_VECTOR_INDEX,
                )
            except Exception as e:
                logger.debug(f"Error storing fact: {e}")

        if documents:
            logger.info(f"Stored {len(documents)} facts to GrafitoDB")

        return len(documents)

    def store_recipes(self, recipes: list[dict[str, Any]]) -> int:
        """
        Store recipes in GrafitoDB as nodes with embeddings.

        Each recipe is stored as a node with:
        - labels: ["Recipe"]
        - properties: {description, confidence, timestamp}
        - embedding: for semantic search

        Deduplicates by checking for similar existing recipes.
        Returns number of recipes stored.
        """
        if not recipes:
            return 0

        timestamp = datetime.now(UTC).isoformat()
        documents = []
        duplicate_threshold = 0.05  # Very similar = duplicate

        for recipe in recipes:
            description = recipe.get("description", "")
            confidence = recipe.get("confidence", 0.9)

            # Skip empty descriptions and N/A values
            if not description.strip() or description.strip().upper() == "N/A":
                continue

            # Check for near-duplicate existing recipes using semantic search
            try:
                existing = self.db.semantic_search(
                    vector=description,  # Will be embedded by registered function
                    k=1,
                    index=RECIPES_VECTOR_INDEX,
                    filter_labels=["Recipe"],
                )
                if existing:
                    score = existing[0].get("score", 0.0)
                    # score is similarity (higher = more similar), convert to distance
                    distance = 1.0 - score
                    if distance < duplicate_threshold:
                        logger.debug(f"Skipping duplicate recipe: {description[:50]}...")
                        continue
            except Exception:
                pass  # If search fails, proceed to add

            documents.append(
                {
                    "description": description,
                    "confidence": confidence,
                    "timestamp": timestamp,
                }
            )

        # Add recipes with embeddings
        for doc in documents:
            try:
                node = self.db.create_node(
                    labels=["Recipe"],
                    properties={
                        "description": doc["description"],
                        "confidence": doc["confidence"],
                        "timestamp": doc["timestamp"],
                    },
                )
                # Compute embedding and upsert for semantic search
                embedding = self._embed_text(doc["description"])
                self.db.upsert_embedding(
                    node_id=node.id,
                    vector=embedding,
                    index=RECIPES_VECTOR_INDEX,
                )
            except Exception as e:
                logger.debug(f"Error storing recipe: {e}")

        if documents:
            logger.info(f"Stored {len(documents)} recipes to GrafitoDB")

        return len(documents)

    def store_extraction(
        self,
        relationships: list[dict[str, Any]],
        facts: list[dict[str, Any]],
        recipes: list[dict[str, Any]],
    ) -> tuple[int, int, int]:
        """
        Store all extracted data.

        Returns tuple of (relationships_count, facts_count, recipes_count).
        """
        relationships_count = self.store_relationships(relationships)
        facts_count = self.store_facts(facts)
        recipes_count = self.store_recipes(recipes)
        return (relationships_count, facts_count, recipes_count)

    def query_facts(self, query: str, n_results: int = 10) -> list[dict[str, Any]]:
        """
        Query facts by semantic similarity.

        Returns list of results with node properties and score.
        """
        try:
            results = self.db.semantic_search(
                vector=query,  # Will be embedded by registered function
                k=n_results,
                index=FACTS_VECTOR_INDEX,
                filter_labels=["Fact"],
            )

            formatted: list[dict[str, Any]] = []
            for result in results:
                node = result.get("node")
                if node:
                    props = node.properties if hasattr(node, "properties") else {}
                    formatted.append(
                        {
                            "document": props.get("statement", ""),
                            "metadata": {
                                "confidence": props.get("confidence", 0.9),
                                "timestamp": props.get("timestamp", ""),
                            },
                            "distance": 1.0
                            - result.get("score", 0.0),  # Convert similarity to distance
                            "id": node.id if hasattr(node, "id") else None,
                        }
                    )
            return formatted
        except Exception as e:
            logger.debug(f"Query facts error: {e}")
            return []

    def query_recipes(self, query: str, n_results: int = 10) -> list[dict[str, Any]]:
        """
        Query recipes by semantic similarity.

        Returns list of results with node properties and score.
        """
        try:
            results = self.db.semantic_search(
                vector=query,  # Will be embedded by registered function
                k=n_results,
                index=RECIPES_VECTOR_INDEX,
                filter_labels=["Recipe"],
            )

            formatted: list[dict[str, Any]] = []
            for result in results:
                node = result.get("node")
                if node:
                    props = node.properties if hasattr(node, "properties") else {}
                    formatted.append(
                        {
                            "document": props.get("description", ""),
                            "metadata": {
                                "confidence": props.get("confidence", 0.9),
                                "timestamp": props.get("timestamp", ""),
                            },
                            "distance": 1.0
                            - result.get("score", 0.0),  # Convert similarity to distance
                            "id": node.id if hasattr(node, "id") else None,
                        }
                    )
            return formatted
        except Exception as e:
            logger.debug(f"Query recipes error: {e}")
            return []

    def query_entity(self, name: str) -> dict[str, Any] | None:
        """
        Query an entity by name from GrafitoDB.

        Returns entity with type and properties, or None if not found.
        """
        try:
            nodes = self.db.match_nodes(
                labels=["Entity"],
                properties={"name": name},
                limit=1,
            )
            if nodes:
                node = nodes[0]
                # Extract type from labels (first label after "Entity")
                entity_type = "other"
                for label in node.labels:
                    if label != "Entity":
                        entity_type = label
                        break
                return {
                    "name": name,
                    "type": entity_type,
                    "confidence": float(node.properties.get("confidence", 0.9)),
                    "extracted_at": node.properties.get("extracted_at", ""),
                }
        except Exception:
            pass
        return None

    def query_relationships(
        self, subject: str = "", predicate: str = "", n_results: int = 20
    ) -> list[dict[str, Any]]:
        """
        Query relationships from GrafitoDB.

        If subject is provided, returns both:
        - Outgoing relationships (entity as subject)
        - Incoming relationships (entity as object)
        Otherwise returns all relationships.

        Returns list of relationships.
        """
        relationships: list[dict[str, Any]] = []
        try:
            if subject:
                # Find the subject entity node
                subject_nodes = self.db.match_nodes(
                    labels=["Entity"],
                    properties={"name": subject},
                    limit=1,
                )
                if not subject_nodes:
                    return relationships

                subject_node = subject_nodes[0]

                # Get outgoing relationships
                outgoing_rels = self.db.match_relationships(
                    source_id=subject_node.id,
                    rel_type=predicate if predicate else None,
                )
                for rel in outgoing_rels:
                    rel_pred = rel.type
                    if rel_pred in ["type", "confidence", "extracted_at"]:
                        continue
                    # Get target node name
                    target_node = self.db.get_node(rel.target_id)
                    if target_node:
                        relationships.append(
                            {
                                "subject": subject,
                                "predicate": rel_pred,
                                "object": target_node.properties.get("name", ""),
                                "confidence": float(rel.properties.get("confidence", 0.9)),
                                "extracted_at": rel.properties.get("extracted_at", ""),
                            }
                        )

                # Get incoming relationships - find nodes that point to this entity
                # Use Cypher query for efficiency
                # Note: GrafitoDB Cypher doesn't support NOT IN, so we filter in Python
                safe_subject = _escape_cypher_string(subject)
                cypher = (
                    "MATCH (e:Entity {name: '" + safe_subject + "'}) "
                    "MATCH (source:Entity)-[r]->(e) "
                    "RETURN source.name AS source_name, r, r.confidence AS confidence, "
                    "r.extracted_at AS extracted_at "
                    "LIMIT " + str(n_results)
                )
                try:
                    incoming_results = self.db.execute(cypher)
                    for result in incoming_results:
                        # Extract type from relationship object
                        rel_obj = result.get("r", {})
                        rel_type = rel_obj.get("type", "") if isinstance(rel_obj, dict) else ""

                        # Skip metadata predicates (filter in Python)
                        if rel_type in ["type", "confidence", "extracted_at"]:
                            continue

                        relationships.append(
                            {
                                "subject": result.get("source_name", ""),
                                "predicate": rel_type,
                                "object": subject,
                                "confidence": float(result.get("confidence", 0.9)),
                                "extracted_at": result.get("extracted_at", ""),
                            }
                        )
                except Exception as e:
                    logger.debug(f"Cypher query error for incoming relationships: {e}")
            else:
                # Get all relationships using Cypher
                # Note: GrafitoDB Cypher doesn't support NOT IN, so we filter in Python
                cypher = (
                    "MATCH (source:Entity)-[r]->(target:Entity) "
                    "RETURN source.name AS subject, r, target.name AS object, "
                    "r.confidence AS confidence, r.extracted_at AS extracted_at "
                    "LIMIT " + str(n_results)
                )
                try:
                    all_results = self.db.execute(cypher)
                    for result in all_results:
                        # Extract type from relationship object
                        rel_obj = result.get("r", {})
                        rel_type = rel_obj.get("type", "") if isinstance(rel_obj, dict) else ""

                        # Skip metadata predicates (filter in Python)
                        if rel_type in ["type", "confidence", "extracted_at"]:
                            continue

                        relationships.append(
                            {
                                "subject": result.get("subject", ""),
                                "predicate": rel_type,
                                "object": result.get("object", ""),
                                "confidence": float(result.get("confidence", 0.9)),
                                "extracted_at": result.get("extracted_at", ""),
                            }
                        )
                except Exception as e:
                    logger.debug(f"Cypher query error for all relationships: {e}")
        except Exception as e:
            logger.debug(f"Query relationships error: {e}")

        return relationships

    def get_stats(self) -> dict[str, Any]:
        """Get stats from GrafitoDB."""
        node_count = self.db.get_node_count()
        relationship_count = self.db.get_relationship_count()

        # Count facts and recipes efficiently using Cypher COUNT
        facts_count = 0
        recipes_count = 0
        try:
            result = self.db.execute("MATCH (n:Fact) RETURN count(n) AS cnt")
            if result:
                facts_count = result[0].get("cnt", 0)
        except Exception:
            # Fallback: load nodes (slower but reliable)
            facts_count = len(self.db.match_nodes(labels=["Fact"]))
        try:
            result = self.db.execute("MATCH (n:Recipe) RETURN count(n) AS cnt")
            if result:
                recipes_count = result[0].get("cnt", 0)
        except Exception:
            recipes_count = len(self.db.match_nodes(labels=["Recipe"]))

        return {
            "grafitodb": {
                "nodes": node_count,
                "relationships": relationship_count,
                "facts": facts_count,
                "recipes": recipes_count,
            },
        }

    def close(self) -> None:
        """Close GrafitoDB connection."""
        try:
            self.db.close()
            logger.info("GrafitoDB closed")
        except Exception as e:
            logger.error(f"Failed to close GrafitoDB: {e}")

    def clear_all(self) -> dict[str, Any]:
        """
        Clear all data from GrafitoDB.

        Returns stats showing what was cleared.
        """
        stats_before = self.get_stats()

        # Delete all nodes and relationships
        try:
            # Get all nodes and delete them
            all_nodes = self.db.match_nodes(limit=10000)
            for node in all_nodes:
                self.db.delete_node(node.id)
            logger.info("GrafitoDB cleared")
        except Exception as e:
            logger.error(f"Failed to clear GrafitoDB: {e}")

        stats_after = self.get_stats()

        return {
            "before": stats_before,
            "after": stats_after,
            "cleared": {
                "facts": stats_before["grafitodb"]["facts"] - stats_after["grafitodb"]["facts"],
                "recipes": stats_before["grafitodb"]["recipes"]
                - stats_after["grafitodb"]["recipes"],
                "nodes": stats_before["grafitodb"]["nodes"]
                - stats_after["grafitodb"]["nodes"],
                "relationships": stats_before["grafitodb"]["relationships"]
                - stats_after["grafitodb"]["relationships"],
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
        Retrieve context from GrafitoDB based on queries with composite scoring and token budget.

        Composite scoring:
        - confidence * similarity * time_factor

        Token budget: stops adding items when estimated token count reaches budget.

        Returns dict with 'relationships', 'facts', 'recipes', 'tokens_used', 'scores' lists.
        """
        all_items: list[dict[str, Any]] = []  # Pool of all scored items

        # Query GrafitoDB for entity relationships
        for entity in entities_queries:
            try:
                entity_rels = self.query_relationships(subject=entity, n_results=query_limit)
                for rel in entity_rels:
                    confidence = rel.get("confidence", 0.9)
                    timestamp = rel.get("extracted_at", "")

                    if confidence >= min_confidence:
                        score = self._calculate_score(confidence, None, timestamp, decay_days)
                        formatted = self._format_relationship_for_context(rel)
                        all_items.append(
                            {
                                "type": "relationship",
                                "data": rel,
                                "score": score,
                                "formatted": formatted,
                                "tokens": self._estimate_tokens(formatted),
                            }
                        )
            except Exception as e:
                logger.debug(f"Query entity relationships error for {entity}: {e}")

        # Query facts by semantic search
        if fact_query:
            try:
                fact_results = self.query_facts(fact_query, n_results=query_limit)
                for fact in fact_results:
                    confidence = fact.get("metadata", {}).get("confidence", 0.9)
                    distance = fact.get("distance", 1.0) or 1.0
                    timestamp = fact.get("metadata", {}).get("timestamp", "")

                    if confidence >= min_confidence and distance <= max_distance:
                        score = self._calculate_score(confidence, distance, timestamp, decay_days)
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

        # Query recipes by semantic search
        if recipe_query:
            try:
                recipe_results = self.query_recipes(recipe_query, n_results=query_limit)
                for recipe in recipe_results:
                    confidence = recipe.get("metadata", {}).get("confidence", 0.9)
                    distance = recipe.get("distance", 1.0) or 1.0
                    timestamp = recipe.get("metadata", {}).get("timestamp", "")

                    if confidence >= min_confidence and distance <= max_distance:
                        score = self._calculate_score(confidence, distance, timestamp, decay_days)
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
        relationships: list[dict[str, Any]] = []
        facts: list[dict[str, Any]] = []
        recipes: list[dict[str, Any]] = []
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
                # Add score to data item
                item["data"]["score"] = item["score"]
                relationships.append(item["data"])
                scores["relationships"].append(item["score"])
            elif item["type"] == "fact":
                # Add score to data item
                item["data"]["score"] = item["score"]
                facts.append(item["data"])
                scores["facts"].append(item["score"])
            elif item["type"] == "recipe":
                # Add score to data item
                item["data"]["score"] = item["score"]
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

    def format_context_for_prompt(
        self, context: dict[str, Any], available_tools: list[dict[str, Any]] | None = None
    ) -> str:
        """
        Format retrieved context for prompt augmentation.

        Returns a formatted string suitable for prepending to user message.
        Optionally includes tool instructions if tools are available.

        Args:
            context: Retrieved context (relationships, facts, recipes)
            available_tools: Optional list of available tools in OpenAI format
        """
        lines: list[str] = []

        # Add tool instructions if tools are available
        if available_tools:
            lines.append("Available tools you can use:")
            for tool in available_tools:
                if tool.get("type") == "function":
                    func = tool.get("function", {})
                    name = func.get("name", "")
                    desc = func.get("description", "")
                    lines.append(f"- {name}: {desc}")

            lines.append("")
            lines.append("Use tools when they would be helpful for answering the user's question.")
            lines.append("")
            lines.append("To use a tool, call it like a function in your response:")
            lines.append("Example: get_current_time()")
            lines.append('Example: search_web("recent Python tutorials")')
            lines.append('Example: query_memory("python programming")')
            lines.append("")
            lines.append("Only use tools when necessary. For simple questions, answer directly.")
            lines.append("")

        # Add context from memory
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
                # Don't truncate recipes - they need full procedural content
                lines.append(f"- {desc}")

        if lines:
            return "\n".join(lines) + "\n\n"
        return ""
