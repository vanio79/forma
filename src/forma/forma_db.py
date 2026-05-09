"""Forma database for system data and request history.

Stores requests, extractions, retrievals, and upstream configurations.
Uses SQLite for lightweight, file-based storage.
"""

import json
import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class FormaDatabase:
    """SQLite-based database for Forma operations and configuration."""

    # SQL schema - simplified design
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS upstreams (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        upstream_model TEXT DEFAULT '',
        base_url TEXT NOT NULL,
        api_key TEXT DEFAULT '',
        timeout REAL DEFAULT 300.0,
        is_enabled INTEGER DEFAULT 1,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
    );
    
    CREATE TABLE IF NOT EXISTS agents (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        purpose TEXT NOT NULL,
        instruction_prompt TEXT NOT NULL,
        upstream_id TEXT NULL,
        tools_enabled INTEGER DEFAULT 0,
        tool_whitelist TEXT DEFAULT '[]',
        max_iterations INTEGER DEFAULT 5,
        is_enabled INTEGER DEFAULT 1,
        rag_config TEXT DEFAULT '{}',
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        FOREIGN KEY (upstream_id) REFERENCES upstreams(id)
    );
    
    CREATE TABLE IF NOT EXISTS requests (
        id TEXT PRIMARY KEY,
        model TEXT,
        upstream_id TEXT DEFAULT '',
        user_prompt TEXT,
        history TEXT,
        extraction_response TEXT,
        extraction_ms REAL DEFAULT 0.0,
        augmented_prompt TEXT,
        agent_response TEXT,
        timestamp INTEGER NOT NULL
    );
    
    CREATE TABLE IF NOT EXISTS extractions (
        id TEXT PRIMARY KEY,
        request_id TEXT NOT NULL,
        extraction_type TEXT,
        data TEXT,
        confidence REAL DEFAULT 0.9
    );
    
    CREATE TABLE IF NOT EXISTS retrievals (
        id TEXT PRIMARY KEY,
        request_id TEXT NOT NULL,
        retrieval_type TEXT,
        data TEXT,
        confidence REAL DEFAULT 0.9,
        score REAL DEFAULT 0.0
    );
    
    CREATE INDEX IF NOT EXISTS idx_upstreams_name ON upstreams(name);
    CREATE INDEX IF NOT EXISTS idx_agents_name ON agents(name);
    CREATE INDEX IF NOT EXISTS idx_agents_enabled ON agents(is_enabled);
    CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(timestamp);
    CREATE INDEX IF NOT EXISTS idx_requests_upstream_id ON requests(upstream_id);
    CREATE INDEX IF NOT EXISTS idx_extractions_request_id ON extractions(request_id);
    CREATE INDEX IF NOT EXISTS idx_retrievals_request_id ON retrievals(request_id);
    """

    def __init__(self, db_path: str = "./data/forma.db", max_records: int = 100):
        """Initialize the database.

        Args:
            db_path: Path to SQLite database file
            max_records: Maximum number of request records to keep (older ones are pruned)
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_records = max_records
        self._local = threading.local()

        # Initialize database
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path), check_same_thread=False, timeout=30.0
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    @contextmanager
    def _transaction(self):
        """Transaction context manager."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Transaction error: {e}")
            raise

    def _init_db(self):
        """Initialize database schema."""
        with self._transaction() as conn:
            conn.executescript(self.SCHEMA)
            # Add extraction_prompt column if it doesn't exist (migration)
            try:
                conn.execute("ALTER TABLE requests ADD COLUMN extraction_prompt TEXT")
            except sqlite3.OperationalError:
                # Column already exists, ignore
                pass
            # Add agent_id column if it doesn't exist (migration for multi-agent)
            try:
                conn.execute("ALTER TABLE requests ADD COLUMN agent_id TEXT DEFAULT ''")
                # Create index for agent_id after column is added
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_requests_agent_id ON requests(agent_id)"
                )
            except sqlite3.OperationalError:
                # Column or index already exists, ignore
                pass
            # Add rag_config column if it doesn't exist (migration for agent-specific RAG)
            try:
                conn.execute("ALTER TABLE agents ADD COLUMN rag_config TEXT DEFAULT '{}'")
            except sqlite3.OperationalError:
                # Column already exists, ignore
                pass
        logger.info(f"Forma database initialized at {self.db_path}")

    def _prune_old_records(self):
        """Remove records exceeding max_records limit."""
        with self._transaction() as conn:
            count = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
            if count > self.max_records:
                # Find cutoff timestamp
                cutoff = conn.execute(
                    "SELECT timestamp FROM requests ORDER BY timestamp DESC LIMIT 1 OFFSET ?",
                    (self.max_records,),
                ).fetchone()

                if cutoff:
                    cutoff_ts = cutoff[0]
                    # Delete old records from all tables
                    old_ids = conn.execute(
                        "SELECT id FROM requests WHERE timestamp < ?", (cutoff_ts,)
                    ).fetchall()
                    old_ids = [row[0] for row in old_ids]

                    if old_ids:
                        placeholders = ",".join("?" * len(old_ids))
                        conn.execute(
                            f"DELETE FROM extractions WHERE request_id IN ({placeholders})", old_ids
                        )
                        conn.execute(
                            f"DELETE FROM retrievals WHERE request_id IN ({placeholders})", old_ids
                        )
                        conn.execute(f"DELETE FROM requests WHERE id IN ({placeholders})", old_ids)
                        logger.debug(f"Pruned {len(old_ids)} old records")

    def _format_history(self, messages: list[dict]) -> str:
        """Format chat messages as human-readable history."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str):
                lines.append(f"{role.upper()}: {content}")
            elif isinstance(content, list):
                # Handle multi-modal content
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        lines.append(f"{role.upper()}: {part.get('text', '')}")
        return "\n\n".join(lines)

    def _format_extraction_data(self, item: dict, extraction_type: str) -> str:
        """Format extraction item as human-readable string."""
        if extraction_type == "entity":
            return f"{item.get('name', 'Unknown')} ({item.get('type', 'unknown')})"
        elif extraction_type == "relationship":
            return f"{item.get('subject', '')} -> {item.get('predicate', '')} -> {item.get('object', '')}"
        elif extraction_type == "fact":
            return item.get("statement", item.get("fact", str(item)))
        elif extraction_type == "recipe":
            return item.get("description", str(item))
        return str(item)

    def record_request(
        self,
        model: str,
        user_prompt: str,
        messages: list[dict],
        extraction_response: str = "",
        extraction_prompt: str = "",
        extraction_ms: float = 0.0,
        augmented_prompt: str = "",
        agent_response: str = "",
    ) -> str:
        """Record a request with all associated data.

        Returns:
            The request ID (UUID)
        """
        request_id = str(uuid.uuid4())
        timestamp = int(datetime.now(UTC).timestamp())
        history = self._format_history(messages)

        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO requests (id, model, user_prompt, history, extraction_response,
                                      extraction_prompt, extraction_ms, augmented_prompt,
                                      agent_response, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    model,
                    user_prompt,
                    history,
                    extraction_response,
                    extraction_prompt,
                    extraction_ms,
                    augmented_prompt,
                    agent_response,
                    timestamp,
                ),
            )

        self._prune_old_records()
        return request_id

    def record_extraction(
        self,
        request_id: str,
        extraction_type: str,
        data: str,
        confidence: float = 0.9,
    ) -> str:
        """Record a single extraction item."""
        extraction_id = str(uuid.uuid4())

        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO extractions (id, request_id, extraction_type, data, confidence)
                VALUES (?, ?, ?, ?, ?)
                """,
                (extraction_id, request_id, extraction_type, data, confidence),
            )

        return extraction_id

    def record_extractions_batch(
        self,
        request_id: str,
        relationships: list[dict],
        facts: list[dict],
        recipes: list[dict],
    ) -> int:
        """Record all extractions for a request.

        Returns:
            Number of extractions recorded
        """
        count = 0
        with self._transaction() as conn:
            for rel in relationships:
                extraction_id = str(uuid.uuid4())
                data = self._format_extraction_data(rel, "relationship")
                confidence = rel.get("confidence", 0.9)
                conn.execute(
                    "INSERT INTO extractions (id, request_id, extraction_type, data, confidence) VALUES (?, ?, ?, ?, ?)",
                    (extraction_id, request_id, "relationship", data, confidence),
                )
                count += 1

            for fact in facts:
                extraction_id = str(uuid.uuid4())
                data = self._format_extraction_data(fact, "fact")
                confidence = fact.get("confidence", 0.9)
                conn.execute(
                    "INSERT INTO extractions (id, request_id, extraction_type, data, confidence) VALUES (?, ?, ?, ?, ?)",
                    (extraction_id, request_id, "fact", data, confidence),
                )
                count += 1

            for recipe in recipes:
                extraction_id = str(uuid.uuid4())
                data = self._format_extraction_data(recipe, "recipe")
                confidence = recipe.get("confidence", 0.9)
                conn.execute(
                    "INSERT INTO extractions (id, request_id, extraction_type, data, confidence) VALUES (?, ?, ?, ?, ?)",
                    (extraction_id, request_id, "recipe", data, confidence),
                )
                count += 1

        return count

    def record_retrieval(
        self,
        request_id: str,
        retrieval_type: str,
        data: str,
        confidence: float = 0.9,
        score: float = 0.0,
    ) -> str:
        """Record a single retrieval item."""
        retrieval_id = str(uuid.uuid4())

        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO retrievals (id, request_id, retrieval_type, data, confidence, score)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (retrieval_id, request_id, retrieval_type, data, confidence, score),
            )

        return retrieval_id

    def record_retrievals_batch(
        self,
        request_id: str,
        results: list[dict],
    ) -> int:
        """Record all retrievals for a request.

        Args:
            results: List of result dicts with type, data, confidence, score

        Returns:
            Number of retrievals recorded
        """
        count = 0
        with self._transaction() as conn:
            for result in results:
                retrieval_id = str(uuid.uuid4())
                retrieval_type = result.get("type", "unknown")
                data = self._format_extraction_data(result.get("data", {}), retrieval_type)
                confidence = result.get("data", {}).get("confidence", 0.9)
                score = result.get("data", {}).get("score", 0.0)
                conn.execute(
                    "INSERT INTO retrievals (id, request_id, retrieval_type, data, confidence, score) VALUES (?, ?, ?, ?, ?, ?)",
                    (retrieval_id, request_id, retrieval_type, data, confidence, score),
                )
                count += 1

        return count

    def get_requests(self, limit: int = 100, offset: int = 0) -> list[dict]:
        """Get recent requests."""
        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT * FROM requests 
            ORDER BY timestamp DESC 
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

        return [dict(row) for row in rows]

    def get_request_detail(self, request_id: str) -> Optional[dict]:
        """Get full details for a specific request."""
        conn = self._get_connection()

        request_row = conn.execute("SELECT * FROM requests WHERE id = ?", (request_id,)).fetchone()

        if not request_row:
            return None

        extractions = conn.execute(
            "SELECT * FROM extractions WHERE request_id = ? ORDER BY extraction_type, id",
            (request_id,),
        ).fetchall()

        retrievals = conn.execute(
            "SELECT * FROM retrievals WHERE request_id = ? ORDER BY retrieval_type, id",
            (request_id,),
        ).fetchall()

        return {
            "request": dict(request_row),
            "extractions": [dict(row) for row in extractions],
            "retrievals": [dict(row) for row in retrievals],
        }

    def get_stats(self) -> dict[str, Any]:
        """Get summary statistics."""
        conn = self._get_connection()

        total_requests = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        total_extractions = conn.execute("SELECT COUNT(*) FROM extractions").fetchone()[0]
        total_retrievals = conn.execute("SELECT COUNT(*) FROM retrievals").fetchone()[0]

        avg_extraction_ms = (
            conn.execute(
                "SELECT AVG(extraction_ms) FROM requests WHERE extraction_ms > 0"
            ).fetchone()[0]
            or 0.0
        )

        # Count by type
        entity_count = conn.execute(
            "SELECT COUNT(*) FROM extractions WHERE extraction_type = 'entity'"
        ).fetchone()[0]
        relationship_count = conn.execute(
            "SELECT COUNT(*) FROM extractions WHERE extraction_type = 'relationship'"
        ).fetchone()[0]
        fact_count = conn.execute(
            "SELECT COUNT(*) FROM extractions WHERE extraction_type = 'fact'"
        ).fetchone()[0]
        recipe_count = conn.execute(
            "SELECT COUNT(*) FROM extractions WHERE extraction_type = 'recipe'"
        ).fetchone()[0]

        # Count upstreams
        upstream_count = conn.execute("SELECT COUNT(*) FROM upstreams").fetchone()[0]

        return {
            "total_requests": total_requests,
            "total_extractions": total_extractions,
            "total_retrievals": total_retrievals,
            "avg_extraction_ms": round(avg_extraction_ms, 2),
            "extractions_by_type": {
                "entities": entity_count,
                "relationships": relationship_count,
                "facts": fact_count,
                "recipes": recipe_count,
            },
            "upstream_count": upstream_count,
        }

    # === Upstream Management ===

    def _convert_upstream_row(self, row: sqlite3.Row) -> dict:
        """Convert a database row to a dict with proper boolean conversion."""
        data = dict(row)
        # Convert integer is_enabled to boolean
        data["is_enabled"] = bool(data.get("is_enabled", 1))
        return data

    def get_upstreams(self) -> list[dict]:
        """Get all upstream configurations."""
        conn = self._get_connection()
        rows = conn.execute("SELECT * FROM upstreams ORDER BY name ASC").fetchall()
        return [self._convert_upstream_row(row) for row in rows]

    def get_upstream_by_id(self, upstream_id: str) -> Optional[dict]:
        """Get a specific upstream by ID."""
        conn = self._get_connection()
        row = conn.execute("SELECT * FROM upstreams WHERE id = ?", (upstream_id,)).fetchone()
        return self._convert_upstream_row(row) if row else None

    def get_upstream_by_name(self, name: str) -> Optional[dict]:
        """Get a specific upstream by name (model name)."""
        conn = self._get_connection()
        row = conn.execute("SELECT * FROM upstreams WHERE name = ?", (name,)).fetchone()
        return self._convert_upstream_row(row) if row else None

    def get_enabled_upstream_by_name(self, name: str) -> Optional[dict]:
        """Get an enabled upstream by name (model name)."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM upstreams WHERE name = ? AND is_enabled = 1", (name,)
        ).fetchone()
        return self._convert_upstream_row(row) if row else None

    def create_upstream(
        self,
        id: str,
        name: str,
        base_url: str,
        upstream_model: str = "",
        api_key: str = "",
        timeout: float = 300.0,
        is_enabled: bool = True,
    ) -> str:
        """Create a new upstream configuration.

        Args:
            id: Unique ID
            name: Local model name (routing key - client requests with this model go to this upstream)
            upstream_model: Model name to send to upstream API (if empty, uses name)
            base_url: Upstream API base URL
            api_key: API key for authentication
            timeout: Request timeout in seconds
            is_enabled: Whether this upstream is enabled
        """
        timestamp = int(datetime.now(UTC).timestamp())
        # If upstream_model is empty, use name as the model to send upstream
        model_to_send = upstream_model or name

        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO upstreams (id, name, upstream_model, base_url, api_key, timeout, 
                                       is_enabled, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    id,
                    name,
                    model_to_send,
                    base_url,
                    api_key,
                    timeout,
                    int(is_enabled),
                    timestamp,
                    timestamp,
                ),
            )

        logger.info(f"Created upstream: {name} -> {model_to_send} ({base_url})")
        return id

    def update_upstream(
        self,
        id: str,
        name: str,
        base_url: str,
        upstream_model: str = "",
        api_key: str = "",
        timeout: float = 300.0,
        is_enabled: bool = True,
    ) -> bool:
        """Update an existing upstream configuration."""
        timestamp = int(datetime.now(UTC).timestamp())
        # If upstream_model is empty, use name as the model to send upstream
        model_to_send = upstream_model or name

        with self._transaction() as conn:
            result = conn.execute(
                """
                UPDATE upstreams 
                SET name = ?, upstream_model = ?, base_url = ?, api_key = ?, timeout = ?,
                    is_enabled = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    name,
                    model_to_send,
                    base_url,
                    api_key,
                    timeout,
                    int(is_enabled),
                    timestamp,
                    id,
                ),
            )

            if result.rowcount > 0:
                logger.info(f"Updated upstream: {name} -> {model_to_send}")
                return True
        return False

    def delete_upstream(self, upstream_id: str) -> bool:
        """Delete an upstream configuration."""
        with self._transaction() as conn:
            # Check if any agents reference this upstream
            agents = conn.execute(
                "SELECT name FROM agents WHERE upstream_id = ?", (upstream_id,)
            ).fetchall()
            if agents:
                logger.warning(
                    f"Cannot delete upstream {upstream_id}: used by agents {[a[0] for a in agents]}"
                )
                return False

            result = conn.execute("DELETE FROM upstreams WHERE id = ?", (upstream_id,))
            if result.rowcount > 0:
                logger.info(f"Deleted upstream: {upstream_id}")
                return True
        return False

    # === Agent Management ===

    def _convert_agent_row(self, row: sqlite3.Row) -> dict:
        """Convert a database row to a dict with proper type conversions."""
        data = dict(row)
        # Convert integer fields to boolean
        data["is_enabled"] = bool(data.get("is_enabled", 1))
        data["tools_enabled"] = bool(data.get("tools_enabled", 0))
        # Parse tool_whitelist JSON
        whitelist_str = data.get("tool_whitelist", "[]")
        try:
            data["tool_whitelist"] = json.loads(whitelist_str) if whitelist_str else []
        except json.JSONDecodeError:
            data["tool_whitelist"] = []
        # Parse rag_config JSON
        rag_config_str = data.get("rag_config", "{}")
        try:
            data["rag_config"] = json.loads(rag_config_str) if rag_config_str else {}
        except json.JSONDecodeError:
            data["rag_config"] = {}
        return data

    def get_agents(self) -> list[dict]:
        """Get all agent configurations."""
        conn = self._get_connection()
        rows = conn.execute("SELECT * FROM agents ORDER BY name ASC").fetchall()
        return [self._convert_agent_row(row) for row in rows]

    def get_enabled_agents(self) -> list[dict]:
        """Get all enabled agent configurations."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT * FROM agents WHERE is_enabled = 1 ORDER BY name ASC"
        ).fetchall()
        return [self._convert_agent_row(row) for row in rows]

    def get_agent_by_id(self, agent_id: str) -> Optional[dict]:
        """Get a specific agent by ID."""
        conn = self._get_connection()
        row = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
        return self._convert_agent_row(row) if row else None

    def get_agent_by_name(self, name: str) -> Optional[dict]:
        """Get a specific agent by name."""
        conn = self._get_connection()
        row = conn.execute("SELECT * FROM agents WHERE name = ?", (name,)).fetchone()
        return self._convert_agent_row(row) if row else None

    def get_enabled_agent_by_name(self, name: str) -> Optional[dict]:
        """Get an enabled agent by name."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM agents WHERE name = ? AND is_enabled = 1", (name,)
        ).fetchone()
        return self._convert_agent_row(row) if row else None

    def create_agent(
        self,
        id: str,
        name: str,
        purpose: str,
        instruction_prompt: str,
        upstream_id: Optional[str] = None,
        tools_enabled: bool = False,
        tool_whitelist: list[str] = [],
        max_iterations: int = 5,
        is_enabled: bool = True,
        rag_config: dict = {},
    ) -> str:
        """Create a new agent configuration.

        Args:
            id: Unique ID
            name: Agent name (human-readable, used for routing like @researcher)
            purpose: Brief description of agent's role
            instruction_prompt: System prompt/instruction for this agent
            upstream_id: Reference to upstream configuration (null = use default upstream)
            tools_enabled: Whether tools are enabled for this agent
            tool_whitelist: List of allowed tool names (empty = all tools)
            max_iterations: Max tool iterations for this agent
            is_enabled: Whether this agent is active
            rag_config: RAG configuration dict (enabled, token_budget, min_confidence, max_distance)
        """
        timestamp = int(datetime.now(UTC).timestamp())
        whitelist_json = json.dumps(tool_whitelist)
        rag_config_json = json.dumps(rag_config)

        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO agents (id, name, purpose, instruction_prompt, upstream_id,
                                    tools_enabled, tool_whitelist, max_iterations,
                                    is_enabled, rag_config, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    id,
                    name,
                    purpose,
                    instruction_prompt,
                    upstream_id,
                    int(tools_enabled),
                    whitelist_json,
                    max_iterations,
                    int(is_enabled),
                    rag_config_json,
                    timestamp,
                    timestamp,
                ),
            )

        logger.info(f"Created agent: {name} ({purpose})")
        return id

    def update_agent(
        self,
        id: str,
        name: str,
        purpose: str,
        instruction_prompt: str,
        upstream_id: Optional[str] = None,
        tools_enabled: bool = False,
        tool_whitelist: list[str] = [],
        max_iterations: int = 5,
        is_enabled: bool = True,
        rag_config: dict = {},
    ) -> bool:
        """Update an existing agent configuration."""
        timestamp = int(datetime.now(UTC).timestamp())
        whitelist_json = json.dumps(tool_whitelist)
        rag_config_json = json.dumps(rag_config)

        with self._transaction() as conn:
            result = conn.execute(
                """
                UPDATE agents 
                SET name = ?, purpose = ?, instruction_prompt = ?, upstream_id = ?,
                    tools_enabled = ?, tool_whitelist = ?, max_iterations = ?,
                    is_enabled = ?, rag_config = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    name,
                    purpose,
                    instruction_prompt,
                    upstream_id,
                    int(tools_enabled),
                    whitelist_json,
                    max_iterations,
                    int(is_enabled),
                    rag_config_json,
                    timestamp,
                    id,
                ),
            )

            if result.rowcount > 0:
                logger.info(f"Updated agent: {name}")
                return True
        return False

    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent configuration."""
        with self._transaction() as conn:
            result = conn.execute("DELETE FROM agents WHERE id = ?", (agent_id,))
            if result.rowcount > 0:
                logger.info(f"Deleted agent: {agent_id}")
                return True
        return False

    def clear_all(self):
        """Clear all tracking data."""
        with self._transaction() as conn:
            conn.execute("DELETE FROM extractions")
            conn.execute("DELETE FROM retrievals")
            conn.execute("DELETE FROM requests")
        logger.info("All Forma data cleared")

    def close(self):
        """Close database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# Global database instance (initialized on first use)
_db: Optional[FormaDatabase] = None


def get_db(db_path: str = None, max_records: int = 100) -> FormaDatabase:
    """Get or create the global database instance."""
    global _db
    if _db is None:
        _db = FormaDatabase(db_path=db_path or "./data/forma.db", max_records=max_records)
    return _db
