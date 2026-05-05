"""Extraction system for entities, relationships, and facts."""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from forma.config import Settings

logger = logging.getLogger(__name__)

# Path to extraction prompt template
PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "extraction.txt"
LOGS_DIR = Path(__file__).parent.parent.parent / "logs"


class ExtractionResult:
    """Structured extraction result."""

    def __init__(self, raw_response: str) -> None:
        self.raw_response = raw_response
        self.entities: list[dict[str, Any]] = []
        self.relationships: list[dict[str, Any]] = []
        self.facts: list[dict[str, Any]] = []
        self.recipes: list[dict[str, Any]] = []
        self.entities_queries: list[str] = []
        self.fact_query: str | None = None
        self.recipe_query: str | None = None
        self.parse_error: str | None = None
        self._parse_response()

    def _parse_response(self) -> None:
        """Parse structured text from extraction response."""
        response = self.raw_response

        # Parse text format
        if "=== ENTITIES ===" in response:
            self._parse_text_format(response)
        else:
            self.parse_error = "No valid format found in response"
            logger.warning(self.parse_error)

    def _parse_text_format(self, response: str) -> None:
        """Parse the structured text format."""
        try:
            # Parse entities - look for lines with (type) pattern
            entities_section = re.search(
                r"=== ENTITIES ===\n(.*?)(?==== RELATIONSHIPS ===|=== FACTS ===|=== RECIPES ===|=== ENTITIES_QUERY ===|=== END ===|$)",
                response,
                re.DOTALL,
            )
            if entities_section:
                for line in entities_section.group(1).strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    # Format: [Name] (type) or Name (type) - brackets optional
                    match = re.match(
                        r"\[?([^\[\]\(\)]+)\]?\s*\(([^)]+)\)\s*(?:confidence:\s*([\d.]+))?", line
                    )
                    if match:
                        confidence = float(match.group(3)) if match.group(3) else 0.9
                        self.entities.append(
                            {
                                "name": match.group(1).strip(),
                                "type": match.group(2)
                                .strip()
                                .lower(),  # Normalize type to lowercase
                                "confidence": max(0.0, min(1.0, confidence)),
                            }
                        )

            # Parse relationships - look for lines with -> pattern
            rels_section = re.search(
                r"=== RELATIONSHIPS ===\n(.*?)(?==== FACTS ===|=== RECIPES ===|=== ENTITIES_QUERY ===|=== END ===|$)",
                response,
                re.DOTALL,
            )
            if rels_section:
                for line in rels_section.group(1).strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    # Must have -> to be a relationship
                    if "->" not in line:
                        continue
                    # Format: Subject -> predicate -> Object confidence: 0.9
                    # Need to capture object before confidence suffix
                    match = re.match(
                        r"[\[\(]?([^\[\]\(\)]+)[\]\)]?\s*->\s*(.+?)\s*->\s*[\[\(]?([^\[\]\(\)]+)[\]\)]?\s*(?:confidence:\s*([\d.]+))?",
                        line,
                    )
                    if match:
                        confidence = float(match.group(4)) if match.group(4) else 0.9
                        # Clean predicate and object - remove trailing confidence if accidentally captured
                        predicate = match.group(2).strip()
                        object = match.group(3).strip()
                        # Remove any "confidence: X.X" that might be in predicate/object
                        predicate = re.sub(r"\s*confidence:\s*[\d.]+\s*$", "", predicate)
                        object = re.sub(r"\s*confidence:\s*[\d.]+\s*$", "", object)
                        self.relationships.append(
                            {
                                "subject": match.group(1).strip(),
                                "predicate": predicate,
                                "object": object,
                                "confidence": max(0.0, min(1.0, confidence)),
                            }
                        )

            # Parse facts - clean sentences, may be on multiple lines or combined
            facts_section = re.search(
                r"=== FACTS ===\n(.*?)(?==== RECIPES ===|=== ENTITIES_QUERY ===|=== END ===|$)",
                response,
                re.DOTALL,
            )
            if facts_section:
                facts_text = facts_section.group(1).strip()
                # Split by periods if all facts are combined in one line
                if "." in facts_text and "\n" not in facts_text.strip():
                    # Facts are combined - split by sentences
                    sentences = re.split(r"\.\s+", facts_text)
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                        # Remove confidence suffix if present
                        match = re.match(r"(.+?)\s*(?:confidence:\s*([\d.]+))?$", sentence)
                        if match:
                            statement = match.group(1).strip()
                            confidence = float(match.group(2)) if match.group(2) else 0.9
                            # Skip N/A values
                            if statement.upper() == "N/A":
                                continue
                            self.facts.append(
                                {
                                    "statement": statement,
                                    "confidence": max(0.0, min(1.0, confidence)),
                                }
                            )
                else:
                    # Facts are on separate lines
                    for line in facts_text.split("\n"):
                        line = line.strip()
                        if not line:
                            continue
                        # Skip lines that look like entities (have (type) pattern)
                        if "(" in line and re.search(r"\([^)]+\)", line):
                            continue
                        # Skip lines that look like relationships (have ->)
                        if "->" in line:
                            continue
                        # Format: Statement confidence: 0.9 (confidence optional)
                        match = re.match(r"(.+?)\s*(?:confidence:\s*([\d.]+))?$", line)
                        if match:
                            statement = match.group(1).strip()
                            confidence = float(match.group(2)) if match.group(2) else 0.9
                            # Skip if statement is just "confidence: X.X" or empty or N/A
                            if (
                                statement.lower().startswith("confidence:")
                                or not statement
                                or statement.upper() == "N/A"
                            ):
                                continue
                            self.facts.append(
                                {
                                    "statement": statement,
                                    "confidence": max(0.0, min(1.0, confidence)),
                                }
                            )

            # Parse recipes - procedural knowledge as plain text
            recipes_section = re.search(
                r"=== RECIPES ===\n(.*?)(?==== ENTITIES_QUERY ===|=== FACT_QUERY ===|=== RECIPE_QUERY ===|=== END ===|$)",
                response,
                re.DOTALL,
            )
            if recipes_section:
                recipes_text = recipes_section.group(1).strip()
                # Split by blank lines to get individual recipes
                recipe_blocks = re.split(r"\n\n+", recipes_text)
                for block in recipe_blocks:
                    block = block.strip()
                    if not block:
                        continue

                    # Extract confidence if present
                    conf_match = re.search(r"confidence:\s*([\d.]+)", block)
                    confidence = float(conf_match.group(1)) if conf_match else 0.9

                    # Remove confidence suffix from the description
                    description = re.sub(r"\s*confidence:\s*[\d.]+\s*$", "", block)
                    description = description.strip()

                    # Skip N/A values
                    if description.upper() == "N/A":
                        continue

                    if description:
                        self.recipes.append(
                            {
                                "description": description,
                                "confidence": max(0.0, min(1.0, confidence)),
                            }
                        )

            # Parse entities query - list of entity names to query
            entities_query_section = re.search(
                r"=== ENTITIES_QUERY ===\n(.*?)(?==== FACT_QUERY ===|=== RECIPE_QUERY ===|=== END ===|$)",
                response,
                re.DOTALL,
            )
            if entities_query_section:
                entities_query_text = entities_query_section.group(1).strip()
                for line in entities_query_text.split("\n"):
                    line = line.strip()
                    if not line or line == "N/A":
                        continue
                    self.entities_queries.append(line)

            # Parse fact query - single natural language query
            fact_query_section = re.search(
                r"=== FACT_QUERY ===\n(.*?)(?==== RECIPE_QUERY ===|=== END ===|$)",
                response,
                re.DOTALL,
            )
            if fact_query_section:
                fact_query_text = fact_query_section.group(1).strip()
                if fact_query_text and fact_query_text != "N/A":
                    self.fact_query = fact_query_text

            # Parse recipe query - single natural language query
            recipe_query_section = re.search(
                r"=== RECIPE_QUERY ===\n(.*?)(?==== END ===|$)",
                response,
                re.DOTALL,
            )
            if recipe_query_section:
                recipe_query_text = recipe_query_section.group(1).strip()
                if recipe_query_text and recipe_query_text != "N/A":
                    self.recipe_query = recipe_query_text

        except Exception as e:
            self.parse_error = f"Text parse error: {e}"
            logger.warning(self.parse_error)

    def is_valid(self) -> bool:
        """Check if extraction produced valid results."""
        return self.parse_error is None and (
            len(self.entities) > 0
            or len(self.relationships) > 0
            or len(self.facts) > 0
            or len(self.recipes) > 0
            or len(self.entities_queries) > 0
            or self.fact_query is not None
            or self.recipe_query is not None
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "entities": self.entities,
            "relationships": self.relationships,
            "facts": self.facts,
            "recipes": self.recipes,
            "entities_queries": self.entities_queries,
            "fact_query": self.fact_query,
            "recipe_query": self.recipe_query,
            "parse_error": self.parse_error,
        }

    def has_queries(self) -> bool:
        """Check if extraction has any queries to execute."""
        return (
            len(self.entities_queries) > 0
            or self.fact_query is not None
            or self.recipe_query is not None
        )


class Extractor:
    """Handles extraction of entities, relationships, and facts from text."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._ensure_logs_dir()

    def _load_prompt_template(self) -> str:
        """Load the extraction prompt template from file (called each time for live updates)."""
        try:
            with open(PROMPT_TEMPLATE_PATH, encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt template not found: {PROMPT_TEMPLATE_PATH}")
            raise

    def _ensure_logs_dir(self) -> None:
        """Ensure logs directory exists."""
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

    def build_extraction_prompt(self, text: str) -> list[dict[str, str]]:
        """Build the extraction prompt messages (reads template fresh each time)."""
        prompt_template = self._load_prompt_template()
        prompt = prompt_template.replace("{{TEXT}}", text)
        return [{"role": "user", "content": prompt}]

    def extract_from_text(self, text: str) -> ExtractionResult:
        """
        Extract entities, relationships, and facts from text.

        Note: This is a synchronous wrapper for async extraction.
        Use extract_from_text_async for proper async operation.
        """
        import asyncio

        return asyncio.run(self.extract_from_text_async(text))

    async def extract_from_text_async(self, text: str) -> ExtractionResult:
        """Async extraction from text."""
        if not text.strip():
            return ExtractionResult("")

        from forma.proxy import OpenAIProxy

        # Build prompt
        messages = self.build_extraction_prompt(text)

        # Call extraction LLM
        proxy = OpenAIProxy(self.settings)
        response = await proxy.extract(
            messages=messages,
            max_tokens=512,  # Lower since no reasoning output
            temperature=0.1,
            disable_reasoning=True,
        )

        # Parse result
        result = ExtractionResult(response)

        # Log extraction
        self._log_extraction(text, result)

        return result

    def _log_extraction(self, text: str, result: ExtractionResult) -> None:
        """Log extraction result to file."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "source_text": text[:500],  # Truncate for logging
            "raw_response": result.raw_response,  # Include raw response for debugging
            "extraction": result.to_dict(),
        }

        log_file = LOGS_DIR / "extractions.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
            logger.debug(
                f"Logged extraction: {len(result.entities)} entities, {len(result.relationships)} relationships, {len(result.facts)} facts"
            )
        except Exception as e:
            logger.error(f"Failed to log extraction: {e}")

    def extract_from_messages(self, messages: list[dict[str, str]]) -> ExtractionResult:
        """
        Extract from all user messages in a conversation.

        Note: This is a synchronous wrapper for async extraction.
        Use extract_from_messages_async for proper async operation.
        """
        import asyncio

        return asyncio.run(self.extract_from_messages_async(messages))

    async def extract_from_messages_async(self, messages: list[dict[str, str]]) -> ExtractionResult:
        """Async extraction from all user messages."""
        # Collect all user messages
        user_texts = []
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    user_texts.append(content)
                elif isinstance(content, list):
                    # Handle multi-modal content (text parts)
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            user_texts.append(part.get("text", ""))

        if not user_texts:
            return ExtractionResult("")

        # Combine all user messages for extraction
        combined_text = "\n\n---\n\n".join(user_texts)
        return await self.extract_from_text_async(combined_text)
