"""Extraction system for relationships, facts, and recipes."""

import contextlib
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from forma.config import Settings

logger = logging.getLogger(__name__)

# Path to extraction prompt template
PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "extraction.txt"
LOGS_DIR = Path(__file__).parent.parent.parent / "logs"


class ExtractionResult:
    """Structured extraction result."""

    def __init__(self, raw_response: str, extraction_prompt: str = "") -> None:
        self.raw_response = raw_response
        self.extraction_prompt = extraction_prompt
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
        if "=== RELATIONSHIPS ===" in response:
            self._parse_text_format(response)
        else:
            self.parse_error = "No valid format found in response"
            logger.warning(self.parse_error)

    def _parse_text_format(self, response: str) -> None:
        """Parse the structured text format with confidence on separate lines."""
        try:
            # Helper function to parse two-line blocks (content + confidence)
            def parse_blocks(section_text: str) -> list[tuple[str, float]]:
                """Parse blocks where each item has content line + confidence line."""
                blocks = re.split(r"\n\n+", section_text.strip())
                results = []
                for block in blocks:
                    block = block.strip()
                    # Skip empty blocks and N/A variants (including truncated)
                    if not block or block.upper() in ("N/A", "NA", "N"):
                        continue

                    lines = block.split("\n")
                    if len(lines) < 1:
                        continue

                    # First line is content, look for confidence on any line
                    content = lines[0].strip()
                    confidence = 0.9

                    # Find confidence line
                    for line in lines:
                        conf_match = re.match(r"confidence:\s*([\d.]+)", line.strip())
                        if conf_match:
                            confidence = float(conf_match.group(1))
                            break

                    # Remove confidence line from content if it's on line 2
                    if len(lines) >= 2 and lines[1].strip().startswith("confidence:"):
                        content = lines[0].strip()

                    if content and not content.startswith("confidence:"):
                        results.append((content, confidence))

                return results

            # Parse relationships - Format: Subject -> predicate -> Object on line 1
            rels_section = re.search(
                r"=== RELATIONSHIPS ===\n(.*?)"
                r"(?==== FACTS ===|=== RECIPES ===|=== ENTITIES_QUERY ===|"
                r"=== END ===|$)",
                response,
                re.DOTALL,
            )
            if rels_section:
                blocks = parse_blocks(rels_section.group(1))
                for content, confidence in blocks:
                    # Must have -> to be a relationship
                    if "->" not in content:
                        continue
                    # Format: Subject -> predicate -> Object
                    match = re.match(
                        r"[\[\(]?([^\[\]\(\)]+)[\]\)]?\s*->\s*(.+?)\s*->\s*"
                        r"[\[\(]?([^\[\]\(\)]+)[\]\)]?",
                        content,
                    )
                    if match:
                        self.relationships.append(
                            {
                                "subject": match.group(1).strip(),
                                "predicate": match.group(2).strip(),
                                "object": match.group(3).strip(),
                                "confidence": max(0.0, min(1.0, confidence)),
                            }
                        )

            # Parse facts - Format: Statement on line 1, confidence on line 2
            facts_section = re.search(
                r"=== FACTS ===\n(.*?)(?==== RECIPES ===|=== ENTITIES_QUERY ===|=== END ===|$)",
                response,
                re.DOTALL,
            )
            if facts_section:
                blocks = parse_blocks(facts_section.group(1))
                for content, confidence in blocks:
                    # Skip if content is empty or N/A
                    if content.upper() == "N/A":
                        continue
                    # Skip lines that look like relationships
                    if "->" in content:
                        continue
                    self.facts.append(
                        {
                            "statement": content,
                            "confidence": max(0.0, min(1.0, confidence)),
                        }
                    )

            # Parse recipes - Format: Multi-line description, confidence at end
            # Recipes can span multiple lines, unlike relationships/facts
            recipes_section = re.search(
                r"=== RECIPES ===\n(.*?)"
                r"(?==== ENTITIES_QUERY ===|=== FACT_QUERY ===|"
                r"=== RECIPE_QUERY ===|=== END ===|$)",
                response,
                re.DOTALL,
            )
            if recipes_section:
                # Parse recipes with multi-line content support
                def parse_recipe_blocks(section_text: str) -> list[tuple[str, float]]:
                    """Parse recipe blocks where content can span multiple lines."""
                    blocks = re.split(r"\n\n+", section_text.strip())
                    results = []
                    for block in blocks:
                        block = block.strip()
                        if not block or block.upper() in ("N/A", "NA", "N"):
                            continue

                        lines = block.split("\n")
                        if len(lines) < 1:
                            continue

                        # Find confidence line (usually last line)
                        confidence = 0.9
                        content_lines = []
                        for line in lines:
                            conf_match = re.match(r"confidence:\s*([\d.]+)", line.strip())
                            if conf_match:
                                confidence = float(conf_match.group(1))
                            else:
                                content_lines.append(line.strip())

                        # Join all content lines for multi-line recipe descriptions
                        content = "\n".join(content_lines)
                        if content and not content.startswith("confidence:"):
                            results.append((content, confidence))

                    return results

                blocks = parse_recipe_blocks(recipes_section.group(1))
                for content, confidence in blocks:
                    if content.upper() == "N/A":
                        continue
                    self.recipes.append(
                        {
                            "description": content,
                            "confidence": max(0.0, min(1.0, confidence)),
                        }
                    )

            # Parse entities query - list of entity names to query
            entities_query_section = re.search(
                r"=== ENTITIES_QUERY ===\n(.*?)"
                r"(?==== FACT_QUERY ===|=== RECIPE_QUERY ===|=== END ===|$)",
                response,
                re.DOTALL,
            )
            if entities_query_section:
                entities_query_text = entities_query_section.group(1).strip()
                for line in entities_query_text.split("\n"):
                    line = line.strip()
                    # Skip confidence lines and empty/N/A
                    if not line or line == "N/A" or line.startswith("confidence:"):
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
                # Remove any confidence line if present
                lines = fact_query_text.split("\n")
                fact_query_text = lines[0].strip() if lines else ""
                if (
                    fact_query_text
                    and fact_query_text != "N/A"
                    and not fact_query_text.startswith("confidence:")
                ):
                    self.fact_query = fact_query_text

            # Parse recipe query - single natural language query
            recipe_query_section = re.search(
                r"=== RECIPE_QUERY ===\n(.*?)(?==== END ===|$)",
                response,
                re.DOTALL,
            )
            if recipe_query_section:
                recipe_query_text = recipe_query_section.group(1).strip()
                # Remove any confidence line if present
                lines = recipe_query_text.split("\n")
                recipe_query_text = lines[0].strip() if lines else ""
                if (
                    recipe_query_text
                    and recipe_query_text != "N/A"
                    and not recipe_query_text.startswith("confidence:")
                ):
                    self.recipe_query = recipe_query_text

        except Exception as e:
            self.parse_error = f"Text parse error: {e}"
            logger.warning(self.parse_error)

    def is_valid(self) -> bool:
        """Check if extraction produced valid results."""
        return self.parse_error is None and (
            len(self.relationships) > 0
            or len(self.facts) > 0
            or len(self.recipes) > 0
            or len(self.entities_queries) > 0
            or self.fact_query is not None
            or self.recipe_query is not None
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
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
    """Handles extraction of relationships, facts, and recipes from text."""

    def __init__(self, settings: Settings, proxy: Any = None) -> None:
        self.settings = settings
        self.proxy = proxy
        self._ensure_logs_dir()
        self._log_file = open(  # noqa: SIM115
            LOGS_DIR / "extractions.jsonl", "a", encoding="utf-8"
        )

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
        Extract relationships, facts, and recipes from text.

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

        # Call extraction LLM (reuse global proxy if available)
        proxy = self.proxy or OpenAIProxy(self.settings)
        response = await proxy.extract(
            messages=messages,
            max_tokens=1024,  # Increased for complex extractions
            temperature=0.1,
        )

        # Parse result
        prompt_content = messages[0].get("content", "")
        result = ExtractionResult(response, extraction_prompt=prompt_content)

        # Log extraction
        self._log_extraction(text, result)

        return result

    def close(self) -> None:
        """Close the persistent log file handle."""
        with contextlib.suppress(Exception):
            self._log_file.close()

    def _log_extraction(self, text: str, result: ExtractionResult) -> None:
        """Log extraction result to file."""
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "source_text": text[:500],  # Truncate for logging
            "raw_response": result.raw_response,  # Include raw response for debugging
            "extraction": result.to_dict(),
        }

        try:
            self._log_file.write(json.dumps(log_entry) + "\n")
            self._log_file.flush()
            logger.debug(
                f"Logged extraction: {len(result.relationships)} relationships, "
                f"{len(result.facts)} facts, {len(result.recipes)} recipes"
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
        """Async extraction from all user and assistant messages."""
        texts = []
        for msg in messages:
            role = msg.get("role")
            if role in ("user", "assistant"):
                content = msg.get("content", "")
                if isinstance(content, str):
                    prefix = "User:" if role == "user" else "Assistant:"
                    texts.append(f"{prefix}\n{content}")
                elif isinstance(content, list):
                    # Handle multi-modal content (text parts)
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            prefix = "User:" if role == "user" else "Assistant:"
                            texts.append(f"{prefix}\n{part.get('text', '')}")

        if not texts:
            return ExtractionResult("")

        # Combine all messages for extraction
        combined_text = "\n\n---\n\n".join(texts)
        return await self.extract_from_text_async(combined_text)
