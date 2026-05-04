"""OpenAI-compatible proxy implementation."""

import json
import logging
from typing import Any

import httpx
from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse

from forma.config import Settings

logger = logging.getLogger(__name__)


class OpenAIProxy:
    """Proxy client for OpenAI-compatible APIs."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = settings.upstream_base_url.rstrip("/")
        self.headers = {
            "Content-Type": "application/json",
        }
        if settings.upstream_api_key:
            self.headers["Authorization"] = f"Bearer {settings.upstream_api_key}"

        # Separate embedding endpoint configuration
        self.embedding_url = (
            settings.embedding_base_url.rstrip("/")
            if settings.embedding_base_url
            else self.base_url
        )
        self.embedding_headers = {"Content-Type": "application/json"}
        if settings.embedding_api_key:
            self.embedding_headers["Authorization"] = f"Bearer {settings.embedding_api_key}"
        elif settings.upstream_api_key and not settings.embedding_base_url:
            # Fall back to upstream API key if embedding endpoint not separately configured
            self.embedding_headers["Authorization"] = f"Bearer {settings.upstream_api_key}"

        # Extraction LLM endpoint configuration
        self.extractor_url = (
            settings.extractor_base_url.rstrip("/")
            if settings.extractor_base_url
            else self.base_url
        )
        self.extractor_headers = {"Content-Type": "application/json"}
        if settings.extractor_api_key:
            self.extractor_headers["Authorization"] = f"Bearer {settings.extractor_api_key}"
        elif settings.upstream_api_key and not settings.extractor_base_url:
            # Fall back to upstream API key if extraction endpoint not separately configured
            self.extractor_headers["Authorization"] = f"Bearer {settings.upstream_api_key}"

    def _map_model(self, model: str) -> str:
        """Map local model name to upstream model name."""
        mapping = self.settings.get_model_mapping()
        return mapping.get(model, model)

    async def _forward_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Forward a non-streaming request to upstream API."""
        url = f"{self.base_url}{path}"
        if payload and "model" in payload:
            payload["model"] = self._map_model(payload["model"])

        async with httpx.AsyncClient(timeout=self.settings.upstream_timeout) as client:
            try:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=self.headers,
                    json=payload,
                )
                response.raise_for_status()
                return response.json()
            except httpx.TimeoutException as e:
                logger.error(f"Upstream timeout: {e}")
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Upstream API timeout",
                ) from e
            except httpx.HTTPStatusError as e:
                logger.error(f"Upstream error: {e.response.status_code} - {e.response.text}")
                raise HTTPException(
                    status_code=e.response.status_code,
                    detail=e.response.text,
                ) from e
            except httpx.RequestError as e:
                logger.error(f"Upstream connection error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Upstream connection error: {e}",
                ) from e

    async def stream_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any],
    ) -> StreamingResponse:
        """Forward a streaming request to upstream API."""
        url = f"{self.base_url}{path}"
        if "model" in payload:
            payload["model"] = self._map_model(payload["model"])

        async def stream_generator() -> Any:
            async with httpx.AsyncClient(timeout=self.settings.upstream_timeout) as client:
                try:
                    async with client.stream(
                        method=method,
                        url=url,
                        headers=self.headers,
                        json=payload,
                    ) as response:
                        response.raise_for_status()
                        async for chunk in response.aiter_bytes():
                            yield chunk
                except httpx.TimeoutException as e:
                    logger.error(f"Upstream stream timeout: {e}")
                    error_data = json.dumps(
                        {
                            "error": {
                                "message": "Upstream API timeout",
                                "type": "timeout_error",
                            }
                        }
                    )
                    yield f"data: {error_data}\n\n".encode()
                except httpx.HTTPStatusError as e:
                    logger.error(f"Upstream stream error: {e.response.status_code}")
                    error_data = json.dumps(
                        {
                            "error": {
                                "message": e.response.text,
                                "type": "upstream_error",
                            }
                        }
                    )
                    yield f"data: {error_data}\n\n".encode()
                except httpx.RequestError as e:
                    logger.error(f"Upstream stream connection error: {e}")
                    error_data = json.dumps(
                        {
                            "error": {
                                "message": f"Upstream connection error: {e}",
                                "type": "connection_error",
                            }
                        }
                    )
                    yield f"data: {error_data}\n\n".encode()

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
        )

    async def list_models(self) -> dict[str, Any]:
        """List available models from upstream."""
        return await self._forward_request("GET", "/models")

    async def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any] | StreamingResponse:
        """Forward chat completion request."""
        stream = payload.get("stream", False)

        if stream:
            return await self.stream_request("POST", "/chat/completions", payload)
        return await self._forward_request("POST", "/chat/completions", payload)

    async def completions(self, payload: dict[str, Any]) -> dict[str, Any] | StreamingResponse:
        """Forward legacy completion request."""
        stream = payload.get("stream", False)

        if stream:
            return await self.stream_request("POST", "/completions", payload)
        return await self._forward_request("POST", "/completions", payload)

    async def embeddings(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Forward embedding request to embedding endpoint."""
        url = f"{self.embedding_url}/embeddings"

        # Apply model mapping or use default embedding model
        if "model" in payload:
            payload["model"] = self._map_model(payload["model"])
        elif self.settings.embedding_model_name:
            payload["model"] = self.settings.embedding_model_name

        timeout = self.settings.embedding_timeout

        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.request(
                    method="POST",
                    url=url,
                    headers=self.embedding_headers,
                    json=payload,
                )
                response.raise_for_status()
                return response.json()
            except httpx.TimeoutException as e:
                logger.error(f"Embedding endpoint timeout: {e}")
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Embedding endpoint timeout",
                ) from e
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"Embedding endpoint error: {e.response.status_code} - {e.response.text}"
                )
                raise HTTPException(
                    status_code=e.response.status_code,
                    detail=e.response.text,
                ) from e
            except httpx.RequestError as e:
                logger.error(f"Embedding endpoint connection error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Embedding endpoint connection error: {e}",
                ) from e

    async def extract(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        disable_reasoning: bool = True,
    ) -> str:
        """
        Call extraction LLM for entity/relationship/fact extraction.

        This is an internal method used by Forma's RAG pipeline,
        not exposed as a public endpoint.

        Args:
            messages: Chat messages for extraction prompt
            model: Model name (defaults to settings.extractor_model_name)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (low for deterministic extraction)
            disable_reasoning: Disable reasoning/thinking output for faster response

        Returns:
            Extraction result as string
        """
        url = f"{self.extractor_url}/chat/completions"
        model_name = model or self.settings.extractor_model_name

        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Disable reasoning/thinking for faster extraction
        if disable_reasoning:
            # Try common parameters for disabling reasoning
            payload["reasoning_effort"] = "none"
            payload["enable_thinking"] = False

        async with httpx.AsyncClient(timeout=self.settings.extractor_timeout) as client:
            try:
                response = await client.request(
                    method="POST",
                    url=url,
                    headers=self.extractor_headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except httpx.TimeoutException as e:
                logger.error(f"Extraction endpoint timeout: {e}")
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Extraction endpoint timeout",
                ) from e
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"Extraction endpoint error: {e.response.status_code} - {e.response.text}"
                )
                raise HTTPException(
                    status_code=e.response.status_code,
                    detail=e.response.text,
                ) from e
            except httpx.RequestError as e:
                logger.error(f"Extraction endpoint connection error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Extraction endpoint connection error: {e}",
                ) from e
