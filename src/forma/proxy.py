"""OpenAI-compatible proxy implementation with multi-upstream support."""

import json
import logging
from typing import Any, cast

import httpx
from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse

from forma.config import Settings
from forma.upstream_manager import UpstreamConfig

logger = logging.getLogger(__name__)


class OpenAIProxy:
    """Proxy client for OpenAI-compatible APIs with multi-upstream support."""

    def __init__(self, settings: Settings, upstream_manager) -> None:
        """Initialize the proxy.

        Args:
            settings: Application settings
            upstream_manager: UpstreamManager for model-based routing
        """
        self.settings = settings
        self._upstream_manager = upstream_manager

        # Extraction LLM endpoint configuration (separate from main upstreams)
        self._extractor_url = (
            settings.extractor_base_url.rstrip("/")
            if settings.extractor_base_url
            else None  # Will fail if no extractor URL configured and no upstream for extractor model
        )
        self._extractor_headers = {"Content-Type": "application/json"}
        if settings.extractor_api_key:
            self._extractor_headers["Authorization"] = f"Bearer {settings.extractor_api_key}"

        # Reusable HTTP client (connection pooled)
        self._client = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=50),
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    def reload_upstreams(self) -> None:
        """Reload upstream configurations from database."""
        self._upstream_manager.reload()
        logger.info("Upstreams reloaded")

    def _get_upstream(self, model: str) -> UpstreamConfig | None:
        """Get the upstream configuration for a model.

        Returns None if no upstream is configured for this model.
        """
        return self._upstream_manager.get_upstream_for_model(model)

    def _validate_upstream(self, upstream: UpstreamConfig | None, model: str) -> UpstreamConfig:
        """Validate that an upstream exists for the model, raising error if not."""
        if upstream is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No upstream configured for model '{model}'. Please add an upstream with this model name.",
            )
        return upstream

    async def _forward_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        upstream: UpstreamConfig | None = None,
    ) -> dict[str, Any]:
        """Forward a non-streaming request to upstream API.

        Args:
            method: HTTP method
            path: API path
            payload: Request payload
            upstream: Specific upstream to use (if None, determined by model in payload)

        Raises:
            HTTPException: If no upstream is configured for the model
        """
        # Determine upstream based on model
        model = payload.get("model", "") if payload else ""
        if upstream is None:
            upstream = self._get_upstream(model)
            self._validate_upstream(upstream, model)

        # Replace model in payload with upstream_model
        if payload and "model" in payload and upstream.upstream_model:
            payload["model"] = upstream.upstream_model

        url = f"{upstream.base_url}{path}"
        headers = {"Content-Type": "application/json"}
        if upstream.api_key:
            headers["Authorization"] = f"Bearer {upstream.api_key}"

        logger.debug(
            f"Forwarding request to {upstream.name}: {url} (model: {upstream.upstream_model})"
        )

        try:
            response = await self._client.request(
                method=method,
                url=url,
                headers=headers,
                json=payload,
                timeout=upstream.timeout,
            )
            response.raise_for_status()
            return cast(dict[str, Any], response.json())
        except httpx.TimeoutException as e:
            logger.error(f"Upstream timeout ({upstream.name}): {e}")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Upstream API timeout ({upstream.name})",
            ) from e
        except httpx.HTTPStatusError as e:
            logger.error(
                f"Upstream error ({upstream.name}): {e.response.status_code} - {e.response.text}"
            )
            raise HTTPException(
                status_code=e.response.status_code,
                detail=e.response.text,
            ) from e
        except httpx.RequestError as e:
            logger.error(f"Upstream connection error ({upstream.name}): {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Upstream connection error ({upstream.name}): {e}",
            ) from e

    async def stream_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any],
        upstream: UpstreamConfig | None = None,
    ) -> StreamingResponse:
        """Forward a streaming request to upstream API.

        Args:
            method: HTTP method
            path: API path
            payload: Request payload
            upstream: Specific upstream to use (if None, determined by model in payload)

        Raises:
            HTTPException: If no upstream is configured for the model
        """
        # Determine upstream based on model
        model = payload.get("model", "")
        if upstream is None:
            upstream = self._get_upstream(model)
            self._validate_upstream(upstream, model)

        # Replace model in payload with upstream_model
        if "model" in payload and upstream.upstream_model:
            payload["model"] = upstream.upstream_model

        url = f"{upstream.base_url}{path}"
        headers = {"Content-Type": "application/json"}
        if upstream.api_key:
            headers["Authorization"] = f"Bearer {upstream.api_key}"

        logger.debug(
            f"Streaming request to {upstream.name}: {url} (model: {upstream.upstream_model})"
        )

        async def stream_generator() -> Any:
            try:
                async with self._client.stream(
                    method=method,
                    url=url,
                    headers=headers,
                    json=payload,
                    timeout=upstream.timeout,
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        yield chunk
            except httpx.TimeoutException as e:
                logger.error(f"Upstream stream timeout ({upstream.name}): {e}")
                error_data = json.dumps(
                    {
                        "error": {
                            "message": f"Upstream API timeout ({upstream.name})",
                            "type": "timeout_error",
                        }
                    }
                )
                yield f"data: {error_data}\n\n".encode()
            except httpx.HTTPStatusError as e:
                logger.error(f"Upstream stream error ({upstream.name}): {e.response.status_code}")
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
                logger.error(f"Upstream stream connection error ({upstream.name}): {e}")
                error_data = json.dumps(
                    {
                        "error": {
                            "message": f"Upstream connection error ({upstream.name}): {e}",
                            "type": "connection_error",
                        }
                    }
                )
                yield f"data: {error_data}\n\n".encode()

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
        )

    async def list_models(self, model: str = None) -> dict[str, Any]:
        """List available models from an upstream.

        Args:
            model: Optional model name to determine which upstream to use.
                   If not provided, returns empty list if no upstreams configured.
        """
        if model:
            upstream = self._get_upstream(model)
            if upstream:
                return await self._forward_request("GET", "/models", upstream=upstream)

        # No model specified - return empty models list
        # (Could also iterate all upstreams and merge, but that's complex)
        return {"object": "list", "data": []}

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

    async def extract(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
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

        Returns:
            Extraction result as string

        Raises:
            HTTPException: If extraction endpoint is not configured
        """
        model_name = model or self.settings.extractor_model_name

        # Use dedicated extractor URL if configured
        if self._extractor_url:
            url = f"{self._extractor_url}/chat/completions"
            headers = self._extractor_headers
            timeout = self.settings.extractor_timeout
        else:
            # Try to find upstream for the extractor model name
            upstream = self._get_upstream(model_name)
            if upstream is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Extraction endpoint not configured and no upstream for model '{model_name}'. "
                    f"Please set EXTRACTOR_BASE_URL or add an upstream named '{model_name}'.",
                )
            url = f"{upstream.base_url}/chat/completions"
            headers = {"Content-Type": "application/json"}
            if upstream.api_key:
                headers["Authorization"] = f"Bearer {upstream.api_key}"
            timeout = self.settings.extractor_timeout

        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Add reasoning parameters only if configured (not supported by all APIs)
        if self.settings.extractor_send_reasoning_params:
            payload["reasoning_effort"] = "none"
            payload["enable_thinking"] = False

        try:
            response = await self._client.request(
                method="POST",
                url=url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = cast(dict[str, Any], response.json())
            return cast(str, data["choices"][0]["message"]["content"])
        except httpx.TimeoutException as e:
            logger.error(f"Extraction endpoint timeout: {e}")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Extraction endpoint timeout",
            ) from e
        except httpx.HTTPStatusError as e:
            logger.error(f"Extraction endpoint error: {e.response.status_code} - {e.response.text}")
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
