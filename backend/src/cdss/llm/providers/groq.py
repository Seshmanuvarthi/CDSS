"""Groq API provider implementation."""

import re
import json
import time
import logging
from typing import Optional
import httpx
from ..base import BaseLLMClient, LLMRequest, LLMResponse
from ...config import settings

logger = logging.getLogger(__name__)


class GroqClient(BaseLLMClient):
    """Client for Groq fast inference API."""

    BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

    MODEL_ALIASES = {
        "gemini-2.0-flash": "qwen/qwen3.8-27b",
        "gemini-3.5-flash": "qwen/qwen3.8-27b",
        "qwen-2.5-72b": "qwen/qwen3.8-27b",
        "llama-3.3-70b-versatile": "qwen/qwen3.8-27b",
        "openai/gpt-oss-120b": "openai/gpt-oss-120b",
        "mock-clinical-v1": "qwen/qwen3.8-27b",
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.groq_api_key

    def _failover(self, request: LLMRequest) -> LLMResponse:
        if settings.openrouter_api_key:
            from .openrouter import OpenRouterClient
            return OpenRouterClient().complete(request)
        from ..registry import MockClinicalLLMClient
        return MockClinicalLLMClient().complete(request)

    def complete(self, request: LLMRequest) -> LLMResponse:
        if not self.api_key:
            return self._failover(request)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Resolve model to available Groq models
        target_model = self.MODEL_ALIASES.get(request.model, request.model)
        if target_model not in {"openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b", "allam-2-7b"}:
            target_model = "qwen/qwen3.8-27b"

        # Format messages
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        # Conserve token budget to avoid triggering Groq TPM rate limits
        token_budget = min(request.max_tokens, 350) if request.max_tokens and request.max_tokens > 0 else 300

        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": token_budget,
        }

        # If schema is provided, enforce JSON mode and inject schema contract into messages
        if request.response_schema is not None:
            schema_json = json.dumps(request.response_schema.model_json_schema())
            schema_instruction = f"\nYou MUST return ONLY valid JSON matching this schema:\n{schema_json}"
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] += schema_instruction
            else:
                messages.insert(0, {"role": "system", "content": f"You MUST return ONLY valid JSON matching this schema:\n{schema_instruction}"})
            payload["response_format"] = {"type": "json_object"}
        elif any("json" in m.get("content", "").lower() for m in messages):
            payload["response_format"] = {"type": "json_object"}

        start_time = time.time()
        data = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=30.0) as client:
                    res = client.post(self.BASE_URL, headers=headers, json=payload)
                    if res.status_code == 429:
                        reset_header = res.headers.get("x-ratelimit-reset-tokens", "")
                        sleep_time = 2.0
                        if "s" in reset_header:
                            try:
                                sleep_time = max(1.5, float(reset_header.replace("s", "")) + 0.5)
                            except ValueError:
                                pass
                        if sleep_time > 10.0 and settings.openrouter_api_key:
                            logger.warning(f"Groq TPM reset wait is {sleep_time:.1f}s. Routing immediately to OpenRouter.")
                            return self._failover(request)
                        logger.warning(f"Groq Rate limit reached. Backing off for {sleep_time:.2f}s (attempt {attempt+1}/3)...")
                        time.sleep(sleep_time)
                        continue
                    if res.status_code >= 400:
                        err_text = res.text
                        logger.warning(f"Groq API returned HTTP {res.status_code}: {err_text}")
                    res.raise_for_status()
                    data = res.json()
                    break
            except Exception as e:
                if attempt == 2:
                    logger.warning(f"Groq API call failed ({e}). Failing over to alternative provider.")
                    return self._failover(request)
                time.sleep(1.5)

        if not data:
            return self._failover(request)

        latency_ms = (time.time() - start_time) * 1000
        choice = data["choices"][0]
        text_content = choice["message"].get("content") or choice["message"].get("reasoning", "")
        usage = data.get("usage", {})

        parsed = None
        if request.response_schema is not None and text_content:
            try:
                clean_text = text_content.strip()
                if "```" in clean_text:
                    clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text)
                    clean_text = re.sub(r"\s*```$", "", clean_text)
                json_match = re.search(r"(\{.*\}|\[.*\])", clean_text, re.DOTALL)
                target_json_str = json_match.group(1) if json_match else clean_text
                parsed_json = json.loads(target_json_str)
                parsed = request.response_schema.model_validate(parsed_json)
            except Exception as e:
                logger.warning(f"Failed to parse response as {request.response_schema.__name__}: {e}")

        return LLMResponse(
            text=text_content,
            parsed=parsed,
            model=target_model,
            provider="groq",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency_ms
        )
