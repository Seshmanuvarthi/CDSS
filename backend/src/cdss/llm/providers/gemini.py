"""Google Gemini API provider implementation."""

import re
import json
import time
import logging
from typing import Optional
import httpx
from ..base import BaseLLMClient, LLMRequest, LLMResponse
from ...config import settings

logger = logging.getLogger(__name__)


class GeminiClient(BaseLLMClient):
    """Client for Google Gemini API."""

    _credits_depleted: bool = False

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.gemini_api_key

    @classmethod
    def is_depleted(cls) -> bool:
        return cls._credits_depleted

    def _failover(self, request: LLMRequest) -> LLMResponse:
        """Route to next available provider."""
        if settings.openrouter_api_key:
            from .openrouter import OpenRouterClient
            return OpenRouterClient().complete(request)
        if settings.groq_api_key:
            from .groq import GroqClient
            return GroqClient().complete(request)
        from ..registry import MockClinicalLLMClient
        return MockClinicalLLMClient().complete(request)

    def complete(self, request: LLMRequest) -> LLMResponse:
        if not self.api_key or GeminiClient._credits_depleted:
            return self._failover(request)

        model = request.model
        if "gemini" in model.lower() or model in {"mock-clinical-v1", "qwen-2.5-72b"}:
            model = "gemini-3.5-flash"
        if not model.startswith("models/"):
            model = f"models/{model}"

        url = f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent?key={self.api_key}"

        # Build contents
        contents = []
        for m in request.messages:
            role = "user" if m.role in {"user", "system"} else "model"
            contents.append({
                "role": role,
                "parts": [{"text": m.content}]
            })

        gen_config = {
            "temperature": request.temperature,
            "maxOutputTokens": min(request.max_tokens or 500, 500),
        }

        # Enable JSON mode if response_schema is provided or requested
        if request.response_schema is not None or any("json" in m.content.lower() for m in request.messages):
            gen_config["responseMimeType"] = "application/json"

        payload = {
            "contents": contents,
            "generationConfig": gen_config
        }

        start_time = time.time()
        data = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=30.0) as client:
                    res = client.post(url, json=payload)
                    if res.status_code == 402:
                        GeminiClient._credits_depleted = True
                        logger.warning(
                            "Gemini API returned 402 Payment Required (Google AI Studio credits depleted). "
                            "Failing over immediately to OpenRouter."
                        )
                        return self._failover(request)
                    if res.status_code == 429:
                        logger.warning(f"Gemini RPM Rate limit reached. Backing off for 2.5s (attempt {attempt+1}/3)...")
                        time.sleep(2.5)
                        continue
                    res.raise_for_status()
                    data = res.json()
                    break
            except Exception as e:
                if attempt == 2:
                    logger.warning(f"Gemini API call failed ({e}). Failing over to alternative provider.")
                    return self._failover(request)
                time.sleep(1.5)

        if not data:
            return self._failover(request)

        latency_ms = (time.time() - start_time) * 1000

        text_content = ""
        try:
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text_content = parts[0].get("text", "") if parts else ""
        except Exception as e:
            logger.error(f"Error parsing Gemini response: {e}")

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

        usage = data.get("usageMetadata", {})

        return LLMResponse(
            text=text_content,
            parsed=parsed,
            model=model,
            provider="gemini",
            input_tokens=usage.get("promptTokenCount", 0),
            output_tokens=usage.get("candidatesTokenCount", 0),
            latency_ms=latency_ms
        )
