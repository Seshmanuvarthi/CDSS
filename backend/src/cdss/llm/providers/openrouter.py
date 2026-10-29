"""OpenRouter API provider implementation."""

import re
import json
import time
import logging
from typing import Optional
import httpx
from ..base import BaseLLMClient, LLMRequest, LLMResponse
from ...config import settings

logger = logging.getLogger(__name__)


class OpenRouterClient(BaseLLMClient):
    """Client for OpenRouter multi-provider API."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    MODEL_ALIASES = {
        "gemini-2.0-flash": "google/gemini-2.5-flash",
        "gemini-3.5-flash": "google/gemini-2.5-flash",
        "gemini-2.5-flash": "google/gemini-2.5-flash",
        "qwen-2.5-72b": "meta-llama/llama-3.3-70b-instruct",
        "llama-3.3-70b-versatile": "meta-llama/llama-3.3-70b-instruct",
        "mock-clinical-v1": "google/gemini-2.5-flash",
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.openrouter_api_key

    def complete(self, request: LLMRequest) -> LLMResponse:
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/google-deepmind/cdss",
            "X-Title": "CDSS Clinical Decision Support",
            "Content-Type": "application/json"
        }

        # Resolve target model
        target_model = self.MODEL_ALIASES.get(request.model, request.model)
        if not ("/" in target_model):
            if "llama" in target_model.lower():
                target_model = "meta-llama/llama-3.3-70b-instruct"
            else:
                target_model = "google/gemini-2.5-flash"

        # Format messages
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        # Token budget: Keep max_tokens <= 1000 to ensure complete structured JSON output without truncation
        token_budget = min(request.max_tokens, 1000) if request.max_tokens and request.max_tokens > 0 else 800

        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": token_budget,
        }

        # Enforce JSON mode and inject schema contract if response_schema is provided
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
                with httpx.Client(timeout=35.0) as client:
                    res = client.post(self.BASE_URL, headers=headers, json=payload)
                    if res.status_code == 429:
                        logger.warning(f"OpenRouter Rate limit reached. Backing off for 2.0s (attempt {attempt+1}/3)...")
                        time.sleep(2.0)
                        continue
                    if res.status_code >= 400:
                        err_body = res.text
                        logger.warning(f"OpenRouter returned HTTP {res.status_code}: {err_body}")
                    res.raise_for_status()
                    data = res.json()
                    break
            except Exception as e:
                if attempt == 2:
                    logger.warning(f"OpenRouter API call failed ({e}). Failing over to Groq provider.")
                    if settings.groq_api_key:
                        from .groq import GroqClient
                        return GroqClient().complete(request)
                    from ..registry import MockClinicalLLMClient
                    return MockClinicalLLMClient().complete(request)
                time.sleep(1.5)

        if not data:
            if settings.groq_api_key:
                from .groq import GroqClient
                return GroqClient().complete(request)
            from ..registry import MockClinicalLLMClient
            return MockClinicalLLMClient().complete(request)

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
            provider="openrouter",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency_ms
        )
