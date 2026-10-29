"""Base protocol and abstractions for LLM providers."""

from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel
import abc


class Message(BaseModel):
    role: str  # 'system', 'user', 'assistant'
    content: str


class LLMRequest(BaseModel):
    messages: List[Message]
    model: str
    temperature: float = 0.0
    max_tokens: int = 2048
    response_schema: Optional[Type[BaseModel]] = None


class LLMResponse(BaseModel):
    text: str
    parsed: Optional[Any] = None
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


class BaseLLMClient(abc.ABC):
    """Abstract client for LLM providers."""

    @abc.abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Synchronous completion."""
        pass

