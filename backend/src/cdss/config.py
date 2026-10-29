"""Configuration management for CDSS."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "backend/.env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Base Paths
    project_root: Path = Path(__file__).resolve().parent.parent.parent.parent
    data_dir: Path = project_root / "data"
    artifacts_dir: Path = project_root / "artifacts"
    qdrant_path: Path = artifacts_dir / "qdrant"

    # API Keys
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    groq_api_key: str = Field(default="", validation_alias=AliasChoices("GROQ_API_KEY", "GROK_API_KEY"))
    openrouter_api_key: str = Field(default="", validation_alias="OPENROUTER_API_KEY")
    serper_api_key: str = Field(default="", validation_alias="SERPER_API_KEY")

    # Retrieval & RAG
    embedding_model: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    embedding_dim: int = 768
    qdrant_collection_name: str = "clinical_knowledge"
    chunk_size: int = 512
    chunk_overlap: int = 64
    rag_top_k: int = 5
    rag_similarity_threshold: float = 0.65

    # Orchestration & Adaptive Loop
    confidence_threshold: float = 0.65  # Paper's tau gate
    max_iterations: int = 3             # Max adaptive feedback iterations

    # Model Roles
    clarifier_model: str = "qwen-2.5-72b"
    rag_analyzer_model: str = "gemini-2.0-flash"
    evidence_scanner_model: str = "qwen-2.5-72b"
    fusion_model: str = "gemini-2.0-flash"
    optimizer_model: str = "gemini-2.0-flash"


settings = Settings()

