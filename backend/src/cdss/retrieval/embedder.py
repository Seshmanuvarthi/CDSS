"""BioBERT embedding generator for clinical text chunks."""

import os
import logging
from typing import List, Union
import numpy as np

# Prevent network checks on cached HuggingFace models
if not os.environ.get("HF_HUB_ONLINE"):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

logger = logging.getLogger(__name__)

_CACHED_MODELS = {}


class BioBERTEmbedder:
    """
    Wraps the paper's specified embedding model:
    BioBERT-mnli-snli-scinli-scitail-mednli-stsb (768-dim).
    Automatically detects Apple Silicon (MPS) or CPU.
    """

    def __init__(self, model_name: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"):
        self.model_name = model_name
        self._device = None

    def _load_model(self):
        global _CACHED_MODELS
        if self.model_name in _CACHED_MODELS:
            self._model = _CACHED_MODELS[self.model_name]
            return

        import torch
        from sentence_transformers import SentenceTransformer

        if torch.backends.mps.is_available():
            self._device = "mps"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        logger.info(f"Loading embedding model '{self.model_name}' on device '{self._device}'...")
        try:
            model = SentenceTransformer(self.model_name, device=self._device, model_kwargs={"local_files_only": True})
        except Exception:
            try:
                model = SentenceTransformer(self.model_name, device=self._device)
            except Exception as e:
                logger.warning(f"Could not load {self.model_name}: {e}. Falling back to 'all-MiniLM-L6-v2'.")
                model = SentenceTransformer("all-MiniLM-L6-v2", device=self._device)

        _CACHED_MODELS[self.model_name] = model
        self._model = model

    def embed_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> List[List[float]]:
        """Compute dense vector embeddings for a list of clinical texts."""
        self._load_model()
        if not texts:
            return []
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Compute dense vector embedding for a single clinical query."""
        self._load_model()
        embedding = self._model.encode(query, normalize_embeddings=True)
        return embedding.tolist()

