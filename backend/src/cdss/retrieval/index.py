"""Qdrant vector database indexing and management."""

import os
import uuid
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import settings
from .embedder import BioBERTEmbedder

logger = logging.getLogger(__name__)


_SHARED_QDRANT_CLIENTS: Dict[str, QdrantClient] = {}


def get_shared_qdrant_client(path: Optional[str] = None) -> QdrantClient:
    """Returns a shared process-level QdrantClient instance for local disk paths."""
    p = path or str(settings.qdrant_path)
    if p == ":memory:":
        if p not in _SHARED_QDRANT_CLIENTS:
            _SHARED_QDRANT_CLIENTS[p] = QdrantClient(location=":memory:")
        return _SHARED_QDRANT_CLIENTS[p]
    os.makedirs(p, exist_ok=True)
    if p not in _SHARED_QDRANT_CLIENTS:
        try:
            _SHARED_QDRANT_CLIENTS[p] = QdrantClient(path=p)
        except Exception as exc:
            logger.warning("Could not lock Qdrant storage at %s (%s). Falling back to in-memory QdrantClient.", p, exc)
            _SHARED_QDRANT_CLIENTS[p] = QdrantClient(location=":memory:")
    return _SHARED_QDRANT_CLIENTS[p]


class QdrantIndexManager:
    """Manages local Qdrant vector database collection and indexing."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        path: Optional[str] = None,
        dim: int = 768
    ):
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.dim = dim
        self.path = path or str(settings.qdrant_path)
        self.client = get_shared_qdrant_client(self.path)
        self.embedder = BioBERTEmbedder()
        self._ensure_collection()

    def _ensure_collection(self):
        """Create Qdrant collection with Cosine distance if it doesn't already exist."""
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            logger.info(f"Creating Qdrant collection '{self.collection_name}' with dim {self.dim}...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.dim,
                    distance=models.Distance.COSINE
                )
            )

    def chunk_and_index_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 64
    ) -> int:
        """
        Chunks clinical documents (chunk_size=512, overlap=64) per paper §II-B,
        embeds using BioBERT, and uploads to Qdrant.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " "]
        )

        all_chunks = []
        for doc in documents:
            text = doc.get("text", "")
            if not text.strip():
                continue
            chunks = splitter.split_text(text)
            for chunk_idx, chunk_text in enumerate(chunks):
                if len(chunk_text.strip()) < 30:
                    continue  # Filter out trivial segments
                all_chunks.append({
                    "text": chunk_text,
                    "doc_id": doc.get("doc_id", "unknown"),
                    "source": doc.get("source", "unknown"),
                    "condition": doc.get("condition", ""),
                    "kind": doc.get("kind", "clinical-chunk"),
                    "evidence_level": doc.get("evidence_level", "textbook/guideline"),
                    "chunk_idx": chunk_idx
                })

        logger.info(f"Total chunks generated: {len(all_chunks)}. Embedding and uploading to Qdrant...")

        total_indexed = 0
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            texts = [item["text"] for item in batch]
            vectors = self.embedder.embed_texts(texts)

            points = []
            for j, item in enumerate(batch):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{item['doc_id']}_{item['chunk_idx']}_{i+j}"))
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vectors[j],
                        payload=item
                    )
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            total_indexed += len(points)
            if (i // batch_size) % 5 == 0:
                logger.info(f"Indexed {total_indexed}/{len(all_chunks)} chunks into Qdrant.")

        logger.info(f"Indexing complete! Successfully loaded {total_indexed} points.")
        return total_indexed

    def get_count(self) -> int:
        """Returns the number of points in the collection."""
        res = self.client.get_collection(self.collection_name)
        return res.points_count

