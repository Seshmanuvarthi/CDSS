"""Quick seeding script to index gold clinical passages into local Qdrant collection."""

import logging
from .gold_corpus import load_gold_clinical_passages
from ..retrieval.index import QdrantIndexManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def seed_gold_passages():
    logger.info("Loading gold-standard clinical benchmark passages...")
    docs = load_gold_clinical_passages()
    logger.info(f"Loaded {len(docs)} gold clinical passages.")

    index_mgr = QdrantIndexManager()
    total = index_mgr.chunk_and_index_documents(docs, batch_size=8)
    logger.info(f"Successfully indexed {total} gold clinical chunks into Qdrant ({index_mgr.path}).")
    logger.info(f"Total points in collection now: {index_mgr.get_count()}.")


if __name__ == "__main__":
    seed_gold_passages()

