"""Loader for Hugging Face symptom-disease clinical datasets."""

from typing import List, Dict, Any
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)


def load_symptom_disease_corpus() -> List[Dict[str, Any]]:
    """
    Loads dux-tecblic/symptom-disease-dataset from Hugging Face.
    Formats each record into clean clinical narrative passages for RAG indexing.
    """
    logger.info("Loading dux-tecblic/symptom-disease-dataset from Hugging Face...")
    documents = []

    try:
        ds = load_dataset("dux-tecblic/symptom-disease-dataset", split="train")
        for i, row in enumerate(ds):
            # Fields typically contain disease name, symptoms, and description / precautions
            disease = row.get("label", row.get("disease", row.get("Disease", "Unknown Condition")))
            symptoms = row.get("text", row.get("symptoms", row.get("Symptoms", "")))
            precautions = row.get("precautions", row.get("treatment", ""))

            content = f"Condition: {disease}\nKey Symptoms & Presentation: {symptoms}"
            if precautions:
                content += f"\nRecommended Management / Precautions: {precautions}"

            documents.append({
                "doc_id": f"symptom_ds_{i}",
                "text": content,
                "source": "dux-tecblic/symptom-disease-dataset",
                "kind": "disease-symptom-pair",
                "condition": disease,
                "evidence_level": "guideline/textbook"
            })
        logger.info(f"Loaded {len(documents)} symptom-disease records.")
    except Exception as e:
        logger.warning(f"Could not load primary symptom-disease dataset: {e}. Falling back to curated clinical cases.")

    # Always augment with gold-standard clinical emergency passages
    from .gold_corpus import load_gold_clinical_passages
    gold_docs = load_gold_clinical_passages()
    documents.extend(gold_docs)
    logger.info(f"Augmented corpus with {len(gold_docs)} gold-standard clinical emergency passages.")

    return documents

