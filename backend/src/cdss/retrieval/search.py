"""Customized Search Tool with Medical Term Boosting and 3-Level Fallback Strategy."""

import re
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models

from ..config import settings
from .embedder import BioBERTEmbedder
from .index import get_shared_qdrant_client
from ..schemas import RetrievedPassage

logger = logging.getLogger(__name__)

# Priority clinical terms to boost in vector representation per paper §II-B
HIGH_PRIORITY_CLINICAL_TERMS = [
    "chest pain", "chest pressure", "substernal", "angina", "coronary",
    "shortness of breath", "dyspnea", "syncope", "diaphoresis", "sweating",
    "palpitations", "hemoptysis", "jaundice", "hypotension", "tachycardia",
    "fever", "altered mental status", "cyanosis", "abdominal pain", "ataxia",
    "st-segment elevation", "st elevation", "troponin", "intimal flap",
    "aortic dissection", "pulmonary embolism", "radiating to left arm",
    "radiation to left arm", "calf swelling", "calf discomfort",
    "tearing chest pain", "tearing in character", "shoulder blades", "true and false lumen",
    "asymmetric radial pulses", "blood pressure asymmetry",
    "diabetic ketoacidosis", "ketoacidosis", "ketones", "hyperglycemia",
    "blood glucose", "kussmaul breathing", "kussmaul", "polydipsia", "polyuria",
    "beta-hydroxybutyrate", "insulin deficiency", "type 1 diabetes", "metabolic acidosis",
    "tension pneumothorax", "pneumothorax", "absent breath sounds", "reduced breath sounds",
    "tracheal deviation", "distended neck veins", "jugular venous distension",
    "absent lung sliding", "needle decompression", "tube thoracostomy", "obstructive shock",
    "crohn's disease", "crohn disease", "fecal calprotectin", "terminal ileum",
    "chronic diarrhea", "transmural", "cobblestoning", "skip lesions",
    "focal granulomatous", "granuloma", "ileitis", "weight loss",
    "ischemic stroke", "middle cerebral artery", "mca", "m1 occlusion", "large vessel occlusion",
    "aphasia", "facial droop", "arm weakness", "hemiparesis",
    "appendicitis", "right lower quadrant", "mcburney", "rebound tenderness", "guarding",
    "pancreatitis", "lipase", "amylase", "epigastric", "radiating to back",
    "meningitis", "nuchal rigidity", "neck stiffness", "kernig", "brudzinski",
    "cerebrospinal fluid", "csf", "pleocytosis", "lumbar puncture",
    "heart failure", "pulmonary edema", "bnp", "natriuretic peptide", "orthopnea",
    "paroxysmal nocturnal dyspnea", "pnd", "ejection fraction", "lvef", "crackles",
    "pulmonary congestion", "jugular venous pressure", "jvp", "pleural effusion", "cardiomegaly"
]


class ClinicalRetriever:
    """
    Implements the Paper's Contribution #2 (Figure 5):
    - Medical Term Boosting
    - Cosine Search against Qdrant
    - 3-Level Fallback (L1: Threshold drop, L2: Simplification, L3: Query reconstruction)
    - Clinical Relevance Re-ranking (respecting negation and affirmed pathognomonic findings)
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        path: Optional[str] = None
    ):
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.path = path or str(settings.qdrant_path)
        self.client = get_shared_qdrant_client(self.path)
        self.embedder = BioBERTEmbedder()

    def boost_medical_terms(self, query: str) -> str:
        """
        Boosts critical clinical terms by selective frequency weighting,
        while strictly respecting negation (e.g. 'no chest pain' must not boost 'chest pain').
        """
        boosted = query
        lower_query = query.lower()

        for term in HIGH_PRIORITY_CLINICAL_TERMS:
            # Check if term is present and NOT negated
            negated_pattern = rf"\b(no|denies|without|ruled out)\s+([a-z\s]{{0,15}})?{term}\b"
            if re.search(negated_pattern, lower_query):
                continue  # Skip boosting negated terms

            if term in lower_query:
                # Append high-priority emphasis token
                boosted += f" [Clinical finding: {term}]"

        return boosted

    def search(
        self,
        query: str,
        top_k: int = 5,
        base_threshold: float = 0.50,
        denied_symptoms: Optional[List[str]] = None,
        affirmed_symptoms: Optional[List[str]] = None
    ) -> List[RetrievedPassage]:
        """
        Executes search with 3-level fallback as specified in Figure 5 of the paper,
        coupled with clinical relevance re-ranking to penalize negated symptoms
        and prioritize confirmed pathognomonic findings.
        """
        boosted_query = self.boost_medical_terms(query)
        vector = self.embedder.embed_query(boosted_query)

        # Primary Search: Retrieve candidate pool
        candidate_k = max(top_k * 12, 60)
        results = self._qdrant_search(vector, candidate_k, threshold=base_threshold)

        if len(results) < 2:
            logger.info(f"Primary search retrieved {len(results)} hits. Initiating Fallback Level 1 (Threshold drop)...")
            results = self._qdrant_search(vector, candidate_k, threshold=0.40)

        if len(results) < 2:
            logger.info("Fallback Level 1 insufficient. Initiating Fallback Level 2 (Linguistic simplification)...")
            simplified_query = self._simplify_query(query)
            l2_vector = self.embedder.embed_query(simplified_query)
            results = self._qdrant_search(l2_vector, candidate_k, threshold=0.40)

        if len(results) < 1:
            logger.info("Fallback Level 2 insufficient. Initiating Fallback Level 3 (Semantic reconstruction)...")
            l3_query = self._reconstruct_semantic_query(query)
            l3_vector = self.embedder.embed_query(l3_query)
            results = self._qdrant_search(l3_vector, candidate_k, threshold=0.35)

        # Apply Clinical Relevance Re-ranking if denied or affirmed symptoms provided
        reranked = self._rerank_passages(results, query, denied_symptoms, affirmed_symptoms)
        return reranked[:top_k]

    def _rerank_passages(
        self,
        passages: List[RetrievedPassage],
        query: str,
        denied_symptoms: Optional[List[str]],
        affirmed_symptoms: Optional[List[str]]
    ) -> List[RetrievedPassage]:
        if not passages:
            return []

        denied = [d.lower().strip() for d in (denied_symptoms or []) if d.strip()]
        if any(d in ["fever", "cough"] for d in denied):
            denied.extend(["chills", "itching"])

        affirmed = [a.lower().strip() for a in (affirmed_symptoms or []) if a.strip()]

        scored_passages = []
        for p in passages:
            score = p.score
            text_lower = p.text.lower()

            # Evidential weight for validated clinical guidelines with named condition per paper §II-B
            if p.condition and not p.condition.isdigit() and "dux-tecblic" not in (p.source or "").lower():
                score += 0.08
            elif (p.source or "").lower() == "dux-tecblic/symptom-disease-dataset" or (p.condition and p.condition.isdigit()):
                score -= 0.15

            # Slight penalty if passage centers on a symptom the patient explicitly denied
            for d in denied:
                if re.search(rf"\b{re.escape(d)}\b", text_lower):
                    score -= 0.08
                    break

            # Slight reinforcement for affirmed clinical features
            for a in affirmed:
                if a in text_lower:
                    score += 0.04

            p.score = round(max(0.10, min(0.99, score)), 3)
            scored_passages.append(p)

        scored_passages.sort(key=lambda x: x.score, reverse=True)
        return scored_passages

    def _qdrant_search(
        self,
        vector: List[float],
        top_k: int,
        threshold: float
    ) -> List[RetrievedPassage]:
        """Queries local Qdrant collection."""
        try:
            res = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=top_k,
                score_threshold=threshold
            )
            passages = []
            for hit in res.points:
                payload = hit.payload or {}
                passages.append(
                    RetrievedPassage(
                        doc_id=payload.get("doc_id", "unknown"),
                        text=payload.get("text", ""),
                        source=payload.get("source", "vector_db"),
                        score=float(hit.score),
                        evidence_level=payload.get("evidence_level", "textbook/guideline"),
                        condition=payload.get("condition") or None
                    )
                )
            return passages
        except Exception as e:
            logger.error(f"Qdrant search error: {e}")
            return []

    def _simplify_query(self, query: str) -> str:
        """Removes narrative padding (e.g. 'Patient presents with', 'A 45-year-old male with')."""
        cleaned = re.sub(r"(?i)\b(a\s+\d+\s*year\s*old|\bpatient\b|\bpresents with\b|\bhistory of\b|\bcomplaining of\b)", "", query)
        return cleaned.strip()

    def _reconstruct_semantic_query(self, query: str) -> str:
        """Reconstructs query based on primary symptoms for diagnostic clustering."""
        tokens = [t for t in re.findall(r"\b[a-zA-Z]{4,}\b", query.lower()) if t not in {"patient", "history", "years", "female", "male"}]
        return " ".join(tokens[:10])

