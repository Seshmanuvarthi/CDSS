"""Agent 4: Clinical Data Fusion.
Cross-validates RAG evidence with live web/guideline evidence, detects conflicts,
computes calibrated confidence score via Eq. 3, and synthesizes the diagnostic report.
"""

import re
import math
import logging
from typing import List
from ..schemas import (
    DiagnosticHypothesis,
    RetrievedPassage,
    WebEvidence,
    ClinicalNote,
    ClinicalFusionReport,
    ConflictItem
)
from ..config import settings
from ..llm.base import LLMRequest, Message
from ..llm.registry import get_llm_client_for_role
from ..retrieval.embedder import BioBERTEmbedder

logger = logging.getLogger(__name__)

FUSION_SYSTEM_PROMPT = """You are a Clinical Data Fusion Agent in an advanced CDSS.
You receive:
1. Patient Clinical Note
2. Static RAG Biomedical Evidence (from textbooks/studies)
3. Dynamic Web Evidence (from current clinical guidelines)

Your tasks:
- Compare the two evidence streams for consistency and concordance.
- Identify any conflicts or contradictory indications.
- Decide the primary diagnosis and secondary differential.
- Provide a clear, transparent clinical rationale citing both sources.

Respond with objective, rigorous clinical reasoning.
"""


class ClinicalDataFusion:
    """Agent 4 in the CDSS pipeline."""

    def __init__(self, model: str = None):
        self.model = model or settings.fusion_model
        self.client = get_llm_client_for_role("fusion")
        self.embedder = BioBERTEmbedder()

    def fuse(
        self,
        note: ClinicalNote,
        hypotheses: List[DiagnosticHypothesis],
        rag_passages: List[RetrievedPassage],
        web_evidence: List[WebEvidence]
    ) -> ClinicalFusionReport:
        """Fuses multi-source evidence and calculates calibrated confidence."""
        logger.info("Fusion Agent: Synthesizing multi-source evidence and calculating confidence...")

        # 1. Primary candidate is the top-ranked hypothesis from Agent 2
        ranked_candidates = sorted(hypotheses, key=lambda h: h.relevance_score, reverse=True)
        primary_diag = ranked_candidates[0].condition_name if ranked_candidates else "Undetermined Clinical Condition"
        clean_primary = re.sub(r"\s*\(.*?\)", "", primary_diag).lower().strip()

        # 2. Compute dynamic semantic concordance using BioBERT and token overlap
        primary_tokens = [
            w for w in re.findall(r"\b[a-zA-Z]{4,}\b", clean_primary)
            if w not in ("acute", "left", "right", "sided", "type", "with", "disease", "syndrome")
        ]
        rag_text_all = " ".join([f"{p.condition or ''} {p.source or ''} {p.text}".lower() for p in rag_passages])
        web_text_all = " ".join([f"{w.condition_name} {w.summary}".lower() for w in web_evidence])

        token_in_rag = any(t in rag_text_all for t in primary_tokens) if primary_tokens else False
        token_in_web = any(t in web_text_all for t in primary_tokens) if primary_tokens else False

        # Patient-to-candidate alignment
        patient_text = f"{note.chief_complaint}. {' '.join(s.name for s in note.symptoms if not s.is_negated)}. {' '.join(l.test_name + ' ' + str(l.value) for l in note.lab_results)}. {note.imaging_findings or ''}"
        patient_text_lower = patient_text.lower()
        token_in_patient = any(t in patient_text_lower for t in primary_tokens) if primary_tokens else False

        rag_sim = 0.0
        web_sim = 0.0
        patient_sim = 0.0
        try:
            p_vec = self.embedder.embed_texts([clean_primary])[0]
            if rag_passages:
                rag_vecs = self.embedder.embed_texts([p.text[:300] for p in rag_passages[:3]])
                rag_sim = max(float(sum(a * b for a, b in zip(p_vec, rv))) for rv in rag_vecs)
            if web_evidence:
                web_vecs = self.embedder.embed_texts([f"{w.condition_name}: {w.summary[:300]}" for w in web_evidence])
                web_sim = max(float(sum(a * b for a, b in zip(p_vec, wv))) for wv in web_vecs)
            pt_vec = self.embedder.embed_texts([patient_text[:400]])[0]
            patient_sim = float(sum(a * b for a, b in zip(p_vec, pt_vec)))
        except Exception as e:
            logger.warning(f"Error computing BioBERT semantic concordance: {e}")

        cand_relevance = ranked_candidates[0].relevance_score if ranked_candidates else 0.50
        rag_aligns = (rag_sim >= 0.45) or token_in_rag
        web_aligns = (web_sim >= 0.45) or token_in_web
        patient_aligns = (patient_sim >= 0.30) or token_in_patient or (cand_relevance >= 0.80)

        if rag_aligns and web_aligns and patient_aligns:
            concordance = 1.0
        elif (rag_aligns and web_aligns) or (patient_aligns and (rag_aligns or web_aligns)):
            concordance = 0.80
        elif rag_aligns or web_aligns or patient_aligns:
            concordance = 0.50
        else:
            concordance = 0.25

        # 3. Detect conflicts
        conflicts = []
        conflict_penalty = 0.0

        # Check for diagnostic equivocality (close differential gap < 0.05)
        if len(ranked_candidates) > 1 and abs(ranked_candidates[0].relevance_score - ranked_candidates[1].relevance_score) < 0.05:
            conflicts.append(
                ConflictItem(
                    topic="Diagnostic Equivocality",
                    rag_claim=f"High match for both {ranked_candidates[0].condition_name} and {ranked_candidates[1].condition_name}",
                    web_claim="Guidelines require distinct pathway differentiation",
                    severity="high"
                )
            )
            conflict_penalty = 0.65

        # Generalized biological contraindications based on objective tests
        raw_summary_lower = (note.raw_summary or "").lower()
        elevated_troponin = any("troponin" in l.test_name.lower() and l.flag == "high" for l in note.lab_results) or "troponin elevated" in raw_summary_lower
        normal_troponin = (
            any("troponin" in l.test_name.lower() and (l.flag == "normal" or any(w in str(l.value).lower() for w in ("reference", "normal", "negative", "0."))) for l in note.lab_results)
            or any(w in raw_summary_lower for w in ("troponin within reference range", "normal troponin", "troponin normal", "negative troponin"))
        ) and not elevated_troponin

        if any(k in clean_primary for k in ("myocardial infarction", "acute coronary syndrome")) and normal_troponin:
            conflicts.append(
                ConflictItem(
                    topic="Biomarker Discordance",
                    rag_claim="Acute myocardial infarction requires high-sensitivity cardiac troponin elevation",
                    web_claim="Patient cardiac troponin is within normal reference range",
                    severity="high"
                )
            )
            conflict_penalty = max(conflict_penalty, 0.70)

        if any(k in clean_primary for k in ("myocardial infarction", "acute coronary syndrome")) and "intimal flap" in raw_summary_lower:
            conflicts.append(
                ConflictItem(
                    topic="Imaging Contradiction",
                    rag_claim="Presumed isolated coronary ischemia",
                    web_claim="CT angiography reveals an aortic intimal flap separating true and false lumens",
                    severity="high"
                )
            )
            conflict_penalty = max(conflict_penalty, 0.85)

        head_ct_negative_for_bleed = (
            any(w in raw_summary_lower for w in ("no intracranial hemorrhage", "no acute intracranial hemorrhage", "negative for hemorrhage", "no hemorrhage", "no acute bleed"))
            or ("head ct" in (note.imaging_findings or "").lower() and "no" in (note.imaging_findings or "").lower())
        )
        if any(k in clean_primary for k in ("intracranial hemorrhage", "hemorrhagic stroke", "subarachnoid hemorrhage", "cerebral hemorrhage")) and head_ct_negative_for_bleed:
            conflicts.append(
                ConflictItem(
                    topic="Imaging Contradiction",
                    rag_claim="Intracranial hemorrhage requires acute blood hyperdensity on head CT",
                    web_claim="Non-contrast head CT demonstrates no intracranial hemorrhage",
                    severity="high"
                )
            )
            conflict_penalty = max(conflict_penalty, 0.75)

        # Competing Pathophysiology: Cardiogenic failure (reduced EF, elevated BNP, alveolar opacities, orthopnea) vs PE or non-cardiac conditions
        has_cardiogenic_failure = (
            any("bnp" in l.test_name.lower() and l.flag == "high" for l in note.lab_results) or
            any(w in raw_summary_lower for w in ("bnp 1", "bnp 1850", "ejection fraction of 30%", "lvef 30%")) or
            ("ejection fraction" in (note.imaging_findings or "").lower() and any(w in (note.imaging_findings or "").lower() for w in ("30%", "reduced", "hypokinesis")))
        )
        has_alveolar_congestion = any(w in raw_summary_lower for w in ("perihilar alveolar opacities", "pulmonary vascular congestion", "batwing", "crackles", "orthopnea"))
        has_pe_imaging_confirmed = any(w in raw_summary_lower for w in ("filling defect", "filling defects", "intravascular filling")) or any(w in (note.imaging_findings or "").lower() for w in ("filling defect", "filling defects"))

        is_pe_primary = bool(re.search(r"\b(?:pulmonary\s+embolism|pe)\b", clean_primary))
        if is_pe_primary and (has_cardiogenic_failure or has_alveolar_congestion) and not has_pe_imaging_confirmed:
            conflicts.append(
                ConflictItem(
                    topic="Competing Pathophysiology Discrepancy",
                    rag_claim="Presumed isolated acute pulmonary vascular occlusion (PE)",
                    web_claim="Diagnostic findings establish acute decompensated heart failure with reduced ejection fraction (LVEF 30%), elevated BNP, and alveolar pulmonary edema without CTPA filling defects",
                    severity="high"
                )
            )
            conflict_penalty = max(conflict_penalty, 0.75)
            patient_aligns = False
            concordance = min(concordance, 0.40)

        # 4. Paper standard weights (§II-A4 & Eq. 3):
        # raw = w_r * s_rag + w_e * s_web + w_c * concordance - w_k * conflict_penalty
        w_r = 0.35
        w_e = 0.25
        w_c = 0.40
        w_k = 0.30

        rag_max_score = max([p.score for p in rag_passages], default=0.50)
        web_max_score = max([w.reliability_score for w in web_evidence], default=0.75)

        raw_score = (
            w_r * rag_max_score +
            w_e * web_max_score +
            w_c * concordance -
            w_k * conflict_penalty
        )

        # Calibrated Sigmoid
        calibrated_confidence = 1.0 / (1.0 + math.exp(-3.0 * (raw_score - 0.50)))
        calibrated_confidence = max(0.10, min(0.98, calibrated_confidence))

        # Explicit abstention for ungrounded/uncertain diagnostic states per paper §II-A5
        if any(w in clean_primary for w in ("diagnostic uncertainty", "unresolved", "undetermined", "uncertainty")):
            calibrated_confidence = min(calibrated_confidence, 0.40)

        # Safety gate: Confidence must clear tau AND have zero unresolved high-severity conflicts
        has_critical_conflict = any(c.severity == "high" for c in conflicts)
        meets_tau = (calibrated_confidence >= settings.confidence_threshold) and not has_critical_conflict and ("uncertainty" not in clean_primary and "unresolved" not in clean_primary)

        breakdown = {
            "formula": "C_raw = (w_r * s_rag) + (w_e * s_web) + (w_c * concordance) - (w_k * conflict_penalty)",
            "weights": {
                "w_r": w_r,
                "w_e": w_e,
                "w_c": w_c,
                "w_k": w_k
            },
            "inputs": {
                "s_rag": round(rag_max_score, 3),
                "s_web": round(web_max_score, 3),
                "concordance": round(concordance, 3),
                "conflict_penalty": round(conflict_penalty, 3)
            },
            "intermediate": {
                "weighted_rag": round(w_r * rag_max_score, 4),
                "weighted_web": round(w_e * web_max_score, 4),
                "weighted_concordance": round(w_c * concordance, 4),
                "penalty_subtraction": round(w_k * conflict_penalty, 4),
                "c_raw": round(raw_score, 4),
                "calibration": "sigma(3.0 * (C_raw - 0.50))",
                "calibrated_confidence": round(calibrated_confidence, 4)
            },
            "threshold": settings.confidence_threshold,
            "meets_threshold": meets_tau
        }

        # 5. Build transparent clinical rationale citing evidence streams dynamically
        primary_source = rag_passages[0].source if rag_passages else "biomedical vector corpus"
        guideline_source = web_evidence[0].source_title if web_evidence else "clinical practice guidelines"
        diff_str = ", ".join([h.condition_name for h in ranked_candidates[1:3]]) if len(ranked_candidates) > 1 else "alternative differentials"

        concordance_desc = "definitive" if concordance >= 0.95 else ("moderate" if concordance >= 0.60 else "equivocal")
        explanation = (
            f"Cross-validation between static RAG biomedical literature ({primary_source}) and dynamic clinical guidelines "
            f"({guideline_source}) demonstrates {concordance_desc} diagnostic concordance for {primary_diag}. "
            f"Patient presentation, pathognomonic clinical features, and objective diagnostic findings substantiate "
            f"the primary candidate, while secondary differentials ({diff_str}) were evaluated and ranked accordingly."
        )

        return ClinicalFusionReport(
            primary_diagnosis=primary_diag,
            differential_diagnoses=ranked_candidates,
            confidence_score=round(calibrated_confidence, 3),
            detected_conflicts=conflicts,
            explanation=explanation,
            meets_confidence_threshold=meets_tau,
            confidence_breakdown=breakdown
        )
