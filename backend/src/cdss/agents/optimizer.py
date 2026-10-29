"""Agent 5: Adaptive Optimizer.
Manages the feedback loop when confidence is below tau (0.65).
Identifies ambiguity, expands queries dynamically, and directs re-investigation.
"""

import re
import json
import logging
from typing import List
from ..schemas import ClinicalFusionReport, FeedbackReport, ClinicalNote
from ..config import settings
from ..llm.base import LLMRequest, Message
from ..llm.registry import get_llm_client_for_role

logger = logging.getLogger(__name__)

OPTIMIZER_PROMPT = """You are an Adaptive Optimizer Agent in a Clinical Decision Support System.
The system's diagnostic confidence for this patient case is currently LOW (below tau = 0.65).

Your job is to objectively analyze WHY confidence is low and direct targeted re-investigation:
1. Check whether the current primary candidate is CONTRADICTED or UNSUPPORTED by objective findings.
2. If the current candidate is contradicted or ungrounded, DO NOT force-fit it; prioritize the uncontradicted differentials.
3. Recommend specific pathognomonic clinical keywords (imaging modalities, definitive biomarkers, specific guidelines) to resolve ambiguity.

Return JSON adhering to the FeedbackReport schema:
{
  "low_confidence_reason": "...",
  "recommended_keywords": ["keyword1", "keyword2"],
  "expanded_differential": ["condition1", "condition2"],
  "focal_points_to_investigate": ["point1", "point2"]
}
"""


class AdaptiveOptimizer:
    """Agent 5 in the CDSS pipeline."""

    def __init__(self, model: str = None):
        self.model = model or settings.optimizer_model
        self.client = get_llm_client_for_role("optimizer")

    def generate_feedback(
        self,
        report: ClinicalFusionReport,
        note: ClinicalNote,
        iteration: int
    ) -> FeedbackReport:
        """Generates targeted query adaptation and differential expansion."""
        logger.info(f"Adaptive Optimizer: Generating feedback for iteration {iteration} (Confidence: {report.confidence_score:.2f})...")

        if iteration >= settings.max_iterations:
            logger.info("Maximum iterations reached. Escalating to physician review (Abstention).")
            return FeedbackReport(
                iteration=iteration,
                low_confidence_reason="Exhausted maximum adaptive iterations without clearing confidence threshold.",
                recommended_keywords=[],
                expanded_differential=[h.condition_name for h in report.differential_diagnoses],
                focal_points_to_investigate=["Manual physician review required", "Consider specialist consultation"],
                escalate_to_physician=True
            )

        conflicts_str = "\n".join([f"- {c.topic}: {c.rag_claim} vs {c.web_claim}" for c in report.detected_conflicts])
        differentials_str = ", ".join([f"{h.condition_name} ({h.relevance_score:.2f})" for h in report.differential_diagnoses])

        req = LLMRequest(
            messages=[
                Message(role="system", content=OPTIMIZER_PROMPT),
                Message(
                    role="user",
                    content=(
                        f"ITERATION: {iteration} of {settings.max_iterations}\n"
                        f"CURRENT REPORT:\nPrimary Candidate: {report.primary_diagnosis} (Confidence: {report.confidence_score:.2f})\n"
                        f"DIFFERENTIAL CANDIDATES IN EVALUATION: {differentials_str}\n"
                        f"CONFLICTS / EVIDENCE MISMATCHES:\n{conflicts_str or 'Low evidence density / ambiguous symptom match'}\n\n"
                        f"PATIENT CONTEXT (VITALS, LABS, IMAGING):\n{note.raw_summary}"
                    )
                )
            ],
            model=self.model,
            temperature=0.0
        )

        res = self.client.complete(req)

        # Dynamic fallback analysis based on detected conflicts and gaps
        c_topics = [c.topic for c in report.detected_conflicts]
        cands = report.differential_diagnoses

        if "Diagnostic Equivocality" in c_topics and len(cands) >= 2:
            h1, h2 = cands[0], cands[1]
            dyn_reason = f"Diagnostic equivocality between {h1.condition_name} and {h2.condition_name} (difference < 0.05). Targeted investigation required to differentiate."
            dyn_keywords = [
                f"{h1.condition_name} pathognomonic criteria",
                f"{h2.condition_name} exclusion criteria",
                "confirmatory diagnostic imaging",
                "specific laboratory biomarkers"
            ]
            dyn_diff = [h1.condition_name, h2.condition_name]
            dyn_focal = [
                f"Evaluate definitive diagnostic guidelines for {h1.condition_name}",
                f"Assess discriminative clinical features for {h2.condition_name}"
            ]
        elif any(t in c_topics for t in ("Biomarker Discordance", "Imaging Contradiction", "Competing Pathophysiology Discrepancy")):
            alt_diag = cands[1].condition_name if len(cands) > 1 else "alternative pathology"
            dyn_reason = f"Primary candidate '{report.primary_diagnosis}' conflicts with objective diagnostic findings. Pivoting retrieval toward uncontradicted differential: {alt_diag}."
            dyn_keywords = [
                f"{alt_diag} clinical guidelines",
                "objective diagnostic criteria",
                "pathognomonic imaging findings"
            ]
            dyn_diff = [h.condition_name for h in cands[1:]] or [alt_diag]
            dyn_focal = [
                "Re-evaluate differential against objective laboratory and imaging results",
                "Prioritize conditions uncontradicted by patient findings"
            ]
        else:
            abnormal_labs = [f"{l.test_name} {l.value}" for l in note.lab_results if l.flag in ("high", "low")]
            dyn_reason = f"Low evidence density or discordance for {report.primary_diagnosis} (Confidence {report.confidence_score:.2f} < 0.65 threshold)."
            dyn_keywords = [
                f"{note.chief_complaint} diagnostic workup"
            ] + [f"{l.test_name} diagnostic criteria" for l in note.lab_results if l.flag in ("high", "low")][:2] + [
                "clinical practice guidelines"
            ]
            dyn_diff = [h.condition_name for h in cands]
            dyn_focal = [
                "Expand diagnostic search query to capture unaddressed patient findings",
                "Review secondary clinical features and abnormal diagnostic tests"
            ]

        # Progressive query refinement across iterations
        if iteration == 2:
            dyn_keywords.extend(["confirmatory imaging", "specific serology"])
            dyn_focal.append("Prepare secondary specialized diagnostic pathways")

        try:
            raw_t = res.text.strip()
            if "```" in raw_t:
                raw_t = re.sub(r"^```(?:json)?\s*", "", raw_t)
                raw_t = re.sub(r"\s*```$", "", raw_t)
            match = re.search(r"(\{.*\})", raw_t, re.DOTALL)
            parsed = json.loads(match.group(1)) if match else json.loads(raw_t)

            low_reason = parsed.get("low_confidence_reason") or dyn_reason
            rec_keywords = parsed.get("recommended_keywords") or dyn_keywords
            exp_diff = parsed.get("expanded_differential") or dyn_diff
            focal_pts = parsed.get("focal_points_to_investigate") or dyn_focal

            # Ensure keywords are non-empty and specific
            if not rec_keywords or rec_keywords == ["differential diagnosis"]:
                rec_keywords = dyn_keywords

            return FeedbackReport(
                iteration=iteration,
                low_confidence_reason=low_reason,
                recommended_keywords=rec_keywords,
                expanded_differential=exp_diff,
                focal_points_to_investigate=focal_pts,
                escalate_to_physician=False
            )
        except Exception as e:
            logger.info(f"Using dynamic gap analysis for feedback report: {e}")
            return FeedbackReport(
                iteration=iteration,
                low_confidence_reason=dyn_reason,
                recommended_keywords=dyn_keywords,
                expanded_differential=dyn_diff,
                focal_points_to_investigate=dyn_focal,
                escalate_to_physician=False
            )
