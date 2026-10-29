"""Full 5-Agent Adaptive CDSS Orchestrator Pipeline."""

import time
import logging
from typing import Dict, Any, List
from ..schemas import ClinicalFusionReport, FeedbackReport
from ..agents.clarifier import ClinicalTextClarifier
from ..agents.rag_analyzer import SymptomRAGAnalyzer
from ..agents.evidence_scanner import EvidenceBasedScanner
from ..agents.fusion import ClinicalDataFusion
from ..agents.optimizer import AdaptiveOptimizer
from ..config import settings

logger = logging.getLogger(__name__)


class CDSSPipeline:
    """
    End-to-End Orchestrator executing the 5-Agent Adaptive CDSS flow.
    """

    def __init__(self):
        self.clarifier = ClinicalTextClarifier()
        self.rag_analyzer = SymptomRAGAnalyzer()
        self.evidence_scanner = EvidenceBasedScanner()
        self.fusion = ClinicalDataFusion()
        self.optimizer = AdaptiveOptimizer()

    def run(self, raw_clinical_text: str) -> Dict[str, Any]:
        """
        Executes the full clinical reasoning pipeline with adaptive feedback.
        """
        start_time = time.time()
        trace = {
            "input_text": raw_clinical_text,
            "iterations": [],
            "final_report": None,
            "status": "in_progress"
        }

        # Step 1: Clarifier
        logger.info("=== STEP 1: Clinical Text Clarification ===")
        note = self.clarifier.clarify(raw_clinical_text)
        trace["clinical_note"] = note.model_dump()

        current_feedback_keywords: List[str] = []
        final_report: ClinicalFusionReport = None

        # Adaptive Loop (Max 3 iterations per paper §II-A5)
        for iteration in range(1, settings.max_iterations + 1):
            logger.info(f"\n=== ITERATION {iteration}/{settings.max_iterations} ===")
            iter_trace = {
                "iteration": iteration,
                "feedback_keywords_applied": list(current_feedback_keywords)
            }

            # Step 2: RAG Analyzer
            hypotheses, passages = self.rag_analyzer.analyze(
                note=note,
                feedback_keywords=current_feedback_keywords
            )
            iter_trace["hypotheses"] = [h.model_dump() for h in hypotheses]
            iter_trace["passages_count"] = len(passages)
            iter_trace["passages"] = [p.model_dump() for p in passages[:5]]

            # Step 3: Evidence Scanner
            web_evidence = self.evidence_scanner.scan_evidence(
                hypotheses=hypotheses,
                note=note
            )
            iter_trace["web_evidence_count"] = len(web_evidence)
            iter_trace["web_evidence"] = [w.model_dump() for w in web_evidence]

            # Step 4: Clinical Data Fusion
            report = self.fusion.fuse(
                note=note,
                hypotheses=hypotheses,
                rag_passages=passages,
                web_evidence=web_evidence
            )
            iter_trace["report"] = report.model_dump()
            final_report = report

            logger.info(f"Iteration {iteration} Result: Primary = '{report.primary_diagnosis}', Confidence = {report.confidence_score:.3f}")

            # Check termination condition: confidence >= tau (0.65)
            if report.meets_confidence_threshold:
                logger.info(f"Confidence threshold satisfied ({report.confidence_score:.2f} >= {settings.confidence_threshold})! Halting loop.")
                iter_trace["status"] = "converged"
                trace["iterations"].append(iter_trace)
                trace["status"] = "success"
                break

            # If not converged and iterations remain, invoke Step 5 (Adaptive Optimizer)
            logger.info(f"Confidence {report.confidence_score:.2f} < {settings.confidence_threshold}. Invoking Adaptive Optimizer...")
            feedback = self.optimizer.generate_feedback(
                report=report,
                note=note,
                iteration=iteration
            )
            iter_trace["feedback"] = feedback.model_dump()
            iter_trace["status"] = "low_confidence_retrying"
            trace["iterations"].append(iter_trace)

            if feedback.escalate_to_physician:
                logger.warning("Feedback requested physician escalation (Abstention).")
                trace["status"] = "abstained_for_physician_review"
                break

            # Update keywords for next iteration
            current_feedback_keywords = feedback.recommended_keywords

        total_latency = (time.time() - start_time) * 1000
        trace["total_latency_ms"] = round(total_latency, 1)
        trace["final_report"] = final_report.model_dump() if final_report else None

        return trace

