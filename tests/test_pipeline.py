"""Unit and integration tests for the CDSS pipeline."""

import pytest
from cdss.schemas import ClinicalNote, Symptom, DiagnosticHypothesis, ClinicalFusionReport
from cdss.retrieval.search import ClinicalRetriever
from cdss.orchestration.pipeline import CDSSPipeline


def test_clinical_note_schema():
    """Verify ClinicalNote Pydantic validation."""
    note = ClinicalNote(
        chief_complaint="Chest pain",
        symptoms=[
            Symptom(name="substernal chest pain", severity="severe", is_negated=False),
            Symptom(name="fever", is_negated=True)
        ],
        raw_summary="Patient reports acute substernal chest pain. Denies fever."
    )
    assert note.chief_complaint == "Chest pain"
    assert len(note.symptoms) == 2
    assert note.symptoms[1].is_negated is True


def test_medical_term_boosting():
    """Verify medical term boosting and negation handling."""
    retriever = ClinicalRetriever()

    # Positive symptom should be boosted
    pos_query = "Patient has severe shortness of breath"
    boosted_pos = retriever.boost_medical_terms(pos_query)
    assert "[Clinical finding: shortness of breath]" in boosted_pos

    # Negated symptom must NOT be boosted
    neg_query = "Patient denies chest pain"
    boosted_neg = retriever.boost_medical_terms(neg_query)
    assert "[Clinical finding: chest pain]" not in boosted_neg


def test_end_to_end_pipeline_mock():
    """Verify end-to-end 5-agent pipeline execution."""
    pipeline = CDSSPipeline()
    sample_vignette = (
        "A 54-year-old man presents with acute substernal chest pain radiating to his left arm. "
        "He is diaphoretic and dyspneic. ST elevations in leads II, III, aVF."
    )
    result = pipeline.run(sample_vignette)

    assert result["status"] in {"success", "abstained_for_physician_review"}
    assert "clinical_note" in result
    assert "iterations" in result
    assert len(result["iterations"]) >= 1

    final_report = result["final_report"]
    assert final_report is not None
    assert "primary_diagnosis" in final_report
    assert 0.0 <= final_report["confidence_score"] <= 1.0

