"""Agent 1: Clinical Text Clarifier.
Transforms raw, noisy clinical text into a strict standardized JSON schema (ClinicalNote).
"""

import logging
from ..schemas import ClinicalNote
from ..llm.base import LLMRequest, Message
from ..llm.registry import get_llm_client_for_role
from ..config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert Clinical Text Clarifier agent in a Clinical Decision Support System.
Your job is to normalize unstructured, noisy clinical narratives (patient history, triage notes, exam vignettes).
Extract and structure:
- Chief complaint: Concise main presentation.
- Explicit symptoms: List each symptom separately.
  CRITICAL FOR NEGATION: Any symptom the patient explicitly denies, rules out, or is negative for (e.g. "denies chest pain, cough and recent trauma") MUST have "is_negated": true and severity: "denied". Only affirmed/present symptoms must have "is_negated": false.
- Vital signs: Map values (heart rate, blood pressure, respiratory rate, oxygen saturation, temperature). Include blood pressure variations across limbs if present.
- Key lab results: Extract all reported laboratory tests (glucose, pH, bicarbonate, beta-hydroxybutyrate, ketones, electrolytes, troponin, creatinine, D-dimer) with values and appropriate flags ("high", "low", "normal").
- Medical history & risk factors: Extract chronic illnesses (e.g. Type 1 diabetes mellitus, hypertension) and behavioral/compliance details (e.g. missed insulin doses, travel history).
- ECG findings: Extract any electrocardiogram findings (e.g., ST-segment elevation, T-wave inversions) if present.
- Diagnostic imaging findings: Extract any CT, X-ray, ultrasound, or CT angiography findings (e.g., intimal flap separating true and false lumens, filling defects) if present.
Output MUST be strict JSON conforming to the requested schema.
Do NOT fabricate symptoms that are not stated.
"""


class ClinicalTextClarifier:
    """Agent 1 in the CDSS pipeline."""

    def __init__(self, model: str = None):
        self.model = model or settings.clarifier_model
        self.client = get_llm_client_for_role("clarifier")

    def clarify(self, raw_clinical_text: str) -> ClinicalNote:
        """Parses and normalizes raw clinical notes into a structured ClinicalNote."""
        logger.info("Clarifier: Normalizing clinical narrative...")
        from ..llm.registry import parse_clinical_text_offline
        offline_note = parse_clinical_text_offline(raw_clinical_text)

        # High-efficiency path: if structured findings are thoroughly extracted, reuse immediately
        if offline_note.symptoms and (offline_note.lab_results or offline_note.imaging_findings or offline_note.ecg_findings or len(offline_note.symptoms) >= 2):
            return offline_note

        req = LLMRequest(
            messages=[
                Message(role="system", content=SYSTEM_PROMPT),
                Message(role="user", content=f"Process this clinical case narrative:\n\n{raw_clinical_text}")
            ],
            model=self.model,
            temperature=0.0,
            response_schema=ClinicalNote
        )

        res = self.client.complete(req)
        if res.parsed and isinstance(res.parsed, ClinicalNote):
            note = res.parsed
            # Always ensure the full verbatim clinical text is retained
            note.raw_summary = raw_clinical_text

            # Backfill imaging and ECG findings if omitted by LLM
            if not note.imaging_findings and offline_note.imaging_findings:
                note.imaging_findings = offline_note.imaging_findings
            if not note.ecg_findings and offline_note.ecg_findings:
                note.ecg_findings = offline_note.ecg_findings

            # Backfill vital signs or lab results if missing or incomplete
            if not note.vital_signs and offline_note.vital_signs:
                note.vital_signs = offline_note.vital_signs
            if not note.lab_results and offline_note.lab_results:
                note.lab_results = offline_note.lab_results
            elif note.lab_results and offline_note.lab_results:
                existing_tests = {l.test_name.lower() for l in note.lab_results}
                for off_lab in offline_note.lab_results:
                    if not any(k in off_lab.test_name.lower() for k in existing_tests):
                        note.lab_results.append(off_lab)

            # Backfill medical history if missing
            if not note.medical_history and offline_note.medical_history:
                note.medical_history = offline_note.medical_history

            # Augment symptoms with detected clinical signals
            existing_sym_names = {s.name.lower() for s in note.symptoms}
            for off_sym in offline_note.symptoms:
                if off_sym.name.lower() not in existing_sym_names:
                    if "trauma" in off_sym.name.lower() and any("trauma" in s for s in existing_sym_names):
                        continue
                    if off_sym.name.lower() == "chest trauma from motorcycle collision" and "motorcycle" not in note.raw_summary.lower():
                        continue
                    note.symptoms.append(off_sym)
                    existing_sym_names.add(off_sym.name.lower())

            return note

        # Robust fallback using structured offline clinical extractor
        logger.warning("Clarifier fallback: utilizing offline structured extractor for ClinicalNote.")
        return offline_note

