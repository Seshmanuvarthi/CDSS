"""Pydantic schemas for inter-agent communication and clinical contracts."""

import re
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field, AliasChoices, field_validator, model_validator


class Symptom(BaseModel):
    name: str = Field(
        default="",
        validation_alias=AliasChoices("name", "symptom", "finding", "symptom_name"),
        description="Name of the symptom"
    )
    onset: Optional[str] = Field(default=None, description="Onset timing if known")
    duration: Optional[str] = Field(default=None, description="Duration of symptom")
    severity: Optional[str] = Field(default="moderate", description="Mild, moderate, severe")
    is_negated: bool = Field(
        default=False,
        validation_alias=AliasChoices("is_negated", "negated", "is_denied", "denied", "absent"),
        description="True if patient specifically denies symptom"
    )

    @model_validator(mode="before")
    @classmethod
    def check_negation(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Check negation flags across various LLM output shapes
            is_neg = (
                bool(data.get("is_negated")) or
                bool(data.get("negated")) or
                bool(data.get("is_denied")) or
                bool(data.get("denied")) or
                bool(data.get("absent"))
            )
            status = str(data.get("status", "")).lower()
            severity_raw = data.get("severity")
            severity_str = str(severity_raw).lower() if severity_raw is not None else ""

            if status in ("denied", "negated", "absent", "false") or severity_str in ("denied", "negated", "absent", "null", "none"):
                if status in ("denied", "negated", "absent") or severity_str in ("denied", "negated", "absent"):
                    is_neg = True

            name = str(data.get("name", "")).strip()
            if re.match(r"^(?:no|denies|negative for|without)\s+", name, re.IGNORECASE):
                is_neg = True
                name = re.sub(r"^(?:no|denies|negative for|without)\s+", "", name, flags=re.IGNORECASE).strip()
                data["name"] = name

            data["is_negated"] = is_neg
            if is_neg:
                data["severity"] = "denied"
            elif not severity_raw or severity_str in ("null", "none", "unspecified", ""):
                data["severity"] = "moderate"

        return data


class LabResult(BaseModel):
    test_name: str = Field(default="", validation_alias=AliasChoices("test_name", "test"))
    value: Any = Field(default="", validation_alias=AliasChoices("value", "result"))
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    flag: Optional[str] = None  # e.g., 'high', 'low', 'normal'

    @model_validator(mode="before")
    @classmethod
    def infer_flag(cls, data: Any) -> Any:
        if isinstance(data, dict):
            test = str(data.get("test_name") or data.get("test") or "").lower()
            val_str = str(data.get("value") or data.get("result") or "").lower()
            flag = data.get("flag")

            if not flag:
                if any(k in val_str for k in ("elevated", "high", "positive", "+")):
                    flag = "high"
                elif any(k in val_str for k in ("low", "decreased", "reduced")):
                    flag = "low"
                elif any(k in val_str for k in ("normal", "reference", "negative")):
                    flag = "normal"
                else:
                    # Parse numeric values for common diagnostic labs
                    num_match = re.search(r"[-+]?\d*\.?\d+", val_str)
                    if num_match:
                        num = float(num_match.group(0))
                        if "glucose" in test and num > 140:
                            flag = "high"
                        elif "ph" in test and num < 7.35:
                            flag = "low"
                        elif ("bicarbonate" in test or "hco3" in test) and num < 22:
                            flag = "low"
                        elif ("beta-hydroxybutyrate" in test or "ketone" in test) and num > 0.5:
                            flag = "high"
                        elif "creatinine" in test and num > 1.2:
                            flag = "high"
                        elif "potassium" in test and (num < 3.5 or num > 5.0):
                            flag = "high" if num > 5.0 else "low"

            data["flag"] = flag
        return data


class ClinicalNote(BaseModel):
    """Structured representation produced by Agent 1 (Clinical Text Clarifier)."""
    patient_age: Optional[Union[int, str]] = None
    patient_sex: Optional[str] = None
    chief_complaint: str = Field(default="", validation_alias=AliasChoices("chief_complaint", "complaint", "presenting_complaint"))
    symptoms: List[Symptom] = Field(
        default_factory=list,
        validation_alias=AliasChoices("symptoms", "explicit_symptoms", "presenting_symptoms", "findings", "clinical_findings")
    )
    anatomical_regions: List[str] = Field(default_factory=list)
    vital_signs: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("vital_signs", "vitals", "vital_sign_measurements")
    )
    lab_results: List[LabResult] = Field(
        default_factory=list,
        validation_alias=AliasChoices("lab_results", "laboratory_findings", "labs", "laboratory_results", "lab_findings", "key_labs")
    )
    medical_history: List[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("medical_history", "history", "past_medical_history", "risk_factors")
    )
    current_medications: List[str] = Field(default_factory=list)
    ecg_findings: Optional[str] = Field(default=None, validation_alias=AliasChoices("ecg_findings", "ecg", "electrocardiogram"))
    imaging_findings: Optional[str] = Field(default=None, validation_alias=AliasChoices("imaging_findings", "imaging", "radiology", "ct_scan", "cta", "ctpa"))
    ontology_mappings: Dict[str, Any] = Field(default_factory=dict, description="e.g. Concept -> ICD-10 or SNOMED")
    raw_summary: Optional[str] = Field(default="", description="Cleaned, standardized clinical summary")

    @field_validator("ecg_findings", "imaging_findings", mode="before")
    @classmethod
    def normalize_findings(cls, v):
        if isinstance(v, list):
            return ", ".join(str(item).strip() for item in v if item)
        return v

    @field_validator("lab_results", mode="before")
    @classmethod
    def normalize_lab_results(cls, v):
        if isinstance(v, dict):
            res = []
            for k, val in v.items():
                if isinstance(val, dict):
                    res.append({"test_name": k, **val})
                else:
                    res.append({"test_name": k, "value": str(val)})
            return res
        return v or []

    @field_validator("symptoms", mode="before")
    @classmethod
    def normalize_symptoms(cls, v):
        if isinstance(v, dict):
            res = []
            for k, val in v.items():
                if isinstance(val, dict):
                    res.append({"name": k, **val})
                else:
                    res.append({"name": k, "severity": str(val)})
            return res
        return v or []

    @field_validator("medical_history", mode="before")
    @classmethod
    def normalize_medical_history(cls, v):
        if isinstance(v, dict):
            return [f"{k}: {val}" for k, val in v.items()]
        if isinstance(v, str):
            return [v]
        return v or []

    @field_validator("vital_signs", mode="before")
    @classmethod
    def normalize_vital_signs(cls, v):
        if isinstance(v, dict):
            return {k: str(val) for k, val in v.items()}
        return v or {}

    @field_validator("raw_summary", mode="before")
    @classmethod
    def normalize_raw_summary(cls, v):
        return v or ""


class RetrievedPassage(BaseModel):
    doc_id: str
    text: str
    source: str
    score: float
    evidence_level: Optional[str] = "textbook/guideline"
    condition: Optional[str] = None


class DiagnosticHypothesis(BaseModel):
    condition_name: str
    icd10_code: Optional[str] = None
    relevance_score: float = Field(description="Initial score from RAG / Analyzer")
    supporting_evidence: List[str] = Field(default_factory=list)
    clinical_justification: str


class WebEvidence(BaseModel):
    condition_name: str
    query: str
    source_title: str
    source_url: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    evidence_type: str = Field(default="guideline", description="guideline, systematic-review, meta-analysis, case-report")
    summary: str
    reliability_score: float = Field(default=0.85, description="Evidence trust score")


class ConflictItem(BaseModel):
    topic: str
    rag_claim: str
    web_claim: str
    severity: str = "moderate"  # low, moderate, high
    resolution: Optional[str] = None


class ClinicalFusionReport(BaseModel):
    """Output from Agent 4 (Clinical Data Fusion)."""
    primary_diagnosis: str
    differential_diagnoses: List[DiagnosticHypothesis]
    confidence_score: float = Field(description="Calibrated confidence score between 0.0 and 1.0")
    detected_conflicts: List[ConflictItem] = Field(default_factory=list)
    explanation: str
    meets_confidence_threshold: bool = Field(description="True if confidence >= tau (0.65)")
    confidence_breakdown: Optional[Dict[str, Any]] = Field(default=None, description="Detailed weights, inputs, and intermediate calculations")


class FeedbackReport(BaseModel):
    """Output from Agent 5 (Adaptive Optimizer) when confidence < tau."""
    iteration: int
    low_confidence_reason: str
    recommended_keywords: List[str]
    expanded_differential: List[str]
    focal_points_to_investigate: List[str]
    escalate_to_physician: bool = False

