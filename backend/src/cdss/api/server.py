"""FastAPI REST and Streaming server for the CDSS pipeline."""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..orchestration.pipeline import CDSSPipeline
from ..retrieval.index import QdrantIndexManager
from ..eval.datasets import (
    load_medqa_50,
    load_pubmedqa_50,
    get_random_dataset_cases,
    format_case_prompt
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Clinical Decision Support System (CDSS) API",
    description="Multi-Agent LLM-based CDSS integrating Biomedical RAG and Web Intelligence",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_PUBLIC_DIR = Path(__file__).resolve().parents[4] / "frontend" / "public"

if FRONTEND_PUBLIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_PUBLIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def serve_index():
    index_file = FRONTEND_PUBLIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(
            index_file,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return {"message": "CDSS API is online. Frontend index.html not found."}


pipeline = CDSSPipeline()


class ClinicalCaseInput(BaseModel):
    clinical_text: str


class BenchmarkRunRequest(BaseModel):
    dataset: Optional[str] = "medqa"  # "medqa", "pubmedqa", "medbullets"
    cases_count: Optional[int] = 10
    random_sample: Optional[bool] = True


@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    """Returns system status and Qdrant index stats."""
    index_mgr = QdrantIndexManager()
    count = index_mgr.get_count()
    return {
        "status": "healthy",
        "service": "CDSS-Multi-Agent",
        "qdrant_indexed_points": count,
        "qdrant_path": str(index_mgr.path)
    }


@app.get("/api/benchmark/results")
def get_benchmark_results() -> Dict[str, Any]:
    """Returns dynamic benchmark results placeholder."""
    return {
        "status": "cleared",
        "message": "Static pre-computed figures removed. Use /api/benchmark/run for live empirical testing."
    }


@app.get("/api/cases/presets")
def get_preset_cases() -> List[Dict[str, Any]]:
    """Returns curated clinical cases from the gold-standard benchmark datasets."""
    return [
        {
            "id": "CASE_STEMI",
            "title": "Acute Inferior Myocardial Infarction",
            "category": "Cardiology / Emergency",
            "clinical_text": (
                "Case ID: MEDQA_001\n"
                "Clinical Case Vignette:\n"
                "A 54-year-old male presents to the ED with acute retrosternal crushing chest pain radiating to his left arm and jaw. "
                "He is diaphoretic, short of breath, with ST elevations in leads II, III, and aVF. Elevated cardiac troponin I.\n\n"
                "Candidate Diagnostic Options:\n"
                "A: Myocardial Infarction\n"
                "B: Aortic Dissection\n"
                "C: Gastroesophageal Reflux\n"
                "D: Costochondritis"
            ),
            "gold_diagnosis": "Myocardial Infarction"
        },
        {
            "id": "CASE_PE",
            "title": "Acute Pulmonary Embolism (Post-Flight)",
            "category": "Pulmonology / Critical Care",
            "clinical_text": (
                "Case ID: MEDQA_002\n"
                "Clinical Case Vignette:\n"
                "A 32-year-old female presents with acute pleuritic chest pain and dyspnea after a 14-hour flight. "
                "Tachycardia (118 bpm), hypoxia (89% O2 sat), and unilateral right calf swelling and tenderness.\n\n"
                "Candidate Diagnostic Options:\n"
                "A: Pulmonary Embolism\n"
                "B: Pneumothorax\n"
                "C: Pneumonia\n"
                "D: Anxiety Disorder"
            ),
            "gold_diagnosis": "Pulmonary Embolism"
        },
        {
            "id": "CASE_DKA",
            "title": "Diabetic Ketoacidosis (DKA)",
            "category": "Endocrinology / Acute",
            "clinical_text": (
                "Case ID: MEDQA_003\n"
                "Clinical Case Vignette:\n"
                "A 19-year-old male with type 1 diabetes presents with vomiting, diffuse abdominal pain, confusion, Kussmaul respirations, fruity breath odor. "
                "Blood glucose 480 mg/dL, arterial pH 7.18, positive ketones.\n\n"
                "Candidate Diagnostic Options:\n"
                "A: Diabetic Ketoacidosis\n"
                "B: Hyperosmolar Hyperglycemic State\n"
                "C: Acute Pancreatitis\n"
                "D: Gastroenteritis"
            ),
            "gold_diagnosis": "Diabetic Ketoacidosis"
        },
        {
            "id": "CASE_CROHN",
            "title": "Crohn Disease (Transmural Cobblestoning)",
            "category": "Gastroenterology",
            "clinical_text": (
                "Case ID: MEDQA_020\n"
                "Clinical Case Vignette:\n"
                "A 24-year-old male presents with recurrent abdominal pain in RLQ, chronic non-bloody diarrhea, perianal fistulas, aphthous oral ulcers, "
                "skip lesions on colonoscopy, and transmural cobblestoning.\n\n"
                "Candidate Diagnostic Options:\n"
                "A: Crohn Disease\n"
                "B: Ulcerative Colitis\n"
                "C: Celiac Disease\n"
                "D: Irritable Bowel Syndrome"
            ),
            "gold_diagnosis": "Crohn Disease"
        },
        {
            "id": "CASE_AORTIC",
            "title": "Acute Aortic Dissection",
            "category": "Vascular Surgery",
            "clinical_text": (
                "Case ID: MEDQA_009\n"
                "Clinical Case Vignette:\n"
                "A 62-year-old male with hypertension presents with sudden catastrophic tearing chest pain radiating between shoulder blades. "
                "Blood pressure asymmetry between arms (185/100 right arm, 130/70 left arm).\n\n"
                "Candidate Diagnostic Options:\n"
                "A: Aortic Dissection\n"
                "B: Myocardial Infarction\n"
                "C: Esophageal Perforation\n"
                "D: Pneumothorax"
            ),
            "gold_diagnosis": "Aortic Dissection"
        },
        {
            "id": "CASE_PNEUMOTHORAX",
            "title": "Tension Pneumothorax",
            "category": "Thoracic Surgery",
            "clinical_text": (
                "Case ID: MEDQA_010\n"
                "Clinical Case Vignette:\n"
                "A tall 22-year-old male presents with sudden unilateral sharp pleuritic chest pain, dyspnea, hyperresonance to percussion, "
                "and absent breath sounds over the right hemithorax with tracheal deviation.\n\n"
                "Candidate Diagnostic Options:\n"
                "A: Tension Pneumothorax\n"
                "B: Pneumonia\n"
                "C: Pulmonary Embolism\n"
                "D: Asthma Exacerbation"
            ),
            "gold_diagnosis": "Tension Pneumothorax"
        },
        {
            "id": "CASE_STROKE",
            "title": "Acute Ischemic Stroke (Left M1 MCA Territory)",
            "category": "Neurology / Emergent Reperfusion",
            "clinical_text": (
                "Case ID: CDSS-STROKE-007\n"
                "Clinical Case Vignette:\n"
                "A 67-year-old man develops sudden expressive aphasia, right facial droop, and right arm weakness 45 minutes before arrival. "
                "No preceding headache, seizure, or trauma. BP 174/96 mmHg, HR 92 bpm, RR 18/min, SpO2 97%, temperature 36.7°C. "
                "Point-of-care glucose 112 mg/dL. Non-contrast head CT shows no intracranial hemorrhage. "
                "CT angiography shows an occlusion of the left M1 middle cerebral artery.\n\n"
                "Candidate Diagnostic Options:\n"
                "A: Acute Ischemic Stroke\n"
                "B: Intracranial Hemorrhage\n"
                "C: Transient Ischemic Attack\n"
                "D: Todd's Paralysis"
            ),
            "gold_diagnosis": "Acute Ischemic Stroke"
        }
    ]


@app.post("/api/diagnose")
def diagnose_case(input_data: ClinicalCaseInput) -> Dict[str, Any]:
    """Executes the full 5-agent CDSS adaptive loop for a clinical narrative."""
    if not input_data.clinical_text.strip():
        raise HTTPException(status_code=400, detail="Clinical narrative cannot be empty.")

    try:
        t0 = time.time()
        result = pipeline.run(input_data.clinical_text)
        result["execution_time_ms"] = round((time.time() - t0) * 1000, 1)
        return result
    except Exception as e:
        logger.error(f"Error executing diagnosis pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/benchmark/run")
def run_benchmark_test(req: BenchmarkRunRequest) -> Dict[str, Any]:
    """Executes live verification tests over selected benchmark cases with full logging."""
    dataset_name = (req.dataset or "medqa").strip()
    cases_count = max(1, min(50, req.cases_count or 10))
    random_sample = req.random_sample if req.random_sample is not None else True

    logger.info("=" * 70)
    logger.info(f"STARTING BENCHMARK RUN: Dataset='{dataset_name.upper()}', Count={cases_count}, Randomized={random_sample}")
    logger.info("=" * 70)

    cases = get_random_dataset_cases(dataset_name=dataset_name, count=cases_count, random_sample=random_sample)
    
    t0 = time.time()
    results = []
    logs: List[str] = []
    correct_count = 0

    init_msg = f"Initialized evaluation for dataset '{dataset_name.upper()}' with {len(cases)} cases (randomized: {random_sample})."
    logger.info(init_msg)
    logs.append(f"[{time.strftime('%H:%M:%S')}] {init_msg}")

    for idx, item in enumerate(cases):
        case_id = item.get("id", f"CASE_{idx+1:03d}")
        gold = item.get("gold_diagnosis", "").strip()
        q_snippet = item.get("question", "")[:80] + "..." if len(item.get("question", "")) > 80 else item.get("question", "")

        case_start_msg = f"[{idx+1}/{len(cases)}] Executing Case '{case_id}': {q_snippet}"
        logger.info(case_start_msg)
        logs.append(f"[{time.strftime('%H:%M:%S')}] {case_start_msg}")

        prompt = format_case_prompt(item, dataset_name)
        
        t_case0 = time.time()
        res = pipeline.run(prompt)
        case_duration = round(time.time() - t_case0, 2)

        final_rep = res.get("final_report") or {}
        pred = final_rep.get("primary_diagnosis", "").strip()
        conf = final_rep.get("confidence_score", 0.0)
        iters = len(res.get("iterations", [1]))

        # Evaluate match
        norm_ds = dataset_name.lower().replace("-", "").replace("_", "")
        if "pubmed" in norm_ds:
            gold_dec = gold.lower()
            pred_lower = pred.lower()
            expl_lower = (final_rep.get("explanation") or "").lower()
            is_match = False
            if gold_dec in ("yes", "no", "maybe"):
                if pred_lower.startswith(gold_dec) or f" {gold_dec}" in pred_lower or f": {gold_dec}" in pred_lower:
                    is_match = True
                elif gold_dec == "yes" and ("option a" in pred_lower or pred_lower == "a"):
                    is_match = True
                elif gold_dec == "no" and ("option b" in pred_lower or pred_lower == "b"):
                    is_match = True
                elif gold_dec == "maybe" and ("option c" in pred_lower or pred_lower == "c"):
                    is_match = True
                elif gold_dec in pred_lower:
                    is_match = True
                elif any(phrase in expl_lower for phrase in [f"answer is {gold_dec}", f"decision is {gold_dec}", f"result: {gold_dec}", f"supports {gold_dec}"]):
                    is_match = True
        else:
            is_match = (
                gold.lower() in pred.lower() or
                pred.lower() in gold.lower() or
                any(k.lower() == pred.lower() and v.lower() == gold.lower() for k, v in item.get("options", {}).items()) or
                any(f"option {k.lower()}" in pred.lower() and v.lower() == gold.lower() for k, v in item.get("options", {}).items())
            )

        if is_match:
            correct_count += 1
            verdict_str = "✓ PASS"
        else:
            verdict_str = "✗ FAIL"

        running_acc = round((correct_count / (idx + 1)) * 100, 1)
        res_msg = (
            f"[{idx+1}/{len(cases)}] '{case_id}' Verdict: {verdict_str} | "
            f"Gold: '{gold}' | Pred: '{pred}' | Conf: {conf:.3f} | Iterations: {iters} | "
            f"Time: {case_duration}s | Running Accuracy: {correct_count}/{idx+1} ({running_acc}%)"
        )
        logger.info(res_msg)
        logs.append(f"[{time.strftime('%H:%M:%S')}] {res_msg}")

        results.append({
            "id": case_id,
            "question": item.get("question", ""),
            "gold_diagnosis": gold,
            "predicted_diagnosis": pred,
            "confidence_score": conf,
            "iterations_count": iters,
            "matched": is_match,
            "duration_s": case_duration,
            "explanation": final_rep.get("explanation", "")
        })

    elapsed_s = round(time.time() - t0, 2)
    accuracy_pct = round((correct_count / len(cases)) * 100, 1) if cases else 0.0

    final_summary_msg = (
        f"BENCHMARK COMPLETED: Dataset='{dataset_name.upper()}', "
        f"Score={correct_count}/{len(cases)} ({accuracy_pct}%), Elapsed={elapsed_s}s"
    )
    logger.info("=" * 70)
    logger.info(final_summary_msg)
    logger.info("=" * 70)
    logs.append(f"[{time.strftime('%H:%M:%S')}] {final_summary_msg}")

    return {
        "dataset": dataset_name,
        "tested_cases": len(cases),
        "correct_count": correct_count,
        "accuracy_pct": accuracy_pct,
        "elapsed_seconds": elapsed_s,
        "results": results,
        "logs": logs
    }
