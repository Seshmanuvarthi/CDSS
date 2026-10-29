"""Evaluation harness to benchmark CDSS against medical QA cases (MedQA)."""

import math
import logging
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table

from ..orchestration.pipeline import CDSSPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
console = Console()

# Curated MedQA USMLE clinical vignettes representing core diagnostic categories
BENCHMARK_CASES = [
    {
        "id": "MEDQA_001",
        "question": "A 54-year-old man presents to the emergency department with acute retrosternal chest pain that began 2 hours ago. He describes the pain as crushing and radiating down his left arm and up to his jaw. He is diaphoretic and short of breath. ECG reveals ST-segment elevations in leads II, III, and aVF. Troponin I is significantly elevated.",
        "gold_diagnosis": "Myocardial Infarction",
        "options": ["Myocardial Infarction", "Aortic Dissection", "Gastroesophageal Reflux", "Costochondritis"]
    },
    {
        "id": "MEDQA_002",
        "question": "A 32-year-old woman presents with acute onset of severe pleuritic chest pain and shortness of breath. She recently completed a 14-hour transcontinental flight 3 days ago. Her heart rate is 118 bpm, respiratory rate 26 bpm, and oxygen saturation 89% on ambient air. Physical exam shows unilateral right calf swelling and tenderness.",
        "gold_diagnosis": "Pulmonary Embolism",
        "options": ["Pulmonary Embolism", "Pneumothorax", "Pneumonia", "Anxiety Disorder"]
    },
    {
        "id": "MEDQA_003",
        "question": "A 19-year-old male with type 1 diabetes presents with nausea, vomiting, diffuse abdominal pain, and confusion. Physical exam reveals deep, rapid respirations with a fruity breath odor and signs of severe dehydration. Laboratory results: Blood glucose 480 mg/dL, serum bicarbonate 11 mEq/L, arterial pH 7.18, and positive serum ketones.",
        "gold_diagnosis": "Diabetic Ketoacidosis",
        "options": ["Diabetic Ketoacidosis", "Hyperosmolar Hyperglycemic State", "Acute Pancreatitis", "Gastroenteritis"]
    },
    {
        "id": "MEDQA_004",
        "question": "A 21-year-old college student presents with abdominal pain that started around his umbilicus 12 hours ago and has now localized to the right lower quadrant. He reports nausea and loss of appetite. On exam, there is tenderness at McBurney's point with guarding. WBC count is 14,500/uL with 82% neutrophils.",
        "gold_diagnosis": "Appendicitis",
        "options": ["Appendicitis", "Meckel Diverticulitis", "Crohn Disease", "Renal Colic"]
    },
    {
        "id": "MEDQA_005",
        "question": "A 48-year-old woman presents with severe epigastric pain radiating directly through to her mid-back. The pain is partially relieved when sitting forward. She has had multiple episodes of vomiting. Serum amylase is 850 U/L and serum lipase is 1,200 U/L (normal < 60 U/L).",
        "gold_diagnosis": "Acute Pancreatitis",
        "options": ["Acute Pancreatitis", "Cholecystitis", "Peptic Ulcer Disease", "Mesenteric Ischemia"]
    },
    {
        "id": "MEDQA_006",
        "question": "A 68-year-old man presents with shaking chills, high fever (39.2 C), and a productive cough with rust-colored sputum. Auscultation reveals bronchial breath sounds and inspiratory crackles over the right lower lobe. Chest radiography confirms right lower lobe alveolar consolidation.",
        "gold_diagnosis": "Pneumonia",
        "options": ["Pneumonia", "Pulmonary Embolism", "Bronchiectasis", "Lung Abscess"]
    },
    {
        "id": "MEDQA_007",
        "question": "A 28-year-old woman presents with recurrent bouts of chronic diarrhea, abdominal cramping, flatulence, and unintentional 5 kg weight loss over the past 6 months. Lab analysis reveals microcytic iron deficiency anemia. Serology demonstrates markedly elevated anti-tissue transglutaminase (anti-tTG) IgA titers.",
        "gold_diagnosis": "Celiac Disease",
        "options": ["Celiac Disease", "Irritable Bowel Syndrome", "Lactose Intolerance", "Ulcerative Colitis"]
    },
    {
        "id": "MEDQA_008",
        "question": "A 35-year-old woman presents with dry cough, fatigue, and painful red subcutaneous nodules on her shins (erythema nodosum). Chest radiograph reveals bilateral symmetrical hilar lymphadenopathy. Serum angiotensin-converting enzyme (ACE) is elevated. Biopsy confirms non-caseating granulomas.",
        "gold_diagnosis": "Sarcoidosis",
        "options": ["Sarcoidosis", "Tuberculosis", "Histoplasmosis", "Lymphoma"]
    }
]


def calculate_wilson_ci(correct: int, total: int, z: float = 1.96):
    """Calculates Wilson score 95% confidence interval for binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    p = correct / total
    denom = 1 + (z**2) / total
    center = (p + (z**2) / (2 * total)) / denom
    margin = (z * math.sqrt((p * (1 - p) / total) + ((z**2) / (4 * total**2)))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def run_eval():
    console.print("\n[bold cyan]=== Running MedQA Clinical Diagnostic Evaluation ===[/bold cyan]\n")
    pipeline = CDSSPipeline()

    total = len(BENCHMARK_CASES)
    correct_count = 0
    total_latency = 0.0
    iteration_counts = []

    table = Table(title="Evaluation Case Results")
    table.add_column("Case ID", style="cyan")
    table.add_column("Gold Diagnosis", style="green")
    table.add_column("System Diagnosis", style="yellow")
    table.add_column("Confidence", style="magenta")
    table.add_column("Iterations", style="blue")
    table.add_column("Correct?", style="bold")

    for case in BENCHMARK_CASES:
        res = pipeline.run(case["question"])
        final_report = res.get("final_report") or {}
        predicted = final_report.get("primary_diagnosis", "Unknown")
        confidence = final_report.get("confidence_score", 0.0)
        num_iters = len(res.get("iterations", []))
        latency = res.get("total_latency_ms", 0.0)

        gold = case["gold_diagnosis"]
        is_correct = (gold.lower() in predicted.lower()) or (predicted.lower() in gold.lower())
        if is_correct:
            correct_count += 1

        total_latency += latency
        iteration_counts.append(num_iters)

        mark = "[green]✓ PASS[/green]" if is_correct else "[red]✗ FAIL[/red]"
        table.add_row(
            case["id"],
            gold,
            predicted,
            f"{confidence:.2f}",
            str(num_iters),
            mark
        )

    console.print(table)

    acc = (correct_count / total) * 100
    ci_low, ci_high = calculate_wilson_ci(correct_count, total)
    avg_latency = total_latency / total
    avg_iters = sum(iteration_counts) / len(iteration_counts)

    summary_table = Table(title="Overall Benchmark Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="bold green")

    summary_table.add_row("Total Cases Evaluated", str(total))
    summary_table.add_row("Correct Diagnoses", f"{correct_count}/{total}")
    summary_table.add_row("Accuracy", f"{acc:.1f}%")
    summary_table.add_row("Wilson 95% Confidence Interval", f"[{ci_low*100:.1f}%, {ci_high*100:.1f}%]")
    summary_table.add_row("Mean Adaptive Iterations", f"{avg_iters:.2f}")
    summary_table.add_row("Mean Latency per Case", f"{avg_latency:.1f} ms")

    console.print(summary_table)
    console.print("\n[bold green]✓ Benchmark Evaluation Finished Successfully![/bold green]\n")


if __name__ == "__main__":
    run_eval()

