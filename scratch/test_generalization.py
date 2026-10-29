"""Evaluation test script for the Generalized Five-Agent CDSS.
Tests the 6 gold benchmark cases, 4 novel clinical conditions, and 1 ambiguous case.
"""

import os
import sys
import json
import time

# Ensure backend/src is on python path
sys.path.insert(0, os.path.abspath("backend/src"))

from cdss.orchestration.pipeline import CDSSPipeline

CASES = [
    {
        "id": "BENCH_01_STEMI",
        "name": "Inferior STEMI",
        "expected": ["myocardial infarction", "stemi", "inferior stemi"],
        "text": (
            "A 58-year-old male presents with 2 hours of crushing substernal chest pressure radiating to the left arm, "
            "accompanied by profuse diaphoresis and nausea. Vital signs: HR 64 bpm, BP 118/74 mmHg, RR 18/min, SpO2 98%. "
            "Diagnostic laboratory findings: High-sensitivity cardiac troponin I is markedly elevated at 2.45 ng/mL (reference < 0.04 ng/mL). "
            "Diagnostic ECG: 12-lead electrocardiogram demonstrates 3 mm ST-segment elevation in inferior leads II, III, and aVF "
            "with reciprocal ST-segment depression in leads I and aVL."
        )
    },
    {
        "id": "BENCH_02_PE",
        "name": "Acute Pulmonary Embolism",
        "expected": ["pulmonary embolism", "pe"],
        "text": (
            "A 46-year-old female presents with acute onset pleuritic right-sided chest pain, sudden dyspnea, and right calf discomfort "
            "4 hours after completing a 12-hour international flight. Vital signs: HR 118 bpm, BP 124/82 mmHg, RR 26/min, SpO2 90% on room air. "
            "Diagnostic laboratory findings: Quantitative D-dimer is markedly elevated at 2,850 ng/mL. High-sensitivity troponin is normal. "
            "Diagnostic Imaging: Contrast-enhanced CT pulmonary angiography demonstrates extensive bilateral filling defects in the main right lower lobar "
            "and segmental pulmonary arteries."
        )
    },
    {
        "id": "BENCH_03_DKA",
        "name": "Diabetic Ketoacidosis",
        "expected": ["diabetic ketoacidosis", "dka"],
        "text": (
            "A 24-year-old female with type 1 diabetes mellitus presents with nausea, vomiting, abdominal pain, and deep, rapid Kussmaul respirations "
            "after omitting insulin doses for three days. Vital signs: HR 122 bpm, BP 98/62 mmHg, RR 30/min, SpO2 99%, temperature 37.1°C. "
            "Diagnostic laboratory findings: Point-of-care blood glucose is 480 mg/dL, venous blood gas shows pH 7.14 with serum bicarbonate 8 mmol/L "
            "(severe high anion gap metabolic acidosis). Serum beta-hydroxybutyrate is elevated at 5.4 mmol/L, and urinalysis confirms 4+ ketones and glucose."
        )
    },
    {
        "id": "BENCH_04_AAD",
        "name": "Stanford Type A Aortic Dissection",
        "expected": ["aortic dissection", "stanford type a"],
        "text": (
            "A 63-year-old male with chronic hypertension presents with sudden, catastrophic tearing chest pain radiating between the shoulder blades, "
            "dizziness, and diaphoresis. Vital signs: HR 104 bpm, RR 22/min. Right arm blood pressure is 182/98 mmHg, while left arm blood pressure is 144/86 mmHg "
            "(38 mmHg inter-arm difference). Asymmetric radial pulses are noted. Diagnostic laboratory findings: High-sensitivity cardiac troponin is within normal reference range. "
            "Diagnostic Imaging: Emergency CT angiography of the chest demonstrates an intimal flap originating in the ascending aorta extending into the aortic arch, "
            "separating true and false lumens (Stanford Type A)."
        )
    },
    {
        "id": "BENCH_05_PNEUMO",
        "name": "Right-Sided Tension Pneumothorax",
        "expected": ["tension pneumothorax", "pneumothorax"],
        "text": (
            "A 28-year-old male is brought to the emergency department following a high-speed motorcycle collision with severe blunt chest trauma. "
            "Physical examination reveals acute respiratory distress, cyanosis, distended jugular neck veins (JVD), contralateral tracheal deviation to the left, "
            "and unilaterally absent breath sounds over the right hemithorax with hyperresonance on percussion. Vital signs: HR 140 bpm, BP 76/44 mmHg "
            "(severe obstructive shock), RR 34/min, SpO2 84%. Diagnostic Imaging: Bedside point-of-care thoracic ultrasound demonstrates absent lung sliding "
            "and stratosphere/barcode sign on the right."
        )
    },
    {
        "id": "BENCH_06_CROHN",
        "name": "Crohn's Disease",
        "expected": ["crohn", "crohn's disease", "inflammatory bowel disease"],
        "text": (
            "A 26-year-old female presents with a 4-month history of chronic watery diarrhea, crampy lower abdominal pain, low-grade fevers, "
            "and unintended 7 kg weight loss. Diagnostic laboratory findings: Fecal calprotectin is markedly elevated at 680 mcg/g (reference < 50 mcg/g). "
            "CRP is elevated at 34 mg/L. Diagnostic Imaging: Ileocolonoscopy demonstrates patchy mucosal ulceration, deep linear fissures, "
            "and cobblestoning with normal intervening mucosa (skip lesions) involving the terminal ileum and ascending colon. "
            "Histopathology of ileal biopsies reveals transmural active inflammation with non-caseating granulomas."
        )
    },
    {
        "id": "BENCH_07_STROKE",
        "name": "Acute Ischemic Stroke (Novel Condition 1)",
        "expected": ["ischemic stroke", "stroke", "middle cerebral artery"],
        "text": (
            "A 67-year-old male develops sudden expressive aphasia, right facial droop, and right arm weakness 45 minutes prior to emergency arrival. "
            "No preceding headache, seizure, or trauma. Vital signs: BP 174/96 mmHg, HR 92 bpm, RR 18/min, SpO2 97%, temperature 36.7°C. "
            "Point-of-care glucose 112 mg/dL. Diagnostic Imaging: Non-contrast head CT demonstrates no acute intracranial hemorrhage. "
            "CT angiography of the head and neck confirms an acute large vessel occlusion of the proximal left M1 middle cerebral artery."
        )
    },
    {
        "id": "BENCH_08_APPENDICITIS",
        "name": "Acute Appendicitis (Novel Condition 2)",
        "expected": ["appendicitis", "acute appendicitis"],
        "text": (
            "A 21-year-old male presents with 18 hours of worsening abdominal pain that began periumbilically and subsequently localized to the right lower quadrant. "
            "Accompanied by anorexia, nausea, and low-grade fever. Physical examination reveals maximal tenderness at McBurney's point with local rebound tenderness "
            "and involuntary guarding in the right lower quadrant. Vital signs: HR 98 bpm, BP 120/78 mmHg, temperature 38.2°C, RR 16/min. "
            "Diagnostic laboratory findings: WBC count is elevated at 14.8 x 10^3/mcL with 82% neutrophils (leukocytosis with left shift). "
            "Diagnostic Imaging: Transabdominal ultrasound demonstrates a non-compressible, blind-ending tubular appendiceal structure measuring 8.5 mm "
            "in outer diameter with hyperemia and surrounding periappendiceal fat stranding."
        )
    },
    {
        "id": "BENCH_09_PANCREATITIS",
        "name": "Acute Pancreatitis (Novel Condition 3)",
        "expected": ["pancreatitis", "acute pancreatitis"],
        "text": (
            "A 48-year-old male with a history of alcohol use presents with 12 hours of acute, severe epigastric abdominal pain radiating straight through to the back, "
            "partially relieved by leaning forward in the seated position. Accompanied by intractable nausea and multiple episodes of vomiting. "
            "Physical examination reveals marked epigastric tenderness with guarding and abdominal distension. Vital signs: HR 112 bpm, BP 110/70 mmHg, "
            "RR 22/min, temperature 37.8°C. Diagnostic laboratory findings: Serum lipase is markedly elevated at 1,480 U/L (reference 10-140 U/L, > 10x upper limit of normal). "
            "Serum amylase is elevated at 640 U/L."
        )
    },
    {
        "id": "BENCH_10_MENINGITIS",
        "name": "Acute Bacterial Meningitis (Novel Condition 4)",
        "expected": ["meningitis", "bacterial meningitis", "acute bacterial meningitis"],
        "text": (
            "A 19-year-old college student presents with acute onset of high fever, severe generalized headache, nausea, and neck stiffness developing over 12 hours. "
            "Physical examination reveals marked nuchal rigidity, positive Kernig sign, and positive Brudzinski sign. "
            "Vital signs: HR 116 bpm, BP 106/68 mmHg, RR 20/min, temperature 39.4°C. "
            "Diagnostic laboratory findings: Emergent lumbar puncture demonstrates cloudy, turbid cerebrospinal fluid (CSF) under opening pressure of 260 mm H2O. "
            "CSF analysis reveals 2,200 WBCs/mcL with 92% polymorphonuclear neutrophils, marked CSF protein elevation at 240 mg/dL, and decreased CSF glucose at 22 mg/dL "
            "(CSF-to-serum glucose ratio 0.22). CSF Gram stain shows Gram-positive diplococci."
        )
    },
    {
        "id": "BENCH_11_AMBIGUOUS",
        "name": "Unexplained Fatigue / Non-Specific Presentation (Ambiguous)",
        "expected": ["abstain"],
        "text": (
            "A 42-year-old female presents with 3 months of mild, intermittent generalized fatigue and vague lightheadedness that occurs intermittently without a clear trigger. "
            "Denies fever, chest pain, shortness of breath, palpitation, weight loss, abdominal pain, or focal neurological deficits. "
            "Vital signs: HR 72 bpm, BP 118/76 mmHg, RR 14/min, SpO2 99%, temperature 36.8°C. Physical examination is completely unremarkable without focal findings. "
            "Diagnostic laboratory findings: Complete blood count, comprehensive metabolic panel, thyroid stimulating hormone, and high-sensitivity troponin are all completely normal. "
            "12-lead ECG is normal sinus rhythm. Non-contrast head CT is normal."
        )
    },
    {
        "id": "BENCH_12_HEART_FAILURE",
        "name": "Acute Decompensated Heart Failure with Pulmonary Edema (CDSS-GEN-012)",
        "expected": ["heart failure", "pulmonary edema", "hfref"],
        "text": (
            "A 68-year-old man presents with rapidly worsening shortness of breath over two days, severe difficulty breathing when lying flat, "
            "waking at night gasping for air, and bilateral leg swelling for one week. He has a history of hypertension but no known prior heart failure. "
            "On examination, he is using accessory respiratory muscles and has bilateral inspiratory crackles, elevated jugular venous pressure, "
            "and pitting edema in both legs. "
            "Vital signs: Temperature 36.8°C, heart rate 118 bpm, blood pressure 168/102 mmHg, respiratory rate 30/min, oxygen saturation 86% on room air. "
            "Laboratory findings: BNP 1,850 pg/mL, high-sensitivity troponin mildly elevated without a significant serial rise, creatinine 1.4 mg/dL. "
            "Imaging: Chest radiograph shows bilateral perihilar alveolar opacities, pulmonary vascular congestion, and small bilateral pleural effusions. "
            "Echocardiography shows left ventricular ejection fraction of 30% with global hypokinesis."
        )
    }
]


def run_tests():
    print("=" * 80)
    print(f"EVALUATING GENERALIZED 5-AGENT CDSS ACROSS {len(CASES)} CLINICAL BENCHMARKS")
    print("=" * 80)

    pipeline = CDSSPipeline()
    results = []

    for i, case in enumerate(CASES, 1):
        print(f"\n[{i}/{len(CASES)}] Running {case['id']}: {case['name']}...")
        t0 = time.time()
        trace = pipeline.run(case["text"])
        elapsed = time.time() - t0

        final_rep = trace.get("final_report") or {}
        primary = final_rep.get("primary_diagnosis", "None")
        conf = final_rep.get("confidence_score", 0.0)
        meets_gate = final_rep.get("meets_confidence_threshold", False)
        status = trace.get("status", "unknown")
        iterations = len(trace.get("iterations", []))

        is_ambiguous = case["id"] == "BENCH_11_AMBIGUOUS"
        if is_ambiguous:
            # Correct behavior: confidence stays below threshold, 3 iterations exhausted, abstained
            passed = (not meets_gate) and (conf < 0.65) and (status == "abstained_for_physician_review" or iterations >= 3)
            verdict = "PASS (Correctly Abstained)" if passed else "FAIL"
        else:
            matches_expected = any(e in primary.lower() for e in case["expected"])
            passed = matches_expected and meets_gate and (conf >= 0.65)
            verdict = "PASS" if passed else "FAIL"

        print(f"    Verdict: {verdict}")
        print(f"    Primary Diagnosis: {primary}")
        print(f"    Confidence: {conf:.3f} | Meets Gate (>=0.65): {meets_gate}")
        print(f"    Iterations: {iterations} | Status: {status} | Latency: {elapsed:.2f}s")

        results.append({
            "id": case["id"],
            "name": case["name"],
            "primary": primary,
            "confidence": conf,
            "meets_gate": meets_gate,
            "iterations": iterations,
            "status": status,
            "verdict": verdict,
            "passed": passed,
            "latency": round(elapsed, 2)
        })

    print("\n" + "=" * 80)
    print("CONSOLIDATED BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Case ID':<22} | {'Name':<35} | {'Reported Diagnosis':<30} | {'Conf':<6} | {'Gate':<6} | {'Verdict'}")
    print("-" * 120)
    total_passed = sum(1 for r in results if r["passed"])
    for r in results:
        g_str = "YES" if r["meets_gate"] else "NO"
        print(f"{r['id']:<22} | {r['name']:<35} | {r['primary'][:30]:<30} | {r['confidence']:<6.3f} | {g_str:<6} | {r['verdict']}")
    print("-" * 120)
    print(f"TOTAL: {total_passed}/{len(results)} cases passed ({total_passed/len(results)*100:.1f}%)")
    print("=" * 80)

    # Save results to json
    with open("benchmark_generalization_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Report saved to benchmark_generalization_report.json")


if __name__ == "__main__":
    run_tests()

