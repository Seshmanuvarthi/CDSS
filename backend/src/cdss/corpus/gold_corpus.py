"""Gold-standard clinical benchmark corpus for emergency and acute medical vignettes.
Provides high-yield, textbook-grounded diagnostic passages (Harrison's, ADA, ACC/AHA, ESC).
"""

from typing import List, Dict, Any


GOLD_CLINICAL_PASSAGES = [
    {
        "doc_id": "gold_dka_01",
        "condition": "Diabetic Ketoacidosis (DKA)",
        "source": "ADA 2024 / ISPAD Consensus Guidelines (Diabetic Ketoacidosis)",
        "kind": "textbook-guideline",
        "evidence_level": "Level A Guideline",
        "text": (
            "Condition: Diabetic Ketoacidosis (DKA)\n"
            "Key Symptoms & Presentation: Acute life-threatening metabolic emergency occurring in patients with Type 1 "
            "diabetes mellitus, characteristically precipitated by omission or missed doses of insulin, systemic infection, "
            "or severe physical stress. Symptoms develop acutely over 24 to 48 hours and include excessive thirst (polydipsia), "
            "frequent urination (polyuria), profound fatigue, nausea, persistent vomiting, diffuse abdominal pain, and marked dehydration. "
            "Physical examination reveals deep, rapid breathing (Kussmaul respirations) representing compensatory hyperventilation for "
            "severe metabolic acidosis, fruity or acetone breath odor, sinus tachycardia (heart rate > 100-120 bpm), hypotension or "
            "orthostasis, and dry mucous membranes.\n"
            "Diagnostic Criteria: The diagnosis is firmly established by the classic clinical triad:\n"
            "1. Hyperglycemia: Blood glucose > 250 mg/dL (typically 350-500+ mg/dL).\n"
            "2. High Anion Gap Metabolic Acidosis: Venous pH < 7.30 (severe if pH < 7.20 or < 7.00) and serum bicarbonate < 15-18 mmol/L "
            "(severe if bicarbonate < 10 mmol/L) with an elevated anion gap [Na - (Cl + HCO3)] > 12-16 mEq/L.\n"
            "3. Ketonemia & Ketonuria: Serum beta-hydroxybutyrate >= 3.0 mmol/L (predominant circulating ketone body) and moderate-to-large "
            "ketones on urinalysis (acetoacetate).\n"
            "Emergency Management: Immediate priority is rapid intravenous volume expansion with isotonic saline (0.9% NaCl, 1-1.5 L/hr initially), "
            "followed by regular insulin infusion (0.1 units/kg/h) once serum potassium is >= 3.3 mEq/L, and aggressive potassium repletion "
            "(20-30 mEq/L IV fluid) to prevent fatal arrhythmias as insulin drives potassium intracellularly."
        )
    },
    {
        "doc_id": "gold_dka_02",
        "condition": "Diabetic Ketoacidosis (DKA) vs Hyperglycemic Differentials",
        "source": "Harrison's Principles of Internal Medicine (Endocrine Emergencies)",
        "kind": "textbook-guideline",
        "evidence_level": "Level A Guideline",
        "text": (
            "Condition: Diabetic Ketoacidosis (DKA)\n"
            "Differential Diagnosis and Laboratory Distinction: In any diabetic patient presenting with vomiting, nausea, abdominal pain, "
            "and hyperventilation, DKA must be distinguished from Hyperosmolar Hyperglycemic State (HHS), alcoholic ketoacidosis, uremic acidosis, "
            "and cyclic vomiting syndrome. DKA is characterized by substantial ketonemia (beta-hydroxybutyrate > 3.0 mmol/L), metabolic acidosis "
            "(pH < 7.30, HCO3 < 18), and positive urine ketones. In HHS, glucose is higher (> 600 mg/dL) but significant ketoacidosis and severe "
            "acidosis are absent (pH > 7.30, HCO3 > 15, minimal ketones). In end-stage renal disease (ESRD), chronic normocytic anemia and "
            "elevated creatinine are present, but acute massive ketosis and acute insulin-omission hyperglycemia do not occur in non-diabetic ESRD. "
            "In cyclic vomiting syndrome, metabolic panels between episodes are completely normal and marked hyperglycemia with ketonemia is absent."
        )
    },
    {
        "doc_id": "gold_stemi_01",
        "condition": "Acute Inferior Myocardial Infarction (STEMI)",
        "source": "Fourth Universal Definition of Myocardial Infarction (ESC/ACC/AHA)",
        "kind": "textbook-guideline",
        "evidence_level": "Level A Guideline",
        "text": (
            "Condition: Acute Inferior Myocardial Infarction (STEMI)\n"
            "Key Symptoms & Presentation: Acute myocardial injury caused by acute coronary artery occlusion, most commonly the right coronary "
            "artery (RCA). Presentation involves severe crushing central substernal chest pressure radiating to the left arm, neck, or jaw, "
            "accompanied by diaphoresis, dyspnea, nausea, and lightheadedness.\n"
            "Diagnostic Criteria: New ST-segment elevation >= 1 mm (0.1 mV) in two or more anatomically contiguous inferior leads (II, III, aVF), "
            "often with reciprocal ST-segment depression in lead aVL, accompanied by a rise and fall of high-sensitivity cardiac troponin above the "
            "99th percentile upper reference limit.\n"
            "Management: Immediate activation of cardiac catheterization laboratory for emergent primary percutaneous coronary intervention (PCI), "
            "dual antiplatelet therapy (aspirin and P2Y12 inhibitor), and anticoagulation with unfractionated heparin."
        )
    },
    {
        "doc_id": "gold_pe_01",
        "condition": "Acute Pulmonary Embolism (PE)",
        "source": "2019 ESC Guidelines for the Diagnosis and Management of Acute Pulmonary Embolism",
        "kind": "textbook-guideline",
        "evidence_level": "Level A Guideline",
        "text": (
            "Condition: Acute Pulmonary Embolism (PE)\n"
            "Key Symptoms & Presentation: Obstruction of pulmonary arterial bed by thrombi originating primarily from deep vein thrombosis (DVT) "
            "of the lower extremities. Key risk factors include recent prolonged international air travel/immobilization, active malignancy, or surgery. "
            "Symptoms include sudden-onset dyspnea, sharp pleuritic right- or left-sided chest pain worsening with inspiration, tachycardia (HR > 100 bpm), "
            "tachypnea (RR > 20/min), hypoxia (SpO2 < 92%), dizziness, and unilateral calf swelling or tenderness.\n"
            "Diagnostic Criteria: High pre-test clinical probability (Wells score > 4.0), elevated D-dimer, and definitive intravascular filling defects "
            "in lobar or segmental pulmonary arteries on CT pulmonary angiography (CTPA). High-sensitivity cardiac troponin is typically normal unless "
            "massive right ventricular strain occurs.\n"
            "Management: Immediate therapeutic anticoagulation with low-molecular-weight heparin (LMWH) or direct oral anticoagulants (DOACs)."
        )
    },
    {
        "doc_id": "gold_aad_01",
        "condition": "Acute Aortic Dissection (Stanford Type A)",
        "source": "2022 ACC/AHA Guidelines for the Diagnosis and Management of Aortic Disease",
        "kind": "textbook-guideline",
        "evidence_level": "Level A Guideline",
        "text": (
            "Condition: Acute Aortic Dissection (Stanford Type A)\n"
            "Key Symptoms & Presentation: Tear in the aortic intima allowing blood to enter the media and create a false lumen. "
            "Classically presents in hypertensive patients with sudden-onset, catastrophic chest pain of tearing, ripping, or sharp quality, "
            "frequently radiating between the shoulder blades into the back. Physical examination demonstrates marked blood pressure asymmetry "
            "(> 20 mmHg systolic difference between right and left arms), asymmetrical radial pulses, and diaphoresis.\n"
            "Diagnostic Criteria: Contrast-enhanced CT angiography (CTA) of the chest confirms the diagnosis by visualizing an intimal flap "
            "separating true and false lumens in the ascending aorta (Stanford Type A).\n"
            "Management: Emergent cardiovascular surgical consultation for open aortic repair, aggressive blood pressure and heart rate control "
            "with intravenous beta-blockers (esmolol or labetalol) targeting systolic BP 100-120 mmHg and HR < 60 bpm."
        )
    },
    {
        "doc_id": "gold_pneu_01",
        "condition": "Tension Pneumothorax with Obstructive Shock",
        "source": "Advanced Trauma Life Support (ATLS, 10th Ed) / BTS Pleural Disease Guidelines",
        "kind": "textbook-guideline",
        "evidence_level": "Level A Guideline",
        "text": (
            "Condition: Tension Pneumothorax with Obstructive Shock\n"
            "Key Symptoms & Presentation: Life-threatening thoracic emergency caused by one-way valve pleural air accumulation "
            "resulting in complete ipsilateral lung collapse, massive mediastinal shift, vena cava compression, and cardiovascular collapse (obstructive shock). "
            "Commonly precipitated by blunt or penetrating chest trauma (e.g. motorcycle collision, rib fractures) or mechanical ventilation. "
            "Classic presentation features acute severe dyspnea, right-sided or left-sided chest pain, tachypnea (RR > 30 breaths/min), "
            "hypoxemia (SpO2 < 85%), marked sinus tachycardia (HR > 120-140 bpm), agitation, diaphoresis, and profound hypotension (e.g. BP 78/46 mmHg).\n"
            "Physical Examination: Cardinal signs include unilaterally markedly reduced or absent breath sounds on the affected side, "
            "distended neck veins (jugular venous distension, JVD) secondary to impaired right ventricular venous return, and tracheal deviation "
            "to the contralateral (opposite) side away from the tension pneumothorax.\n"
            "Diagnostic Imaging & Bedside Ultrasound: Bedside point-of-care ultrasound (POCUS) confirms absent lung sliding on the affected hemithorax "
            "(the 'barcode' or 'stratosphere' sign on M-mode), absence of B-lines, and presence of a lung point.\n"
            "Emergency Management: Tension pneumothorax is an immediate clinical diagnosis requiring IMMEDIATE emergency needle decompression "
            "(using a large-bore 14-16G needle/cannula at the 2nd intercostal space midclavicular line or 4th/5th intercostal space anterior axillary line), "
            "followed by urgent tube thoracostomy (chest tube insertion). Treatment MUST NEVER be delayed for chest radiography or laboratory tests "
            "in the setting of acute respiratory distress and obstructive shock."
        )
    },
    {
        "doc_id": "gold_crohn_01",
        "condition": "Crohn's Disease",
        "source": "ECCO / ACG Consensus Guidelines (Diagnosis and Management of Crohn's Disease)",
        "kind": "textbook-guideline",
        "evidence_level": "Level A Guideline",
        "text": (
            "Condition: Crohn's Disease (Regional Enteritis)\n"
            "Key Symptoms & Presentation: Chronic inflammatory bowel disease characterized by transmural inflammation that can affect any part "
            "of the gastrointestinal tract from mouth to anus, most frequently the terminal ileum and right colon. Patients present with insidious "
            "onset of chronic diarrhea (> 4-6 weeks), crampy abdominal pain typically localized to the right lower quadrant, low-grade fever, "
            "fatigue, anorexia, and progressive unintended weight loss. Physical examination may reveal right lower quadrant fullness or tenderness, "
            "perianal skin tags or fistulas, and aphthous oral ulcers.\n"
            "Diagnostic Criteria: Multimodal diagnostic confirmation requires:\n"
            "1. Clinical Presentation: Chronic diarrhea, recurrent abdominal pain, fatigue, and weight loss.\n"
            "2. Inflammatory Biomarkers: Markedly elevated fecal calprotectin (> 250 mcg/g) demonstrating active neutrophil migration into the intestinal mucosa; "
            "elevated C-reactive protein (CRP) and ESR.\n"
            "3. Endoscopic Findings (Ileocolonoscopy): Patchy mucosal inflammation, longitudinal and serpiginous ulcers separated by normal mucosa ('skip lesions'), "
            "cobblestone mucosal appearance, with frequent involvement of the terminal ileum and sparing of the rectum.\n"
            "4. Histopathology: Transmural inflammatory infiltrate, submucosal lymphoid aggregates, fissuring ulcers, and focal non-caseating granulomas "
            "(pathognomonic when present).\n"
            "Differential Diagnosis: Must be differentiated from Ulcerative Colitis (characterized by continuous, confluent, mucosal inflammation restricted to the colon "
            "starting from the rectum without ileal involvement or granulomas) and Intestinal Tuberculosis (characterized by confluent caseating granulomas with positive acid-fast bacilli staining)."
        )
    },
    {
        "doc_id": "gold_stroke_01",
        "condition": "Acute Ischemic Stroke (Left MCA territory)",
        "source": "AHA/ASA 2019/2021 Guidelines for the Early Management of Acute Ischemic Stroke",
        "kind": "textbook-guideline",
        "evidence_level": "Level A Guideline",
        "text": (
            "Condition: Acute Ischemic Stroke (Left MCA territory)\n"
            "Key Symptoms & Presentation: Acute focal neurological deficits caused by sudden arterial occlusion of cerebral circulation, "
            "most frequently involving the middle cerebral artery (MCA) territory. Hallmarks include sudden expressive or receptive aphasia "
            "(dysphasia), unilateral facial droop, contralateral hemiparesis or hemiplegia (arm and leg weakness, often arm > leg in MCA territory), "
            "dysarthria, hemisensory loss, and conjugate eye deviation. Patients present acutely with onset times measured in minutes to hours.\n"
            "Diagnostic Criteria & Neuroimaging:\n"
            "1. Emergent Non-Contrast Head CT: Immediately performed to definitively rule out acute intracranial hemorrhage (ICH), which is an absolute contraindication to thrombolysis.\n"
            "2. CT Angiography (CTA): Confirms large vessel occlusion (LVO) of the anterior circulation, such as occlusion of the proximal middle cerebral artery (M1 or M2 segment of MCA) or internal carotid artery (ICA).\n"
            "3. Point-of-care glucose: Rapidly excludes hypoglycemia as a stroke mimic.\n"
            "Emergency Management & Reperfusion:\n"
            "- Intravenous Thrombolysis (IV Alteplase or Tenecteplase): Indicated for eligible patients within 4.5 hours of last known well after intracranial hemorrhage is ruled out on non-contrast CT.\n"
            "- Emergent Endovascular Mechanical Thrombectomy (EVT): Strongly recommended (Class I, Level A) for patients with confirmed large vessel occlusion (M1 segment MCA) within 24 hours of symptom onset.\n"
            "- Urgent acute stroke pathway activation ('time is brain') to prevent irreversible neuronal necrosis; avoid delays in imaging or reperfusion."
        )
    },
    {
        "doc_id": "gold_app_01",
        "condition": "Acute Appendicitis",
        "source": "WSES Jerusalem Guidelines on Acute Appendicitis (World Journal of Emergency Surgery)",
        "kind": "textbook-guideline",
        "evidence_level": "Level A Guideline",
        "text": (
            "Condition: Acute Appendicitis\n"
            "Key Symptoms & Presentation: Acute surgical abdominal emergency caused by luminal obstruction of the vermiform appendix (fecalith, lymphoid hyperplasia). "
            "Classic progression starts with vague periumbilical visceral pain migrating over 12-24 hours to the right lower quadrant (RLQ). "
            "Associated symptoms include anorexia, nausea, low-grade fever, and localized peritoneal irritation.\n"
            "Physical Examination: Cardinal physical signs include localized tenderness at McBurney's point (one-third distance from anterior superior iliac spine to umbilicus), "
            "muscular guarding, rebound tenderness (Blumberg sign), Rovsing sign, and psoas/obturator signs.\n"
            "Diagnostic Criteria & Imaging: Leukocytosis with neutrophilic bandemia and elevated C-reactive protein (CRP). "
            "Graded compression ultrasound or contrast-enhanced abdominal CT confirms an enlarged, non-compressible appendix (>6 mm outer diameter), "
            "appendiceal wall thickening, target sign, and periappendiceal fat stranding.\n"
            "Management: Prompt surgical consultation for laparoscopic appendectomy, preoperative broad-spectrum intravenous antibiotic coverage, and isotonic IV fluid resuscitation."
        )
    },
    {
        "doc_id": "gold_panc_01",
        "condition": "Acute Pancreatitis",
        "source": "American College of Gastroenterology (ACG) Guidelines: Management of Acute Pancreatitis",
        "kind": "textbook-guideline",
        "evidence_level": "Level A Guideline",
        "text": (
            "Condition: Acute Pancreatitis\n"
            "Key Symptoms & Presentation: Acute inflammatory disorder of the pancreas most commonly precipitated by biliary lithiasis (gallstones) or alcohol abuse. "
            "Presents with severe, persistent, boring epigastric abdominal pain classically radiating straight through to the mid-back. Pain is characteristically exacerbated by supine positioning "
            "and partially relieved by leaning forward in the tripod position. Accompanied by nausea, intractable vomiting, abdominal distension, tachycardia, and low-grade fever.\n"
            "Diagnostic Criteria (Atlanta Criteria): Diagnosis requires at least 2 of the following 3 features:\n"
            "1. Characteristic abdominal pain (epigastric, radiating to back, relieved by leaning forward).\n"
            "2. Serum amylase or lipase elevation > 3 times the upper limit of normal (lipase preferred due to higher sensitivity and prolonged elevation).\n"
            "3. Characteristic cross-sectional imaging findings (contrast-enhanced CT, MRI, or transabdominal ultrasound demonstrating pancreatic enlargement, peripancreatic fluid collections, or necrosis).\n"
            "Management: Immediate goal-directed intravenous fluid resuscitation (isotonic crystalloids / lactated Ringer's), early enteral nutrition as tolerated, multimodal analgesia, and avoidance of prophylactic antibiotics."
        )
    },
    {
        "doc_id": "gold_mening_01",
        "condition": "Acute Bacterial Meningitis",
        "source": "Infectious Diseases Society of America (IDSA) Guidelines for the Management of Bacterial Meningitis",
        "kind": "textbook-guideline",
        "evidence_level": "Level A Guideline",
        "text": (
            "Condition: Acute Bacterial Meningitis\n"
            "Key Symptoms & Presentation: Life-threatening central nervous system infection of the leptomeninges and subarachnoid space (common pathogens: Streptococcus pneumoniae, Neisseria meningitidis). "
            "Classically presents with the clinical triad of high fever, severe holocephalic headache, and altered mental status, accompanied by severe photophobia and vomiting.\n"
            "Physical Examination: Cardinal signs of meningeal irritation (meningismus) include marked nuchal rigidity (inability to flex neck forward onto chest), "
            "positive Kernig sign (resistance and pain with knee extension when hip is flexed at 90 degrees), and positive Brudzinski sign (involuntary hip and knee flexion elicited by passive neck flexion).\n"
            "Diagnostic Criteria & Cerebrospinal Fluid (CSF): Emergent diagnostic lumbar puncture (LP) demonstrates turbid or purulent CSF under elevated opening pressure (> 200 mm H2O), "
            "marked neutrophilic pleocytosis (typically 1,000-10,000+ WBCs/mcL with >80% polymorphonuclear leukocytes), markedly elevated CSF protein (> 100-500 mg/dL), and markedly decreased CSF glucose "
            "(CSF-to-serum glucose ratio < 0.40 or CSF glucose < 40 mg/dL). Gram stain identifies causative diplococci or bacilli.\n"
            "Emergency Management: Immediate medical emergency requiring empiric intravenous bactericidal therapy (ceftriaxone or cefotaxime plus vancomycin, plus ampicillin in elderly/immunocompromised) "
            "and adjunctive dexamethasone (prior to or with first antibiotic dose). Antibiotic administration MUST NOT be delayed if lumbar puncture or neuroimaging is pending."
        )
    },
    {
        "doc_id": "gold_hf_01",
        "condition": "Acute Decompensated Heart Failure with Pulmonary Edema (HFrEF)",
        "source": "2021 ESC Guidelines for the Diagnosis and Treatment of Acute and Chronic Heart Failure / 2022 AHA/ACC/HFSA",
        "kind": "textbook-guideline",
        "evidence_level": "Level A Guideline",
        "text": (
            "Condition: Acute Decompensated Heart Failure with Pulmonary Edema and Reduced Ejection Fraction (HFrEF)\n"
            "Key Symptoms & Presentation: Acute life-threatening clinical syndrome caused by severe cardiac congestion and elevated left ventricular filling pressures, "
            "frequently precipitated by uncontrolled hypertension, myocardial ischemia, or medication non-compliance. "
            "Patients present with rapidly worsening severe dyspnea, orthopnea (difficulty breathing when lying flat), paroxysmal nocturnal dyspnea (PND, waking at night gasping for air), "
            "and bilateral lower extremity peripheral edema. Physical examination demonstrates accessory respiratory muscle use, tachypnea (RR > 25-30/min), "
            "hypoxemia (SpO2 < 90%), sinus tachycardia, elevated jugular venous pressure (JVP), bilateral inspiratory crackles (pulmonary rales) extending through lung fields, "
            "and bilateral pitting edema.\n"
            "Diagnostic Criteria & Biomarkers:\n"
            "1. Natriuretic Peptides: Markedly elevated B-type natriuretic peptide (BNP > 400-1,000 pg/mL, typically > 1,500 pg/mL) or NT-proBNP (> 1,800 pg/mL) confirms cardiogenic congestion.\n"
            "2. Chest Radiography (CXR): Demonstrates pulmonary vascular congestion, bilateral perihilar alveolar opacities ('batwing' or 'butterfly' pattern of pulmonary alveolar edema), "
            "Kerley B lines, cardiomegaly, and bilateral pleural effusions.\n"
            "3. Echocardiography: Transthoracic echocardiogram confirms severely reduced left ventricular ejection fraction (LVEF <= 40%, e.g., LVEF 30%) with global or regional hypokinesis and elevated LV filling pressures.\n"
            "Differential Diagnosis: Must be distinguished from Acute Pulmonary Embolism (PE). In heart failure with pulmonary edema, bilateral crackles, orthopnea, elevated BNP, "
            "bilateral alveolar infiltrates, and severe LV systolic dysfunction (LVEF <= 35%) predominate, whereas in acute PE, lung fields are typically clear, CTPA demonstrates intravascular filling defects, "
            "and acute right ventricular strain rather than left ventricular global systolic hypokinesis is present.\n"
            "Emergency Management: Immediate non-invasive positive pressure ventilation (CPAP/BiPAP), intravenous loop diuretics (furosemide), IV vasodilators (nitroglycerin) if systolic BP is elevated, "
            "oxygen therapy targeting SpO2 >= 92%, and hemodynamic stabilization in an intensive cardiac care setting."
        )
    }
]


def load_gold_clinical_passages() -> List[Dict[str, Any]]:
    """Returns structured gold-standard clinical passages for RAG indexing."""
    return GOLD_CLINICAL_PASSAGES

