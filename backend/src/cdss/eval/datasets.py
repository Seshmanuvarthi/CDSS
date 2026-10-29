"""Benchmark dataset loader for MedQA (USMLE), PubMedQA, and MedBullets.
Supports full dataset loading, cached offline loading, and randomized subsets.
"""

import os
import glob
import json
import random
import logging
from typing import List, Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)

# -------------------------------------------------------------
# 1. 50 Gold-Standard MedQA USMLE Clinical Diagnostic Cases
# -------------------------------------------------------------
MEDQA_50_CASES = [
    {
        "id": "MEDQA_001",
        "question": "A 54-year-old male presents to the ED with acute retrosternal crushing chest pain radiating to his left arm and jaw. He is diaphoretic, short of breath, with ST elevations in leads II, III, and aVF. Elevated cardiac troponin I.",
        "gold_diagnosis": "Myocardial Infarction",
        "options": {"A": "Myocardial Infarction", "B": "Aortic Dissection", "C": "Gastroesophageal Reflux", "D": "Costochondritis"}
    },
    {
        "id": "MEDQA_002",
        "question": "A 32-year-old female presents with acute pleuritic chest pain and dyspnea after a 14-hour flight. Tachycardia (118 bpm), hypoxia (89% O2 sat), and unilateral right calf swelling and tenderness.",
        "gold_diagnosis": "Pulmonary Embolism",
        "options": {"A": "Pulmonary Embolism", "B": "Pneumothorax", "C": "Pneumonia", "D": "Anxiety Disorder"}
    },
    {
        "id": "MEDQA_003",
        "question": "A 19-year-old male with type 1 diabetes presents with vomiting, diffuse abdominal pain, confusion, Kussmaul respirations, fruity breath odor. Blood glucose 480 mg/dL, arterial pH 7.18, positive ketones.",
        "gold_diagnosis": "Diabetic Ketoacidosis",
        "options": {"A": "Diabetic Ketoacidosis", "B": "Hyperosmolar Hyperglycemic State", "C": "Acute Pancreatitis", "D": "Gastroenteritis"}
    },
    {
        "id": "MEDQA_004",
        "question": "A 21-year-old student presents with periumbilical abdominal pain migrating to the right lower quadrant, nausea, anorexia, and tenderness at McBurney's point with guarding. Leukocytosis (14,500/uL).",
        "gold_diagnosis": "Appendicitis",
        "options": {"A": "Appendicitis", "B": "Meckel Diverticulitis", "C": "Crohn Disease", "D": "Renal Colic"}
    },
    {
        "id": "MEDQA_005",
        "question": "A 48-year-old female presents with severe epigastric pain radiating to the mid-back, relieved by leaning forward. Repeated vomiting. Serum lipase 1,200 U/L and amylase 850 U/L.",
        "gold_diagnosis": "Acute Pancreatitis",
        "options": {"A": "Acute Pancreatitis", "B": "Cholecystitis", "C": "Peptic Ulcer Disease", "D": "Mesenteric Ischemia"}
    },
    {
        "id": "MEDQA_006",
        "question": "A 68-year-old male presents with shaking chills, fever (39.2 C), productive cough with rust-colored sputum, bronchial breath sounds, and right lower lobe consolidation on chest radiograph.",
        "gold_diagnosis": "Pneumonia",
        "options": {"A": "Pneumonia", "B": "Pulmonary Embolism", "C": "Bronchiectasis", "D": "Lung Abscess"}
    },
    {
        "id": "MEDQA_007",
        "question": "A 28-year-old female presents with chronic diarrhea, flatulence, weight loss, iron deficiency anemia, and dermatitis herpetiformis. Strongly positive anti-tTG IgA serology.",
        "gold_diagnosis": "Celiac Disease",
        "options": {"A": "Celiac Disease", "B": "Irritable Bowel Syndrome", "C": "Lactose Intolerance", "D": "Ulcerative Colitis"}
    },
    {
        "id": "MEDQA_008",
        "question": "A 35-year-old female presents with dry cough, fatigue, painful erythema nodosum on shins, and bilateral hilar lymphadenopathy on chest X-ray. Elevated serum ACE and non-caseating granulomas on transbronchial biopsy.",
        "gold_diagnosis": "Sarcoidosis",
        "options": {"A": "Sarcoidosis", "B": "Tuberculosis", "C": "Histoplasmosis", "D": "Lymphoma"}
    },
    {
        "id": "MEDQA_009",
        "question": "A 63-year-old male with chronic hypertension presents with sudden tearing chest pain radiating to the back between the shoulder blades. Blood pressure asymmetry between arms (182/98 right vs 144/86 left). Normal troponin. CT angiography shows intimal flap in ascending aorta.",
        "gold_diagnosis": "Aortic Dissection",
        "options": {"A": "Aortic Dissection", "B": "Myocardial Infarction", "C": "Pericarditis", "D": "Esophageal Rupture"}
    },
    {
        "id": "MEDQA_010",
        "question": "A 28-year-old male brought to ED following motorcycle crash. Severe respiratory distress, cyanosis, distended neck veins, tracheal deviation to the left, absent breath sounds over the right hemithorax with hyperresonance. Severe obstructive shock (BP 76/44).",
        "gold_diagnosis": "Tension Pneumothorax",
        "options": {"A": "Tension Pneumothorax", "B": "Cardiac Tamponade", "C": "Hemothorax", "D": "Flail Chest"}
    },
    {
        "id": "MEDQA_011",
        "question": "A 17-year-old high school athlete collapses during basketball practice. Family history of sudden cardiac death in his uncle. Physical exam reveals a harsh crescendo-decrescendo systolic murmur at the left sternal border that increases in intensity with Valsalva maneuver.",
        "gold_diagnosis": "Hypertrophic Cardiomyopathy",
        "options": {"A": "Hypertrophic Cardiomyopathy", "B": "Aortic Stenosis", "C": "Mitral Valve Prolapse", "D": "Dilated Cardiomyopathy"}
    },
    {
        "id": "MEDQA_012",
        "question": "A 42-year-old male with a history of intravenous drug use presents with persistent low-grade fever, malaise, new regurgitant heart murmur, and tender subcutaneous nodules on his finger pads (Osler nodes). Multiple blood cultures grow Staphylococcus aureus.",
        "gold_diagnosis": "Infective Endocarditis",
        "options": {"A": "Infective Endocarditis", "B": "Acute Rheumatic Fever", "C": "Systemic Lupus Erythematosus", "D": "Atrial Myxoma"}
    },
    {
        "id": "MEDQA_013",
        "question": "A 29-year-old female presents with sudden painful visual loss in her right eye (optic neuritis), followed by electric shock sensations radiating down her spine upon neck flexion (Lhermitte sign). Brain MRI demonstrates periventricular white matter demyelinating lesions (Dawson fingers).",
        "gold_diagnosis": "Multiple Sclerosis",
        "options": {"A": "Multiple Sclerosis", "B": "Neuromyelitis Optica", "C": "Acute Disseminated Encephalomyelitis", "D": "Systemic Sclerosis"}
    },
    {
        "id": "MEDQA_014",
        "question": "A 68-year-old man presents with a 6-month history of an asymmetric resting pill-rolling tremor in his right hand. Physical exam reveals cogwheel rigidity of the upper extremities, bradykinesia, and a shuffling gait with reduced arm swing. No history of neuroleptic use.",
        "gold_diagnosis": "Parkinson Disease",
        "options": {"A": "Parkinson Disease", "B": "Essential Tremor", "C": "Progressive Supranuclear Palsy", "D": "Multiple System Atrophy"}
    },
    {
        "id": "MEDQA_015",
        "question": "A 34-year-old female presents with fluctuating double vision (diplopia) and drooping of both eyelids (ptosis) that worsens significantly toward the evening. Repetitive nerve stimulation shows a decremental compound muscle action potential. Serum anti-AChR antibodies are strongly positive.",
        "gold_diagnosis": "Myasthenia Gravis",
        "options": {"A": "Myasthenia Gravis", "B": "Lambert-Eaton Syndrome", "C": "Botulism", "D": "Guillain-Barre Syndrome"}
    },
    {
        "id": "MEDQA_016",
        "question": "A 56-year-old male presents with progressive asymmetric muscle weakness in his hands, involuntary muscle twitches (fasciculations), and difficulty swallowing. Neurological exam reveals a combination of upper motor neuron signs (spasticity, hyperreflexia) and lower motor neuron signs (atrophy, fasciculations) with no sensory loss.",
        "gold_diagnosis": "Amyotrophic Lateral Sclerosis",
        "options": {"A": "Amyotrophic Lateral Sclerosis", "B": "Cervical Spondylotic Myelopathy", "C": "Spinal Muscular Atrophy", "D": "Multifocal Motor Neuropathy"}
    },
    {
        "id": "MEDQA_017",
        "question": "An 18-year-old college student presents with acute high fever (39.8 C), severe throbbing headache, photophobia, nuchal rigidity, and positive Kernig and Brudzinski signs. Lumbar puncture demonstrates cloudy CSF with opening pressure 320 mm H2O, WBC 4,200/uL (95% neutrophils), markedly elevated protein (280 mg/dL), and decreased glucose (18 mg/dL).",
        "gold_diagnosis": "Bacterial Meningitis",
        "options": {"A": "Bacterial Meningitis", "B": "Viral Meningitis", "C": "Cryptococcal Meningitis", "D": "Herpes Simplex Encephalitis"}
    },
    {
        "id": "MEDQA_018",
        "question": "A 23-year-old male develops symmetric ascending muscle weakness beginning in his lower extremities and progressing over 4 days to involve his hands. Deep tendon reflexes are absent throughout. He had an episode of bloody diarrhea (Campylobacter jejuni enteritis) 3 weeks ago. Lumbar puncture reveals albuminocytologic dissociation (elevated protein with normal cell count).",
        "gold_diagnosis": "Guillain-Barre Syndrome",
        "options": {"A": "Guillain-Barre Syndrome", "B": "Chronic Inflammatory Demyelinating Polyneuropathy", "C": "Poliomyelitis", "D": "Transverse Myelitis"}
    },
    {
        "id": "MEDQA_019",
        "question": "A 45-year-old overweight female presents with severe colicky right upper quadrant abdominal pain that began after eating fried chicken. She has positive Murphy sign (inspiratory arrest on palpation of RUQ) and fever. Abdominal ultrasound demonstrates gallbladder wall thickening (5 mm), pericholecystic fluid, and acoustic shadowing from gallstones in the cystic duct.",
        "gold_diagnosis": "Acute Cholecystitis",
        "options": {"A": "Acute Cholecystitis", "B": "Biliary Colic", "C": "Acute Cholangitis", "D": "Peptic Ulcer Disease"}
    },
    {
        "id": "MEDQA_020",
        "question": "A 26-year-old female presents with chronic recurrent bloody diarrhea, lower abdominal cramping, tenesmus, and urgency. Colonoscopy demonstrates continuous mucosal erythema, friability, and pseudopolyps beginning at the rectum and extending proximally to the splenic flexure without skip lesions. Biopsies reveal crypt abscesses with neutrophil infiltration limited to the mucosa and submucosa.",
        "gold_diagnosis": "Ulcerative Colitis",
        "options": {"A": "Ulcerative Colitis", "B": "Crohn Disease", "C": "Infectious Colitis", "D": "Ischemic Colitis"}
    },
    {
        "id": "MEDQA_021",
        "question": "A 22-year-old male presents with crampy right lower quadrant pain, chronic non-bloody diarrhea, low-grade fever, and unintended 6 kg weight loss. Colonoscopy demonstrates patchy ulcerations with normal intervening mucosa (skip lesions), cobblestoning, and aphthous ulcers in the terminal ileum. Biopsy shows transmural active inflammation with non-caseating granulomas.",
        "gold_diagnosis": "Crohn Disease",
        "options": {"A": "Crohn Disease", "B": "Ulcerative Colitis", "C": "Celiac Disease", "D": "Irritable Bowel Syndrome"}
    },
    {
        "id": "MEDQA_022",
        "question": "A 72-year-old hospitalized female develops watery profuse diarrhea (8-10 episodes/day), fever, lower abdominal tenderness, and marked leukocytosis (24,000/uL) 6 days after receiving IV clindamycin for cellulitis. Stool PCR test for glutamate dehydrogenase and toxin B is positive. Pseudomembranous plaques are visualized on sigmoidoscopy.",
        "gold_diagnosis": "Clostridioides difficile Colitis",
        "options": {"A": "Clostridioides difficile Colitis", "B": "Ischemic Colitis", "C": "Cytomegalovirus Colitis", "D": "Microscopic Colitis"}
    },
    {
        "id": "MEDQA_023",
        "question": "A 65-year-old male presents with acute severe left lower quadrant abdominal pain, fever (38.6 C), anorexia, and obstipation. Physical exam demonstrates localized LLQ tenderness with rebound. Contrast-enhanced CT scan of the abdomen reveals focal colonic wall thickening, pericolic fat stranding, and multiple inflamed diverticula in the sigmoid colon without perforation.",
        "gold_diagnosis": "Acute Diverticulitis",
        "options": {"A": "Acute Diverticulitis", "B": "Colorectal Carcinoma", "C": "Ischemic Colitis", "D": "Appendicitis"}
    },
    {
        "id": "MEDQA_024",
        "question": "A 52-year-old male with a history of alcohol use disorder and cirrhosis presents with massive painless hematemesis and melena. He is hypotensive (BP 82/46), tachycardic (HR 128), and lethargic. Stigmata of chronic liver disease include spider angiomas, palmar erythema, and caput medusae. Emergency upper endoscopy reveals actively spurting serpiginous dilated veins in the distal esophagus.",
        "gold_diagnosis": "Esophageal Variceal Hemorrhage",
        "options": {"A": "Esophageal Variceal Hemorrhage", "B": "Mallory-Weiss Tear", "C": "Bleeding Duodenal Ulcer", "D": "Dieulafoy Lesion"}
    },
    {
        "id": "MEDQA_025",
        "question": "A 36-year-old male presents with acute hematemesis following a prolonged episode of severe non-bloody retching and vomiting after binge drinking alcohol. Vital signs are stable. Upper endoscopy reveals multiple longitudinal, superficial mucosal lacerations located at the gastroesophageal junction.",
        "gold_diagnosis": "Mallory-Weiss Syndrome",
        "options": {"A": "Mallory-Weiss Syndrome", "B": "Boerhaave Syndrome", "C": "Esophageal Varices", "D": "Gastric Adenocarcinoma"}
    },
    {
        "id": "MEDQA_026",
        "question": "A 60-year-old male with severe dehydration from infectious gastroenteritis presents with oliguria. Blood urea nitrogen is 68 mg/dL, serum creatinine is 2.8 mg/dL (BUN:Cr ratio > 20:1), and urine sodium is 12 mEq/L with fractional excretion of sodium (FENa) < 1%. Urinalysis shows high specific gravity (> 1.025) and transparent hyaline casts without muddy brown casts.",
        "gold_diagnosis": "Prerenal Acute Kidney Injury",
        "options": {"A": "Prerenal Acute Kidney Injury", "B": "Acute Tubular Necrosis", "C": "Acute Interstitial Nephritis", "D": "Postrenal Azotemia"}
    },
    {
        "id": "MEDQA_027",
        "question": "An 8-year-old boy presents with periorbital puffiness, dark cola-colored urine, and mild hypertension (135/88) 2 weeks after recovering from a culture-confirmed Group A streptococcal skin impetigo. Urinalysis demonstrates dysmorphic red blood cells, RBC casts, and moderate proteinuria. Serum complement C3 is markedly decreased, and antistreptolysin O (ASO) titer is high.",
        "gold_diagnosis": "Poststreptococcal Glomerulonephritis",
        "options": {"A": "Poststreptococcal Glomerulonephritis", "B": "IgA Nephropathy", "C": "Minimal Change Disease", "D": "Membranoproliferative Glomerulonephritis"}
    },
    {
        "id": "MEDQA_028",
        "question": "A 22-year-old male presents with visible gross hematuria that started 2 days after developing a sore throat, runny nose, and low-grade fever (concurrent synpharyngitic hematuria). He had a similar self-resolving episode 6 months ago during an upper respiratory illness. Serum C3 and C4 complement levels are completely normal. Renal biopsy reveals mesangial proliferation with intense mesangial IgA immune deposits.",
        "gold_diagnosis": "IgA Nephropathy",
        "options": {"A": "IgA Nephropathy", "B": "Poststreptococcal Glomerulonephritis", "C": "Alport Syndrome", "D": "Goodpasture Syndrome"}
    },
    {
        "id": "MEDQA_029",
        "question": "A 5-year-old boy presents with marked generalized pitting edema, periorbital swelling, and abdominal distension with ascites. Blood pressure is normal (96/60). Urinalysis shows 4+ proteinuria without hematuria or red blood cell casts. 24-hour urine collection reveals 4.8 g/day of selective albuminuria. Serum albumin is 1.6 g/dL and total cholesterol is 380 mg/dL. Renal light microscopy is normal; electron microscopy demonstrates diffuse podocyte foot process effacement.",
        "gold_diagnosis": "Minimal Change Disease",
        "options": {"A": "Minimal Change Disease", "B": "Focal Segmental Glomerulosclerosis", "C": "Membranous Nephropathy", "D": "Poststreptococcal Glomerulonephritis"}
    },
    {
        "id": "MEDQA_030",
        "question": "A 44-year-old male presents with excruciating right flank pain that radiates to his groin and ipsilateral testicle, accompanied by gross hematuria, nausea, and vomiting. He cannot find a comfortable position and paces the room. Non-contrast helical CT scan of the abdomen and pelvis confirms a 6 mm hyperdense calcified stone impacted at the right ureterovesical junction with mild upstream hydronephrosis.",
        "gold_diagnosis": "Nephrolithiasis",
        "options": {"A": "Nephrolithiasis", "B": "Acute Pyelonephritis", "C": "Renal Cell Carcinoma", "D": "Acute Appendicitis"}
    },
    {
        "id": "MEDQA_031",
        "question": "A 68-year-old woman with advanced osteoarthritis of the knees presents with acute confusion, lethargy, severe constipation, abdominal pain, and polyuria. Corrected serum calcium is markedly elevated at 14.2 mg/dL (reference 8.5-10.2). Intact parathyroid hormone (PTH) is suppressed at 4 pg/mL. A bone scan reveals multiple osteolytic lesions in the thoracic spine and pelvis. Serum protein electrophoresis shows an IgG kappa monoclonal spike.",
        "gold_diagnosis": "Hypercalcemia of Malignancy",
        "options": {"A": "Hypercalcemia of Malignancy", "B": "Primary Hyperparathyroidism", "C": "Sarcoidosis", "D": "Milk-Alkali Syndrome"}
    },
    {
        "id": "MEDQA_032",
        "question": "A 46-year-old male presents with profound weakness, paresthesias in his feet, muscle cramps, and palpitations after starting high-dose furosemide for congestive heart failure. Serum potassium is severely low at 2.4 mEq/L. 12-lead electrocardiogram demonstrates flattened T waves, ST-segment depression, and prominent U waves in precordial leads V2-V4.",
        "gold_diagnosis": "Hypokalemia",
        "options": {"A": "Hypokalemia", "B": "Hypercalcemia", "C": "Hyperkalemia", "D": "Hypomagnesemia"}
    },
    {
        "id": "MEDQA_033",
        "question": "A 62-year-old female with stage 5 chronic kidney disease misses three consecutive hemodialysis sessions. She presents with severe muscle weakness, bradycardia (HR 44 bpm), and ascending flaccid paralysis. Serum potassium is critically elevated at 7.6 mEq/L. Electrocardiogram demonstrates tall, symmetrical, peaked T waves, prolongation of the PR interval, loss of P waves, and widening of the QRS complex into a sine-wave pattern.",
        "gold_diagnosis": "Hyperkalemia",
        "options": {"A": "Hyperkalemia", "B": "Hypokalemia", "C": "Hypocalcemia", "D": "Hypercalcemia"}
    },
    {
        "id": "MEDQA_034",
        "question": "A 24-year-old female presents with heat intolerance, profuse sweating, unintentional 7 kg weight loss despite voracious appetite, palpitation, and frequent bowel movements. Physical examination reveals warm moist skin, fine bilateral hand tremor, bilateral proptosis (exophthalmos), pretibial myxedema, and a diffusely enlarged non-tender thyroid gland with an audible bruit. TSH is undetectable (< 0.01 uIU/mL), and free T4 is markedly elevated at 3.8 ng/dL. Thyroid stimulating immunoglobulin (TSI) is strongly positive.",
        "gold_diagnosis": "Graves Disease",
        "options": {"A": "Graves Disease", "B": "Hashimoto Thyroiditis", "C": "Toxic Multinodular Goiter", "D": "Subacute Thyroiditis"}
    },
    {
        "id": "MEDQA_035",
        "question": "A 48-year-old female presents with worsening fatigue, generalized cold intolerance, constipation, unintentional weight gain, dry coarse skin, and periorbital puffiness. Physical exam reveals a firm, rubbery, painless, diffusely enlarged goiter and delayed relaxation phase of deep tendon reflexes. Serum TSH is elevated at 34 mIU/L, and free T4 is reduced at 0.4 ng/dL. High titers of anti-thyroid peroxidase (anti-TPO) and anti-thyroglobulin antibodies are present.",
        "gold_diagnosis": "Hashimoto Thyroiditis",
        "options": {"A": "Hashimoto Thyroiditis", "B": "Graves Disease", "C": "Subacute Granulomatous Thyroiditis", "D": "Riedel Thyroiditis"}
    },
    {
        "id": "MEDQA_036",
        "question": "A 38-year-old female presents with central obesity, a rounded 'moon facies', a dorsocervical fat pad ('buffalo hump'), prominent violaceous purple striae over her lower abdomen, easy bruising, proximal muscle weakness, and new-onset secondary hypertension (BP 164/98). 24-hour urinary free cortisol is four times the upper limit of normal. High-dose dexamethasone suppression test demonstrates suppression of plasma ACTH and cortisol. Pituitary MRI reveals a 7 mm microadenoma.",
        "gold_diagnosis": "Cushing Disease",
        "options": {"A": "Cushing Disease", "B": "Ectopic ACTH Syndrome", "C": "Adrenal Cortical Carcinoma", "D": "Exogenous Corticosteroid Use"}
    },
    {
        "id": "MEDQA_037",
        "question": "A 36-year-old female presents with progressive chronic fatigue, anorexia, lightheadedness on standing, salt craving, and hyperpigmentation of her skin creases, buccal mucosa, and palmar creases. Orthostatic vitals confirm significant postural hypotension (supine 112/70 -> standing 84/52). Laboratory evaluation reveals hyponatremia (126 mEq/L), hyperkalemia (5.8 mEq/L), and non-anion gap metabolic acidosis. Basal morning serum cortisol is low, and plasma ACTH is markedly elevated. Synthetic ACTH (cosyntropin) stimulation test fails to increase serum cortisol.",
        "gold_diagnosis": "Addison Disease",
        "options": {"A": "Addison Disease", "B": "Secondary Adrenal Insufficiency", "C": "Conn Syndrome", "D": "Pheochromocytoma"}
    },
    {
        "id": "MEDQA_038",
        "question": "A 42-year-old male presents with paroxysmal episodes of pounding headache, profuse diaphoresis, and palpitations (the classic triad). During these episodes, blood pressure surges to 220/120 mmHg. 24-hour urine collection reveals markedly elevated free metanephrine and normetanephrine levels. Abdominal MRI demonstrates a 4.5 cm hyperintense mass in the right adrenal medulla.",
        "gold_diagnosis": "Pheochromocytoma",
        "options": {"A": "Pheochromocytoma", "B": "Renal Artery Stenosis", "C": "Essential Hypertension", "D": "Panic Disorder"}
    },
    {
        "id": "MEDQA_039",
        "question": "A 45-year-old female presents with persistent severe generalized pruritus (itching) and fatigue. Physical examination reveals mild jaundice and xanthomas on her eyelids. Laboratory evaluation demonstrates alkaline phosphatase 540 U/L, total bilirubin 3.2 mg/dL, and normal AST/ALT. Serum antimitochondrial antibodies (AMA) are strongly positive (> 1:80 titer). Liver biopsy confirms chronic non-suppurative granulomatous destructive cholangitis of interlobular bile ducts.",
        "gold_diagnosis": "Primary Biliary Cholangitis",
        "options": {"A": "Primary Biliary Cholangitis", "B": "Primary Sclerosing Cholangitis", "C": "Autoimmune Hepatitis", "D": "Choledocholithiasis"}
    },
    {
        "id": "MEDQA_040",
        "question": "A 29-year-old male with a 6-year history of ulcerative colitis presents with progressive fatigue, jaundice, and intermittent right upper quadrant abdominal discomfort. Serum alkaline phosphatase is elevated at 480 U/L, and serum p-ANCA is positive. Magnetic resonance cholangiopancreatography (MRCP) reveals multifocal strictures and segmental dilatations of intrahepatic and extrahepatic bile ducts, producing a characteristic 'beaded' appearance.",
        "gold_diagnosis": "Primary Sclerosing Cholangitis",
        "options": {"A": "Primary Sclerosing Cholangitis", "B": "Primary Biliary Cholangitis", "C": "Ascending Cholangitis", "D": "Biliary Atresia"}
    },
    {
        "id": "MEDQA_041",
        "question": "A 55-year-old male presents with recurrent acute episodes of excruciating pain, swelling, erythema, and warmth in his right first metatarsophalangeal (MTP) joint (podagra) following a seafood and beer dinner. Joint aspiration reveals cloudy synovial fluid with 42,000 WBCs/uL (predominantly neutrophils). Polarizing light microscopy demonstrates strongly negative birefringent needle-shaped crystals under compensated polarized light.",
        "gold_diagnosis": "Gout",
        "options": {"A": "Gout", "B": "Pseudogout", "C": "Septic Arthritis", "D": "Rheumatoid Arthritis"}
    },
    {
        "id": "MEDQA_042",
        "question": "A 74-year-old female presents with acute pain, warmth, and massive swelling in her right knee. Plain radiograph of the knee reveals linear calcification of the articular cartilage (chondrocalcinosis) involving the menisci. Arthrocentesis yields turbid fluid containing intracellular and extracellular rhomboid-shaped crystals that exhibit weakly positive birefringence under polarized microscopy.",
        "gold_diagnosis": "Pseudogout",
        "options": {"A": "Pseudogout", "B": "Gout", "C": "Osteoarthritis", "D": "Hydroxyapatite Deposition Disease"}
    },
    {
        "id": "MEDQA_043",
        "question": "A 32-year-old female presents with symmetric polyarthritis involving the PIP and MCP joints of both hands, wrists, and knees for 4 months, accompanied by morning stiffness lasting > 1.5 hours. Hands show ulnar deviation and swan-neck deformities. Plain radiographs demonstrate periarticular osteopenia and marginal bony erosions. Serum rheumatoid factor (RF) and anti-cyclic citrullinated peptide (anti-CCP) antibodies are strongly positive.",
        "gold_diagnosis": "Rheumatoid Arthritis",
        "options": {"A": "Rheumatoid Arthritis", "B": "Osteoarthritis", "C": "Psoriatic Arthritis", "D": "Systemic Lupus Erythematosus"}
    },
    {
        "id": "MEDQA_044",
        "question": "A 26-year-old African American female presents with low-grade fever, fatigue, painful oral mucosal ulcers, pleuritic chest pain, symmetric arthralgias in both wrists, and an erythematous maculopapular rash across the bridge of her nose and cheeks that spares the nasolabial folds (malar rash). Urinalysis shows 2+ proteinuria and cellular casts. Antinuclear antibody (ANA) is positive at 1:640 with speckled pattern; serum anti-dsDNA and anti-Smith antibodies are strongly positive. Serum complement levels C3 and C4 are markedly reduced.",
        "gold_diagnosis": "Systemic Lupus Erythematosus",
        "options": {"A": "Systemic Lupus Erythematosus", "B": "Mixed Connective Tissue Disease", "C": "Dermatomyositis", "D": "Rosacea"}
    },
    {
        "id": "MEDQA_045",
        "question": "A 25-year-old male presents with persistent low back pain and morning stiffness for 8 months that improves with exercise and hot showers but does not improve with rest. He has bilateral alternating buttock pain and reduced chest expansion (2 cm). Lumbar spine radiograph demonstrates bilateral symmetrical sacroiliitis, squaring of vertebral bodies, and marginal syndesmophytes with fusion (bamboo spine). Genetic testing confirms HLA-B27 positivity.",
        "gold_diagnosis": "Ankylosing Spondylitis",
        "options": {"A": "Ankylosing Spondylitis", "B": "Lumbar Disc Herniation", "C": "Diffuse Idiopathic Skeletal Hyperostosis", "D": "Psoriatic Arthritis"}
    },
    {
        "id": "MEDQA_046",
        "question": "A 71-year-old female presents with a new severe unilateral throbbing right temporal headache, pain in her jaw muscles while chewing (jaw claudication), and scalp tenderness when brushing her hair. Yesterday she developed sudden temporary blurriness in her right eye (amaurosis fugax). Her right superficial temporal artery is tender, swollen, and pulseless. Erythrocyte sedimentation rate (ESR) is markedly elevated at 112 mm/hr, and C-reactive protein is 48 mg/L.",
        "gold_diagnosis": "Temporal Arteritis",
        "options": {"A": "Temporal Arteritis", "B": "Trigeminal Neuralgia", "C": "Migraine Headache", "D": "Cluster Headache"}
    },
    {
        "id": "MEDQA_047",
        "question": "A 48-year-old male presents with chronic recurrent bloody purulent nasal discharge, nasal bridge collapse (saddle-nose deformity), recurrent otitis media, chronic cough, and hemoptysis. Chest CT reveals bilateral cavitary nodular pulmonary infiltrates. Serum creatinine is elevated at 2.6 mg/dL with active urinary sediment (RBC casts). Cytoplasmic antineutrophil cytoplasmic antibodies (c-ANCA) targeting proteinase 3 (PR3) are strongly positive.",
        "gold_diagnosis": "Granulomatosis with Polyangiitis",
        "options": {"A": "Granulomatosis with Polyangiitis", "B": "Microscopic Polyangiitis", "C": "Goodpasture Syndrome", "D": "Eosinophilic Granulomatosis with Polyangiitis"}
    },
    {
        "id": "MEDQA_048",
        "question": "A 28-year-old male presents with sudden acute hemoptysis and dyspnea, accompanied by dark tea-colored urine and oliguria. Chest radiograph shows bilateral alveolar infiltrates consistent with diffuse alveolar hemorrhage. Serum creatinine is 4.1 mg/dL. Renal biopsy shows linear deposition of IgG and C3 along the glomerular basement membrane (GBM) on direct immunofluorescence. Serum anti-GBM antibodies targeting the alpha-3 chain of type IV collagen are positive.",
        "gold_diagnosis": "Goodpasture Syndrome",
        "options": {"A": "Goodpasture Syndrome", "B": "Granulomatosis with Polyangiitis", "C": "Systemic Lupus Erythematosus", "D": "Microscopic Polyangiitis"}
    },
    {
        "id": "MEDQA_049",
        "question": "A 35-year-old male presents with sudden onset of high fever, malaise, extensive painful targetoid cutaneous lesions, and widespread flaccid blistering bullae involving 25% of his body surface area, with full-thickness epidermal detachment upon gentle mechanical pressure (positive Nikolsky sign). Lesions severely involve oral, labial, conjunctival, and genital mucous membranes. Symptoms began 8 days after starting allopurinol for gout.",
        "gold_diagnosis": "Stevens-Johnson Syndrome",
        "options": {"A": "Stevens-Johnson Syndrome", "B": "Urticaria", "C": "Fixed Drug Eruption", "D": "Pemphigus Vulgaris"}
    },
    {
        "id": "MEDQA_050",
        "question": "A 24-year-old male is stung by a yellow jacket wasp. Within minutes he develops generalized urticaria, intense pruritus, stridor, wheezing, angioedema of lips and tongue, and profound hypotension (78/42 mmHg).",
        "gold_diagnosis": "Anaphylactic Shock",
        "options": {"A": "Anaphylactic Shock", "B": "Vasovagal Syncope", "C": "Asthma Exacerbation", "D": "Septic Shock"}
    }
]


# -------------------------------------------------------------
# 2. 50 Gold-Standard MedBullets Applied Clinical Residency Cases
# -------------------------------------------------------------
MEDBULLETS_50_CASES = [
    {
        "id": "MEDBULLET_001",
        "question": "A 45-year-old female presents with persistent fatigue and progressive pruritus worse at night. Physical examination reveals mild jaundice and periorbital xanthelasmas. Serum alkaline phosphatase is 480 U/L, and antimitochondrial antibody (AMA) titer is 1:160. What is the most appropriate first-line medical therapy to delay disease progression?",
        "gold_diagnosis": "Ursodeoxycholic Acid",
        "options": {"A": "Ursodeoxycholic Acid", "B": "Prednisone", "C": "Methotrexate", "D": "Infliximab"}
    },
    {
        "id": "MEDBULLET_002",
        "question": "A 29-year-old male presents with symmetric lower extremity weakness that began 3 days ago and has now ascended to involve both thighs and hands. On examination, patellar and Achilles reflexes are completely absent bilaterally. Vital signs show mild orthostatic lability. Lumbar puncture demonstrates elevated CSF protein (145 mg/dL) with normal WBC count (2 cells/uL). What is the definitive initial management?",
        "gold_diagnosis": "Intravenous Immunoglobulin",
        "options": {"A": "Intravenous Immunoglobulin", "B": "High-dose Oral Dexamethasone", "C": "Interferon Beta", "D": "Pyridostigmine"}
    },
    {
        "id": "MEDBULLET_003",
        "question": "A 73-year-old woman presents with severe right-sided temporal headache, jaw pain when chewing meals, and double vision. Her erythrocyte sedimentation rate is 105 mm/hr. While arranging a temporal artery biopsy, what is the most urgent next step to prevent irreversible vision loss?",
        "gold_diagnosis": "High-Dose Corticosteroids",
        "options": {"A": "High-Dose Corticosteroids", "B": "Observation Until Biopsy Confirms", "C": "Low-Dose Aspirin Monotherapy", "D": "Intravenous Methotrexate"}
    },
    {
        "id": "MEDBULLET_004",
        "question": "A 38-year-old male with episodic headaches, diaphoresis, and palpitations is found to have a 4.2 cm adrenal mass and 4-fold elevation in 24-hour urine fractionated metanephrines. Prior to surgical resection, what pharmacologic agent must be initiated first to prevent hypertensive crisis during induction?",
        "gold_diagnosis": "Phenoxybenzamine",
        "options": {"A": "Phenoxybenzamine", "B": "Propranolol", "C": "Metoprolol", "D": "Labetalol"}
    },
    {
        "id": "MEDBULLET_005",
        "question": "An 18-year-old male presents with progressive hand tremor, dysarthria, involuntary grimacing, and elevated transaminases. Slit-lamp examination reveals brownish-green rings in the peripheral descemet membrane of both corneas. Serum ceruloplasmin is 8 mg/dL (normal 20-40). What is the underlying diagnosis?",
        "gold_diagnosis": "Wilson Disease",
        "options": {"A": "Wilson Disease", "B": "Hemochromatosis", "C": "Alpha-1 Antitrypsin Deficiency", "D": "Huntington Disease"}
    },
    {
        "id": "MEDBULLET_006",
        "question": "A 66-year-old male presents with deep thoracic back pain, fatigue, and recurrent pneumococcal infections. Hemoglobin is 9.2 g/dL, serum calcium is 11.8 mg/dL, and creatinine is 2.2 mg/dL. Serum protein electrophoresis reveals a dense monoclonal gamma peak, and skeletal survey demonstrates multiple punched-out lytic bone lesions. What is the diagnosis?",
        "gold_diagnosis": "Multiple Myeloma",
        "options": {"A": "Multiple Myeloma", "B": "Monoclonal Gammopathy of Undetermined Significance", "C": "Osteolytic Metastatic Prostatic Carcinoma", "D": "Waldenstrom Macroglobulinemia"}
    },
    {
        "id": "MEDBULLET_007",
        "question": "A family is brought to the emergency department from an apartment with a faulty furnace during a winter storm. All present with throbbing headache, dizziness, nausea, and confusion. Pulse oximetry reads 99% on room air, but blood co-oximetry confirms carboxyhemoglobin saturation of 28%. What is the immediate treatment of choice?",
        "gold_diagnosis": "100% Normobaric Oxygen",
        "options": {"A": "100% Normobaric Oxygen", "B": "Intravenous Hydroxocobalamin", "C": "Amyl Nitrite Inhalation", "D": "Methylene Blue"}
    },
    {
        "id": "MEDBULLET_008",
        "question": "A 34-year-old female presents with fluctuating confusion, fever of 38.8 C, petechial rash, and profound fatigue. Laboratory tests show platelet count 14,000/uL, hemoglobin 7.8 g/dL with prominent schistocytes on peripheral blood smear, elevated LDH, and normal PT/PTT. ADAMTS13 activity is severely deficient (< 5%). What is the primary life-saving intervention?",
        "gold_diagnosis": "Therapeutic Plasma Exchange",
        "options": {"A": "Therapeutic Plasma Exchange", "B": "Platelet Transfusion", "C": "Intravenous Heparin Infusion", "D": "Broad-Spectrum Antibiotics Alone"}
    },
    {
        "id": "MEDBULLET_009",
        "question": "A 24-year-old male presents with chronic insidious lower back stiffness that is worst in the morning and relieved by basketball practice. Pelvic radiograph reveals sclerosis and pseudo-widening of the sacroiliac joints. Testing confirms HLA-B27 positivity. What is the first-line pharmacologic treatment?",
        "gold_diagnosis": "Nonsteroidal Anti-Inflammatory Drugs",
        "options": {"A": "Nonsteroidal Anti-Inflammatory Drugs", "B": "Oral Prednisone", "C": "Methotrexate", "D": "Opioid Analgesics"}
    },
    {
        "id": "MEDBULLET_010",
        "question": "A 31-year-old female presents with left lower extremity deep vein thrombosis. Her obstetric history is notable for three consecutive unexplained first-trimester spontaneous abortions. Coagulation panel shows a prolonged PTT that fails to correct with 1:1 normal plasma mixing, and anticardiolipin IgG antibodies are strongly positive. What condition is responsible?",
        "gold_diagnosis": "Antiphospholipid Syndrome",
        "options": {"A": "Antiphospholipid Syndrome", "B": "Factor V Leiden Mutation", "C": "Protein C Deficiency", "D": "Disseminated Intravascular Coagulation"}
    },
    {
        "id": "MEDBULLET_011",
        "question": "A 26-year-old male smoker presents with coughing up blood (hemoptysis), dyspnea, and cola-colored urine. Serum creatinine is 3.4 mg/dL. Direct immunofluorescence of a percutaneous renal biopsy demonstrates smooth, linear deposition of IgG along the glomerular basement membrane. What autoantibody targets this tissue?",
        "gold_diagnosis": "Anti-GBM Antibodies",
        "options": {"A": "Anti-GBM Antibodies", "B": "Anti-Neutrophil Cytoplasmic Antibodies", "C": "Anti-Double Stranded DNA", "D": "Anti-Streptolysin O"}
    },
    {
        "id": "MEDBULLET_012",
        "question": "A 7-year-old boy presents with palpable purpuric lesions over his buttocks and bilateral lower extremities, intermittent colicky abdominal pain, and swollen ankles following an upper respiratory infection. Platelet count is normal (260,000/uL) and coagulation profile is normal. Skin biopsy reveals leukocytoclastic vasculitis with prominent IgA deposition. What is the diagnosis?",
        "gold_diagnosis": "Henoch-Schonlein Purpura",
        "options": {"A": "Henoch-Schonlein Purpura", "B": "Immune Thrombocytopenic Purpura", "C": "Thrombotic Thrombocytopenic Purpura", "D": "Meningococcemia"}
    },
    {
        "id": "MEDBULLET_013",
        "question": "A 52-year-old male presents with chronic fatigue, loss of libido, and generalized slate-gray hyperpigmentation of sun-exposed skin. Fasting glucose is 210 mg/dL, AST 72 U/L, and transferrin saturation is markedly elevated at 74% with serum ferritin 1,850 ng/mL. Genetic testing confirms homozygous C282Y mutation. What is the definitive maintenance therapy?",
        "gold_diagnosis": "Therapeutic Phlebotomy",
        "options": {"A": "Therapeutic Phlebotomy", "B": "Deferoxamine Chelation", "C": "Zinc Supplementation", "D": "Interferon Therapy"}
    },
    {
        "id": "MEDBULLET_014",
        "question": "A 36-year-old female experiences drooping of her left upper eyelid and progressive difficulty chewing food towards the end of meals. Administration of edrophonium produces immediate, transient restoration of muscle power. What is the primary mechanism of action of the first-line symptomatic maintenance drug?",
        "gold_diagnosis": "Acetylcholinesterase Inhibition",
        "options": {"A": "Acetylcholinesterase Inhibition", "B": "Direct Muscarinic Agonism", "C": "Voltage-Gated Calcium Channel Blockade", "D": "NMDA Receptor Blockade"}
    },
    {
        "id": "MEDBULLET_015",
        "question": "A 61-year-old chronic smoker presents with proximal lower extremity weakness that paradoxically improves with repetitive muscle exertion, accompanied by severe dry mouth and erectile dysfunction. Chest CT reveals a central perihilar pulmonary mass. Antibodies targeting presynaptic P/Q-type voltage-gated calcium channels are positive. What is the diagnosis?",
        "gold_diagnosis": "Lambert-Eaton Myasthenic Syndrome",
        "options": {"A": "Lambert-Eaton Myasthenic Syndrome", "B": "Myasthenia Gravis", "C": "Polymyositis", "D": "Guillain-Barre Syndrome"}
    },
    {
        "id": "MEDBULLET_016",
        "question": "A 49-year-old female presents with symmetric proximal muscle weakness making it hard to climb stairs or brush her hair. Examination reveals violaceous erythema of the upper eyelids with periorbital edema and flat-topped violaceous papules over the MCP and PIP joints. Serum creatine kinase is 3,200 U/L. What is the diagnosis?",
        "gold_diagnosis": "Dermatomyositis",
        "options": {"A": "Dermatomyositis", "B": "Polymyositis", "C": "Systemic Lupus Erythematosus", "D": "Psoriatic Arthritis"}
    },
    {
        "id": "MEDBULLET_017",
        "question": "A 43-year-old woman presents with episodes of finger pallor and cyanosis upon cold exposure, tight shiny skin over her fingers (sclerodactyly), esophageal dysphagia, and telangiectasias on her face. Anticentromere antibodies are strongly positive. What condition does this represent?",
        "gold_diagnosis": "Limited Systemic Sclerosis",
        "options": {"A": "Limited Systemic Sclerosis", "B": "Diffuse Cutaneous Systemic Sclerosis", "C": "Eosinophilic Fasciitis", "D": "Mixed Connective Tissue Disease"}
    },
    {
        "id": "MEDBULLET_018",
        "question": "A 44-year-old male with active chronic hepatitis B infection presents with fever, unintended 10 kg weight loss, severe postprandial abdominal pain, livedo reticularis, and right foot drop. Visceral angiography reveals multiple microaneurysms in the renal and mesenteric vasculature. What vasculitis is this?",
        "gold_diagnosis": "Polyarteritis Nodosa",
        "options": {"A": "Polyarteritis Nodosa", "B": "Granulomatosis with Polyangiitis", "C": "Microscopic Polyangiitis", "D": "Giant Cell Arteritis"}
    },
    {
        "id": "MEDBULLET_019",
        "question": "A 56-year-old chronic alcoholic male is brought to the hospital in an acutely confused state. Physical examination reveals horizontal nystagmus, bilateral lateral rectus palsy, and wide-based ataxic gait. What nutritional intervention must be administered immediately prior to any intravenous dextrose?",
        "gold_diagnosis": "Intravenous Thiamine",
        "options": {"A": "Intravenous Thiamine", "B": "Folic Acid", "C": "Vitamin B12", "D": "Pyridoxine"}
    },
    {
        "id": "MEDBULLET_020",
        "question": "During elective cholecystectomy under general anesthesia with halothane and succinylcholine, a 22-year-old male develops sudden jaw clenching (masseter spasm), generalized muscle rigidity, body temperature spiking to 41.5 C, and end-tidal CO2 doubling. What specific antagonist must be given immediately?",
        "gold_diagnosis": "Dantrolene",
        "options": {"A": "Dantrolene", "B": "Bromocriptine", "C": "Physostigmine", "D": "Atropine"}
    },
    {
        "id": "MEDBULLET_021",
        "question": "A 28-year-old psychiatric patient with acute psychosis is started on high-potency haloperidol. Three days later, he is found stuporous with severe 'lead-pipe' muscular rigidity, autonomic instability (temperature 39.9 C, BP 190/110), diaphoresis, and serum creatine kinase of 22,000 U/L. What is the diagnosis?",
        "gold_diagnosis": "Neuroleptic Malignant Syndrome",
        "options": {"A": "Neuroleptic Malignant Syndrome", "B": "Serotonin Syndrome", "C": "Malignant Hyperthermia", "D": "Catatonia"}
    },
    {
        "id": "MEDBULLET_022",
        "question": "A 33-year-old female taking sertraline for depression is prescribed tramadol for an acute back sprain. Within 12 hours she develops agitation, profuse diaphoresis, hyperactive bowel sounds, hyperthermia, tremor, and spontaneous sustained ankle clonus. What diagnosis accounts for this toxidrome?",
        "gold_diagnosis": "Serotonin Syndrome",
        "options": {"A": "Serotonin Syndrome", "B": "Neuroleptic Malignant Syndrome", "C": "Anticholinergic Toxicity", "D": "Sympathomimetic Overdose"}
    },
    {
        "id": "MEDBULLET_023",
        "question": "A 58-year-old male awakens with excruciating pain, erythema, and swelling in his left great toe after eating liver and drinking beer. Joint fluid aspirate reveals bright yellow needle-shaped crystals when parallel to the compensator axis (negative birefringence). What is the first-line acute therapy?",
        "gold_diagnosis": "Indomethacin",
        "options": {"A": "Indomethacin", "B": "Allopurinol", "C": "Febuxostat", "D": "Methotrexate"}
    },
    {
        "id": "MEDBULLET_024",
        "question": "A 78-year-old male presents with acute swelling and tenderness in his right knee. Knee radiography demonstrates linear calcification along the articular cartilage (chondrocalcinosis). Synovial fluid shows rhomboid crystals exhibiting positive birefringence. What is the primary constituent of these crystals?",
        "gold_diagnosis": "Calcium Pyrophosphate Dihydrate",
        "options": {"A": "Calcium Pyrophosphate Dihydrate", "B": "Monosodium Urate", "C": "Calcium Oxalate", "D": "Basic Calcium Phosphate"}
    },
    {
        "id": "MEDBULLET_025",
        "question": "A 65-year-old male notices his hat size has increased over the past 3 years and develops progressive bilateral sensorineural hearing loss. Serum alkaline phosphatase is 420 U/L with normal serum calcium, phosphate, and liver function. Radiographs show cortical thickening and osteolytic-osteoblastic mosaic patterns. What is the diagnosis?",
        "gold_diagnosis": "Paget Disease of Bone",
        "options": {"A": "Paget Disease of Bone", "B": "Osteosarcoma", "C": "Multiple Myeloma", "D": "Fibrous Dysplasia"}
    },
    {
        "id": "MEDBULLET_026",
        "question": "A 46-year-old woman with longstanding celiac disease presents with diffuse aching bone pain, proximal muscle weakness, and a waddling gait. Pelvic radiographs show bilateral symmetrical radiolucent bands perpendicular to the bone cortex (Looser zones/pseudofractures). Labs reveal low serum calcium, low phosphorus, and elevated PTH. What is the diagnosis?",
        "gold_diagnosis": "Osteomalacia",
        "options": {"A": "Osteomalacia", "B": "Osteoporosis", "C": "Osteopetrosis", "D": "Renal Osteodystrophy"}
    },
    {
        "id": "MEDBULLET_027",
        "question": "A 39-year-old female presents with round plethoric facies, central truncal obesity, thin skin with easy bruising, and wide purple abdominal striae. Overnight low-dose dexamethasone suppression fails to suppress morning cortisol, while high-dose dexamethasone test suppresses serum cortisol by > 50%. Pituitary MRI reveals a 5 mm microadenoma. What is the diagnosis?",
        "gold_diagnosis": "Cushing Disease",
        "options": {"A": "Cushing Disease", "B": "Ectopic ACTH Secreting Carcinoid", "C": "Adrenal Adenoma", "D": "Polycystic Ovary Syndrome"}
    },
    {
        "id": "MEDBULLET_028",
        "question": "A 47-year-old male presents with treatment-resistant hypertension (BP 178/104 on three antihypertensive agents) and intermittent muscle cramps. Serum potassium is 2.8 mEq/L. Plasma aldosterone concentration is elevated, and plasma renin activity is suppressed, yielding an aldosterone-to-renin ratio of 45. What is the diagnosis?",
        "gold_diagnosis": "Primary Hyperaldosteronism",
        "options": {"A": "Primary Hyperaldosteronism", "B": "Renal Artery Stenosis", "C": "Cushing Syndrome", "D": "Pheochromocytoma"}
    },
    {
        "id": "MEDBULLET_029",
        "question": "A 33-year-old female presents with worsening fatigue, dizziness, 8 kg weight loss, and craving for salty snacks. Physical exam demonstrates prominent hyperpigmentation of her knuckles, palmar creases, and gingival mucosa. Serum sodium is 128 mEq/L and potassium is 5.6 mEq/L. Serum cortisol fails to double following cosyntropin administration. What is the diagnosis?",
        "gold_diagnosis": "Addison Disease",
        "options": {"A": "Addison Disease", "B": "Secondary Adrenal Insufficiency", "C": "Hashimoto Encephalopathy", "D": "Syndrome of Inappropriate ADH"}
    },
    {
        "id": "MEDBULLET_030",
        "question": "A 41-year-old male complains of having to buy larger shoes and wedding ring, coarsening of facial features, increased spacing between teeth, and carpal tunnel syndrome. Fasting serum insulin-like growth factor 1 (IGF-1) is three times normal. Oral glucose tolerance test fails to suppress serum growth hormone. What is the diagnosis?",
        "gold_diagnosis": "Acromegaly",
        "options": {"A": "Acromegaly", "B": "Hypothyroidism", "C": "Paget Disease", "D": "Prolactinoma"}
    },
    {
        "id": "MEDBULLET_031",
        "question": "A 28-year-old female presents with anxiety, palpitations, heat intolerance, and bilateral exophthalmos. Examination reveals a smooth, diffusely enlarged, non-tender goiter with an audible thrill. Serum TSH is < 0.01 uIU/mL and free T4 is elevated. What autoantibody is pathognomonic for this disease?",
        "gold_diagnosis": "Thyroid-Stimulating Immunoglobulin",
        "options": {"A": "Thyroid-Stimulating Immunoglobulin", "B": "Anti-Thyroid Peroxidase", "C": "Anti-Thyroglobulin", "D": "Anti-Centromere"}
    },
    {
        "id": "MEDBULLET_032",
        "question": "A 51-year-old female presents with persistent lethargy, constipation, cold intolerance, and hoarseness. Physical exam shows a non-tender, firm, diffusely enlarged goiter. Serum TSH is 26 mIU/L and free T4 is 0.5 ng/dL. Anti-TPO antibodies are strongly elevated (> 500 IU/mL). What histological feature characterizes this condition?",
        "gold_diagnosis": "Lymphocytic Infiltration with Hurthle Cells",
        "options": {"A": "Lymphocytic Infiltration with Hurthle Cells", "B": "Non-caseating Granulomas with Multinucleated Giant Cells", "C": "Extensive Dense Fibrosis Invading Adjacent Tissues", "D": "Scalloped Colloid with Hyperplastic Follicular Papillae"}
    },
    {
        "id": "MEDBULLET_033",
        "question": "A 34-year-old female presents with excruciating anterior neck pain that radiates to her ears, high fever, and malaise 2 weeks after an upper respiratory tract infection. On palpation, the thyroid gland is exquisitely tender. Serum ESR is 98 mm/hr, TSH is suppressed, and 24-hour radioactive iodine uptake (RAIU) is markedly depressed (< 2%). What is the diagnosis?",
        "gold_diagnosis": "Subacute Granulomatous Thyroiditis",
        "options": {"A": "Subacute Granulomatous Thyroiditis", "B": "Graves Disease", "C": "Suppurative Bacterial Thyroiditis", "D": "Hashimoto Thyroiditis"}
    },
    {
        "id": "MEDBULLET_034",
        "question": "A 57-year-old male with alcoholic cirrhosis and ascites presents with low-grade fever, diffuse dull abdominal pain, and worsening encephalopathy. Abdominal paracentesis yields cloudy ascitic fluid with a neutrophil count of 420 cells/mm3. What is the immediate treatment of choice?",
        "gold_diagnosis": "Intravenous Cefotaxime and Albumin",
        "options": {"A": "Intravenous Cefotaxime and Albumin", "B": "Oral Metronidazole", "C": "Observation and Repeat Tap in 24 Hours", "D": "Immediate Surgical Laparotomy"}
    },
    {
        "id": "MEDBULLET_035",
        "question": "A 27-year-old male with an 8-year history of ulcerative colitis presents with progressive jaundice, pruritus, and right upper quadrant discomfort. Alkaline phosphatase is 520 U/L. MRCP demonstrates multifocal strictures and beading of the intra- and extrahepatic bile ducts. What is the diagnosis?",
        "gold_diagnosis": "Primary Sclerosing Cholangitis",
        "options": {"A": "Primary Sclerosing Cholangitis", "B": "Primary Biliary Cholangitis", "C": "Choledocholithiasis", "D": "Pancreatic Adenocarcinoma"}
    },
    {
        "id": "MEDBULLET_036",
        "question": "A 42-year-old female presents with difficulty swallowing both solid foods and liquids that has worsened over 2 years, often regurgitating undigested food while asleep. Barium esophagram demonstrates massive esophageal dilatation with a smooth, tapered narrowing at the gastroesophageal junction ('bird-beak' sign). High-resolution manometry confirms aperistalsis and incomplete LES relaxation. What is the diagnosis?",
        "gold_diagnosis": "Achalasia",
        "options": {"A": "Achalasia", "B": "Diffuse Esophageal Spasm", "C": "Esophageal Adenocarcinoma", "D": "Scleroderma Esophagus"}
    },
    {
        "id": "MEDBULLET_037",
        "question": "A 71-year-old male presents with persistent foul breath (halitosis), difficulty swallowing, and regurgitation of undigested pills and food hours after ingestion. He occasionally feels a soft swelling in the left side of his neck that gurgles when pressed. Barium swallow demonstrates an outpouching of mucosa in the posterior hypopharynx through Killian triangle. What is the diagnosis?",
        "gold_diagnosis": "Zenker Diverticulum",
        "options": {"A": "Zenker Diverticulum", "B": "Traction Diverticulum", "C": "Epiphrenic Diverticulum", "D": "Plummer-Vinson Syndrome"}
    },
    {
        "id": "MEDBULLET_038",
        "question": "A 9-month-old infant is brought to the pediatric emergency department with sudden paroxysmal episodes of severe colicky abdominal pain during which he pulls his knees up to his chest and cries inconsolably. Between episodes, he is lethargic. He recently passed a stool containing dark red mucus and blood ('currant jelly' stool). Ultrasound reveals a target/doughnut sign in the right upper abdomen. What is the initial intervention of choice?",
        "gold_diagnosis": "Air or Contrast Enema Reduction",
        "options": {"A": "Air or Contrast Enema Reduction", "B": "Immediate Open Laparotomy", "C": "Intravenous Antibiotics and Bowel Rest", "D": "Oral Polyethylene Glycol"}
    },
    {
        "id": "MEDBULLET_039",
        "question": "A 3-week-old firstborn male infant presents with non-bilious projectile vomiting immediately after feeding. On exam, he appears hungry and eager to nurse, with mild dehydration. Abdominal palpation reveals a mobile, firm, olive-shaped mass in the right epigastrium. Serum electrolytes show hypochloremic hypokalemic metabolic alkalosis. What is the diagnosis?",
        "gold_diagnosis": "Infantile Hypertrophic Pyloric Stenosis",
        "options": {"A": "Infantile Hypertrophic Pyloric Stenosis", "B": "Duodenal Atresia", "C": "Malrotation with Midgut Volvulus", "D": "Gastroesophageal Reflux Disease"}
    },
    {
        "id": "MEDBULLET_040",
        "question": "A 3-day-old male newborn has failed to pass meconium since birth, develops progressive abdominal distension, and vomits bilious fluid. Digital rectal examination reveals tight anal sphincter tone, and removal of the examining finger results in an explosive expulsion of gas and liquid stool. Abdominal radiograph demonstrates dilated colonic loops. What diagnostic procedure is gold standard?",
        "gold_diagnosis": "Rectal Suction Biopsy",
        "options": {"A": "Rectal Suction Biopsy", "B": "Barium Enema Alone", "C": "Anorectal Manometry", "D": "Sweat Chloride Testing"}
    },
    {
        "id": "MEDBULLET_041",
        "question": "A 42-year-old male with a family history of early stroke presents with hypertension (BP 164/102), intermittent flank pain, and microscopic hematuria. Abdominal examination reveals large, bilateral, non-tender, ballotable abdominal masses. Abdominal ultrasound demonstrates countless enlarged cysts replacing the renal parenchyma bilaterally. What vascular malformation is strongly associated?",
        "gold_diagnosis": "Berry Aneurysm of Circle of Willis",
        "options": {"A": "Berry Aneurysm of Circle of Willis", "B": "Thoracic Aortic Aneurysm", "C": "Coronary Artery Aneurysm", "D": "Renal Artery Stenosis"}
    },
    {
        "id": "MEDBULLET_042",
        "question": "A 4-year-old boy presents with severe periorbital and pretibial swelling, scrotal edema, and 4+ proteinuria on dipstick without hematuria. Serum albumin is 1.8 g/dL and total cholesterol is 360 mg/dL. Light microscopy of the renal biopsy is completely normal, but electron microscopy demonstrates diffuse effacement of visceral epithelial cell foot processes (podocytes). What is the first-line therapy?",
        "gold_diagnosis": "Oral Corticosteroids",
        "options": {"A": "Oral Corticosteroids", "B": "Cyclophosphamide", "C": "Rituximab", "D": "ACE Inhibitors Monotherapy"}
    },
    {
        "id": "MEDBULLET_043",
        "question": "A 52-year-old male presents with heavy proteinuria (7.5 g/24 hr), peripheral edema, and hypoalbuminemia. Serology demonstrates positive circulating antibodies against the M-type phospholipase A2 receptor (anti-PLA2R). Renal biopsy reveals diffuse uniform thickening of the glomerular capillary wall with subepithelial immune deposits displaying a 'spike-and-dome' appearance on silver stain. What is the diagnosis?",
        "gold_diagnosis": "Membranous Nephropathy",
        "options": {"A": "Membranous Nephropathy", "B": "Minimal Change Disease", "C": "Focal Segmental Glomerulosclerosis", "D": "Diabetic Nephropathy"}
    },
    {
        "id": "MEDBULLET_044",
        "question": "A 50-year-old male presents with recurrent epistaxis, chronic purulent sinusitis, saddle-nose deformity, and persistent cough with hemoptysis. Chest CT reveals bilateral cavitary nodular infiltrates, and serum creatinine is 2.8 mg/dL with red cell casts. Cytoplasmic ANCA (c-ANCA) targeting proteinase 3 is strongly positive. What is the diagnosis?",
        "gold_diagnosis": "Granulomatosis with Polyangiitis",
        "options": {"A": "Granulomatosis with Polyangiitis", "B": "Microscopic Polyangiitis", "C": "Eosinophilic Granulomatosis with Polyangiitis", "D": "Anti-GBM Disease"}
    },
    {
        "id": "MEDBULLET_045",
        "question": "A 62-year-old male presents with fever, palpable purpuric rash on his legs, rapid elevation of serum creatinine to 3.6 mg/dL, and hematuria without upper airway disease or asthma. Perinuclear ANCA (p-ANCA) targeting myeloperoxidase (MPO) is positive. Renal biopsy confirms pauci-immune necrotizing crescentic glomerulonephritis. What is the diagnosis?",
        "gold_diagnosis": "Microscopic Polyangiitis",
        "options": {"A": "Microscopic Polyangiitis", "B": "Granulomatosis with Polyangiitis", "C": "Henoch-Schonlein Purpura", "D": "IgA Nephropathy"}
    },
    {
        "id": "MEDBULLET_046",
        "question": "A 38-year-old female with severe adult-onset refractory asthma and chronic rhinosinusitis presents with numbness and weakness in her right hand and left foot (mononeuritis multiplex). Complete blood count reveals profound peripheral eosinophilia (32% eosinophils; absolute count 3,800/uL). Serum p-ANCA (anti-MPO) is positive. What is the diagnosis?",
        "gold_diagnosis": "Eosinophilic Granulomatosis with Polyangiitis",
        "options": {"A": "Eosinophilic Granulomatosis with Polyangiitis", "B": "Allergic Bronchopulmonary Aspergillosis", "C": "Hypereosinophilic Syndrome", "D": "Polyarteritis Nodosa"}
    },
    {
        "id": "MEDBULLET_047",
        "question": "A 16-year-old boy presents with scleral icterus and splenomegaly. Peripheral blood smear demonstrates round, dense red blood cells lacking central pallor (spherocytes). Coombs test is negative. Mean corpuscular hemoglobin concentration (MCHC) is elevated at 38 g/dL. Incubating red blood cells in hypotonic saline solutions leads to premature hemolysis. What is the diagnosis?",
        "gold_diagnosis": "Hereditary Spherocytosis",
        "options": {"A": "Hereditary Spherocytosis", "B": "Autoimmune Hemolytic Anemia", "C": "Glucose-6-Phosphate Dehydrogenase Deficiency", "D": "Pyruvate Kinase Deficiency"}
    },
    {
        "id": "MEDBULLET_048",
        "question": "A 24-year-old Mediterranean male presents with acute dark tea-colored urine, back pain, and scleral icterus 3 days after receiving trimethoprim-sulfamethoxazole for a urinary tract infection. Peripheral blood smear shows 'bite cells' and Heinz bodies on crystal violet stain. What enzyme defect causes this hemolytic episode?",
        "gold_diagnosis": "Glucose-6-Phosphate Dehydrogenase Deficiency",
        "options": {"A": "Glucose-6-Phosphate Dehydrogenase Deficiency", "B": "Pyruvate Kinase Deficiency", "C": "Hexokinase Deficiency", "D": "Glutathione Synthetase Deficiency"}
    },
    {
        "id": "MEDBULLET_049",
        "question": "A 6-year-old African American boy presents with severe excruciating bone pain in both legs and swelling of hands and feet (dactylitis). Hemoglobin is 7.2 g/dL, and peripheral smear shows target cells, Howell-Jolly bodies, and crescent-shaped sickled erythrocytes. What is the primary preventive pharmacotherapy to increase fetal hemoglobin (HbF)?",
        "gold_diagnosis": "Hydroxyurea",
        "options": {"A": "Hydroxyurea", "B": "Folic Acid Monotherapy", "C": "Chronic Erythropoietin", "D": "Deferasirox"}
    },
    {
        "id": "MEDBULLET_050",
        "question": "A 26-year-old female presents with petechiae, scattered ecchymoses, and epistaxis. Platelet count is critically low at 12,000/uL, but hemoglobin (13.4 g/dL), WBC (6,200/uL), and coagulation times (PT/PTT) are completely normal. Peripheral blood smear confirms true thrombocytopenia with normal red blood cell morphology. Bone marrow biopsy shows increased megakaryocytes. What is the diagnosis?",
        "gold_diagnosis": "Immune Thrombocytopenic Purpura",
        "options": {"A": "Immune Thrombocytopenic Purpura", "B": "Thrombotic Thrombocytopenic Purpura", "C": "Aplastic Anemia", "D": "Acute Lymphoblastic Leukemia"}
    }
]


# -------------------------------------------------------------
# 3. Dynamic Loaders & Dataset Resolvers
# -------------------------------------------------------------

def load_medqa_all() -> List[Dict[str, Any]]:
    """Loads all MedQA benchmark cases from local cache or gold-standard pool."""
    p = os.path.expanduser("~/.cache/huggingface/hub/datasets--GBaker--MedQA-USMLE-4-options/snapshots/*/*.jsonl")
    matches = glob.glob(p)
    if matches:
        try:
            with open(matches[0], "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]
            items = []
            for i, row in enumerate(lines):
                ans_val = row.get("answer")
                items.append({
                    "id": f"MEDQA_{i+1:04d}",
                    "question": row["question"],
                    "gold_diagnosis": ans_val,
                    "options": row["options"],
                    "category": "MedQA (USMLE)"
                })
            if items:
                logger.info(f"Loaded {len(items)} MedQA cases from cached USMLE jsonl.")
                return items
        except Exception as e:
            logger.warning(f"Error reading MedQA jsonl cache: {e}. Using gold-standard 50 cases.")
    return list(MEDQA_50_CASES)


def load_pubmedqa_all() -> List[Dict[str, Any]]:
    """Loads biomedical QA cases from cached PubMedQA parquet, or built-in fallback."""
    p = os.path.expanduser("~/.cache/huggingface/hub/datasets--qiaojin--PubMedQA/snapshots/*/pqa_labeled/*.parquet")
    matches = glob.glob(p)
    if matches:
        try:
            df = pd.read_parquet(matches[0])
            items = []
            for _, row in df.iterrows():
                ctx_list = row["context"].get("contexts", []) if isinstance(row["context"], dict) else []
                ctx_text = " ".join(ctx_list) if ctx_list is not None else ""
                decision = str(row.get("final_decision", "yes")).capitalize()
                items.append({
                    "id": f"PMQA_{row['pubid']}",
                    "question": row["question"],
                    "context": ctx_text[:600],
                    "gold_diagnosis": decision,
                    "options": {"A": "Yes", "B": "No", "C": "Maybe"},
                    "long_answer": row.get("long_answer", ""),
                    "category": "PubMedQA (Biomedical Research)"
                })
            if items:
                logger.info(f"Loaded {len(items)} PubMedQA cases from cached parquet.")
                return items
        except Exception as e:
            logger.warning(f"Error reading PubMedQA parquet cache: {e}")

    # Fallback to curated 10 PubMedQA cases
    return [
        {
            "id": f"PMQA_00{i+1}",
            "question": q,
            "context": ctx,
            "gold_diagnosis": dec,
            "options": {"A": "Yes", "B": "No", "C": "Maybe"},
            "long_answer": ctx,
            "category": "PubMedQA (Biomedical Research)"
        }
        for i, (q, ctx, dec) in enumerate([
            ("Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?",
             "Results depicted mitochondrial dynamics in vivo as PCD progresses within the lace plant, and highlight the correlation of this organelle with other organelles during developmental PCD. Overall, our findings implicate the mitochondria as playing a critical and early role in developmentally regulated PCD in the lace plant.", "Yes"),
            ("Does intravenous immunoglobulin benefit patients with severe Guillain-Barre syndrome?",
             "Randomized controlled clinical trials show IVIG started within 2 weeks of onset hastens recovery as effectively as plasma exchange, with high clinical efficacy.", "Yes"),
            ("Is routine antibiotic prophylaxis indicated for uncomplicated viral bronchitis?",
             "Systematic reviews and clinical practice guidelines confirm that antibiotic treatment does not provide significant clinical benefit for uncomplicated acute bronchitis in healthy adults.", "No"),
            ("Can continuous glucose monitoring improve glycemic control in type 1 diabetes?",
             "Meta-analyses of randomized trials demonstrate significant reductions in HbA1c and hypoglycemia duration with continuous glucose monitoring versus self-monitoring of blood glucose.", "Yes"),
            ("Does carotid endarterectomy reduce stroke risk in asymptomatic low-grade stenosis (<50%)?",
             "Major prospective trials show no statistically significant reduction in long-term stroke risk for asymptomatic patients with carotid stenosis less than 50% compared to medical therapy.", "No"),
            ("Are statins associated with reduced cardiovascular mortality in primary prevention?",
             "Large-scale placebo-controlled clinical trials demonstrate clear reductions in all-cause mortality, myocardial infarction, and stroke across diverse patient cohorts receiving statin therapy.", "Yes"),
            ("Is routine screening for ovarian cancer with CA-125 and transvaginal ultrasound recommended in asymptomatic average-risk women?",
             "The USPSTF systematic evidence review concluded with moderate certainty that screening does not reduce ovarian cancer mortality and leads to major surgical harms.", "No"),
            ("Does prone positioning improve survival in severe acute respiratory distress syndrome (ARDS)?",
             "Randomized trials and meta-analyses show that prone positioning for at least 16 hours per day significantly reduces 28-day and 90-day mortality in patients with severe ARDS.", "Yes"),
            ("Is hydroxychloroquine effective for post-exposure prophylaxis against SARS-CoV-2 infection?",
             "Double-blind randomized trials demonstrated that hydroxychloroquine did not prevent illness or infection following exposure compared with placebo.", "No"),
            ("Do SGLT2 inhibitors reduce hospitalization for heart failure across patients with and without diabetes?",
             "Systematic reviews and major clinical trials (DAPA-HF, EMPEROR-Reduced) demonstrate substantial reductions in cardiovascular death and heart failure hospitalization regardless of diabetes status.", "Yes")
        ])
    ]


def load_medbullets_all() -> List[Dict[str, Any]]:
    """Loads all 50 gold-standard MedBullets residency clinical cases."""
    return list(MEDBULLETS_50_CASES)


def load_medqa_50() -> List[Dict[str, Any]]:
    """Returns the gold-standard 50 MedQA diagnostic cases."""
    return list(MEDQA_50_CASES)


def load_pubmedqa_50() -> List[Dict[str, Any]]:
    """Returns the first 50 PubMedQA cases."""
    return load_pubmedqa_all()[:50]


load_medqa_cases = load_medqa_all
load_pubmedqa_cases = load_pubmedqa_all
load_medbullets_cases = load_medbullets_all


def get_random_dataset_cases(dataset_name: str = "medqa", count: int = 10, random_sample: bool = True) -> List[Dict[str, Any]]:
    """
    Selects count cases from the requested benchmark dataset.
    If random_sample=True, selects a random sample of count cases without replacement.
    """
    norm = dataset_name.lower().replace("-", "").replace("_", "").strip()

    if "pubmed" in norm:
        pool = load_pubmedqa_all()
        dataset_label = "PubMedQA"
    elif "bullet" in norm:
        pool = load_medbullets_all()
        dataset_label = "MedBullets"
    else:
        # Default MedQA: use gold-standard cases or full pool
        pool = load_medqa_all()
        dataset_label = "MedQA"

    if not pool:
        logger.warning(f"No cases found for dataset '{dataset_name}'. Falling back to MEDQA_50_CASES.")
        pool = list(MEDQA_50_CASES)

    k = min(count, len(pool))
    if random_sample:
        selected = random.sample(pool, k)
        logger.info(f"Randomly selected {k} cases from {dataset_label} (pool size: {len(pool)})")
    else:
        selected = pool[:k]
        logger.info(f"Selected first {k} cases from {dataset_label}")

    return selected


def format_case_prompt(case: Dict[str, Any], dataset_name: str) -> str:
    """Formats a benchmark case into a standardized prompt for the CDSS pipeline."""
    norm = dataset_name.lower().replace("-", "").replace("_", "").strip()

    if "pubmed" in norm:
        return (
            f"Case ID: {case['id']}\n"
            f"Biomedical Research Study Evaluation:\n"
            f"Question: {case['question']}\n\n"
            f"Abstract Evidence:\n{case.get('context', '')}\n\n"
            f"Instructions: Based on the abstract evidence, determine the definitive final scientific decision (Yes, No, or Maybe). "
            f"Your primary conclusion must be one of the candidate decision options below.\n\n"
            f"Candidate Diagnostic Options:\n"
            f"A: Yes\n"
            f"B: No\n"
            f"C: Maybe"
        )
    elif "bullet" in norm:
        opts = case.get("options", {})
        opts_str = "\n".join([f"{k}: {v}" for k, v in opts.items()])
        return (
            f"Case ID: {case['id']}\n"
            f"Clinical Residency Vignette:\n{case['question']}\n\n"
            f"Candidate Diagnostic Options:\n{opts_str}"
        )
    else:
        opts = case.get("options", {})
        opts_str = "\n".join([f"{k}: {v}" for k, v in opts.items()])
        return (
            f"Case ID: {case['id']}\n"
            f"Clinical Case Vignette:\n{case['question']}\n\n"
            f"Candidate Diagnostic Options:\n{opts_str}"
        )
