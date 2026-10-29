# Paper Deep Dive — *An Adaptive Multi-Agent LLM-Based Clinical Decision Support System Integrating Biomedical RAG and Web Intelligence*

> Complete reading guide, dataset/metric teardown, independent statistical audit, improvement strategy, and panel Q&A bank.
> Prepared for presenting and critically extending this work.

---

## 0. Bibliographic record

| Field | Value |
|---|---|
| Title | An Adaptive Multi-Agent LLM-Based Clinical Decision Support System Integrating Biomedical RAG and Web Intelligence |
| Authors | Çağatay Umut Öğdü, Kübra Arslanoğlu (Member, IEEE — **corresponding**), Mehmet Karaköse (Senior Member, IEEE) |
| Affiliation | Dept. of Computer Engineering, Fırat University, 23119 Elâzığ, Türkiye |
| Contact | karslanoglu@firat.edu.tr |
| Venue | IEEE Access, Vol. 13, 2025, pp. 167390–167404 |
| DOI | `10.1109/ACCESS.2025.3613340` |
| Timeline | Received 4 Sep 2025 · Accepted 19 Sep 2025 · Published 23 Sep 2025 · Current version 29 Sep 2025 |
| Licence | CC BY 4.0 |
| Associate editor | Qiang Li |
| Funding note | "This study was conducted within the scope of Ph.D. thesis of Kübra Arslanoğlu" |
| PDF producer | LaTeX/pdfTeX 1.40.24 · 15 pages |

**Venue read.** IEEE Access is an open-access multidisciplinary journal with fast review (15 days here: 4 Sep → 19 Sep). It is a legitimate venue but it is *not* a top-tier clinical-AI venue. Expect the panel to be more pointed about methodology than the venue's acceptance implies.

---

## 1. Executive summary

The paper proposes a **Clinical Decision Support System (CDSS)** built from **five LLM agents that each run a different model**, orchestrated with the **CrewAI** framework:

1. **Clinical Text Clarifier** (DeepSeek-R1) — turns messy free-text patient notes into a strict JSON clinical schema
2. **Symptom RAG Analyzer** (Gemini-2.0-Flash) — retrieves top-*k* passages from a **BioBERT/Qdrant** vector index and proposes ranked diagnoses
3. **Evidence-Based Scanner** (Qwen2-72B) — independently searches the live web (Serper → PubMed, Google Scholar, NICE) for current guidelines and literature
4. **Clinical Data Fusion** (GPT-4o-mini) — cross-checks the static RAG evidence against the fresh web evidence, detects conflicts, computes a confidence score
5. **Adaptive Optimizer** (Gemini-2.0-Flash) — if confidence `< τ = 0.65`, generates a structured feedback report and **re-runs** the RAG + web agents with reworded/expanded queries, up to 3 iterations

Headline results: **94% MedQA, 88% PubMedQA, 84% MedBullets**, on **50 questions per dataset**. Ablation shows removing the Fusion agent is most damaging (→60%), removing the RAG Analyzer →65%, removing the Adaptive Optimizer →70%.

No training or fine-tuning is performed. Everything is inference-time orchestration.

---

## 2. The claim sheet — what is claimed vs. what is supported

This is the single most important table in this document. Read it before the panel does.

| # | Claim in paper | Where | Actually supported? |
|---|---|---|---|
| 1 | "94%, 88%, 84% accuracy" | Abstract, §III, §V | **Partially.** Correct arithmetic for n=50 (47/44/42). But no confidence intervals and no significance test anywhere. |
| 2 | "substantial improvements over state-of-the-art" | Abstract | **Not on MedQA.** 94% vs DeepSeek-R1's 92% is p = 0.70. **Not on PubMedQA** (p = 0.28) or **MedBullets** (p = 0.60) vs their best baselines. |
| 3 | "statistically significant performance improvement over SOTA" | §V | **Unsupported.** No statistical test is reported in the paper at all. |
| 4 | Accuracy rises 86.3% → 94.7% over 3 iterations | §III, Table 6 | **Directionally plausible, statistically void.** n = 20. 86.3% of 20 = 17.26 correct — not an integer. p = 0.37. |
| 5 | "improving output quality by 14.2%" | §III | **Arithmetically wrong.** 86.3 → 94.7 is +8.4 points, +9.7% relative. |
| 6 | "increasing system consistency by 1.8 times" | §III | **Undefined metric.** No consistency measure is ever defined or computed. |
| 7 | "18M embeddings from 200,000 chunks for 1,000 PDFs, index 2.1 GB" | §II-B | **Internally contradictory** (see §21.4). 200k chunks = 200k embeddings; 200k × 768 × fp32 = 0.61 GB. |
| 8 | "processing times increased logarithmically" | §III | **Wrong shape.** Table 6 shows 2.5 → 3.0 → 3.7 → 4.4 → 5.5 min: linear. |
| 9 | "final calibrated confidence" | §II-A4, §II-C | **No calibration method.** No formula, no ECE, no Brier, no reliability diagram. τ = 0.65 is asserted. |
| 10 | Customized BioBERT RAG correctly interprets medical terminology | Contribution 2 | **Unmeasured.** No recall@k, MRR, or nDCG is ever reported for retrieval. |
| 11 | MedQA "multiple choice (A–E)" | Table 3 | **Check this.** Base MedQA-USMLE is 4-option (A–D); the 5-option variant is a separate release. Worth confirming which was used. |
| 12 | Citation [27] justifies choosing Gemini-2.0-Flash | §II-A2 | **Citation mismatch.** Ref [27] is a *gender/content bias* study of Gemini 2.0 Flash, not a capability comparison. |
| 13 | Citation [29] justifies GPT-4o-mini | §II-A4 | **Citation mismatch.** Ref [29] is "A study of ChatGPT-4 vs ChatGPT-4o" — not about GPT-4o-mini. |

**How to use this table.** If the panel challenges one number, concede the *measurement* limitation gracefully and pivot to the ablation (§19), which **is** statistically defensible. That is the paper's real contribution and the safe ground.

---

## 3. Problem framing (Section I)

### 3.1 The motivating statistics

| Statistic | Source cited | Note |
|---|---|---|
| 44,000–98,000 preventable medical-error deaths per year, US alone | IOM 2000 report [1] | ~25 year old. Panel may note this. |
| ~70% of those errors are human-caused | [3] Ohta et al., *J Patient Safety* 2019 | Cardiology inpatients specifically — narrower than the claim implies. |

### 3.2 Why existing systems fail

**Rule-based / static systems.** Cited failure mode: they cannot adapt to dynamic patient profiles, and they are limited to static algorithms [4] (Greenes, *Clinical Decision Support: The Road to Broad Adoption*).

**Ontology-based systems.** Worked example given is **OntoDiabetic** [12] — Sherimon & Krishnan (2016), *Arabian J Sci Eng* — which models diabetic clinical guidelines as **Web Ontology Language** rules for risk analysis. Strength: formal machine-processable knowledge, consistent logical reasoning. Weakness: constrained to static, predetermined knowledge bases, so it cannot handle dynamic, noisy, unstructured clinical data or fast-changing evidence.

**Single-model LLMs.** Worked example: **Levra et al.** [14], *Eur J Intern Med* 131:113–120 (2025) — LLM analysis of free-text ED EHR notes to distinguish syncope cases. Strength: LLMs understand implicit patterns in unstructured clinical text. Weakness: it is a pure **knowledge-based recognition** task inside existing records — no cross-validation against external sources, no dynamic optimization in low-confidence situations, and no integration with real-time patient data streams.

**Traditional LLM eval.** [16] Öğdü et al. tested LLMs across nine outpatient clinic branches. Radiology trials: **GPT-4 topped at 98%**, **Llama-3.3-70B at 96%**. But in some cases **GatorTron's accuracy dropped by as much as 12%**. Lesson: model choice alone is high-variance and cannot be relied on.

### 3.3 The two systems this paper positions against

**KG4Diagnosis [17]** — Zuo, Jiang, Mo & Liò (AAI Bridge Program AI Med. Healthcare, vol. 281, pp. 195–204, 2025). Hierarchical multi-agent LLM + knowledge graph. Uses **BioBERT** to break medical texts into semantic chunks, extracts entities/relations, builds a KG, splits work between **GP-LLM** and **expert advisor** agents.
- ✅ Strong: semantic integrity and structure.
- ❌ Weak: unsuitable for real-time data streams, no web-based up-to-date knowledge scanning, no feedback optimization with confidence-score-based outputs.

**MDAgents [18]** — Kim, Park, Jeong, Chan, Xu, McDuff, Lee, Ghassemi, Breazeal & Park, *NeurIPS* 37:79410–79452 (2024). "MedAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making." Divides tasks among role-defined expert agents: answering clinical questions, scoring diagnostic candidates, analysing symptom–diagnosis matches. A multi-stage pool of experts produces more explainable/reasoned output than single-agent, and can compare hypotheses across an internal representation space, analyse clinical consistency, and re-evaluate when necessary.
- ✅ Strong: interpretability, LLM visibility into decision-making.
- ❌ Weak: relies heavily on **LLM content-based inference**; **limited integration of external information sources** (literature, web, databases) constrains accuracy and timeliness; **no dynamic adaptation or feedback loops between agents**, so quality depends on getting the predefined task allocation right.

Also discussed: OntoDiabetic-style structuring, symptom identification [24], patient triage [25], text-guided echocardiography segmentation, medical image deepfake detection [11].

### 3.4 The gap statement and research question

The authors' framing: no existing solution *simultaneously* ensures **timeliness** (fresh evidence), **robustness to heterogeneous inputs**, and **confidence-aware, verifiable outputs**.

> **Research question.** *"Can an LLM-based, multi-agent CDSS that simultaneously retrieves, authenticates, and aggregates biomedical RAG knowledge with web evidence in real-time and closes the loop using confidence-driven feedback deliver sound, timely, and interpretable clinical recommendations?"*

### 3.5 The four sub-tasks

The paper decomposes the problem so each gets a dedicated mechanism:

| # | Sub-task | Mechanism |
|---|---|---|
| 1 | Robust clinical text normalization & structuring (noisy input) | Clinical Text Clarifier agent |
| 2 | Domain-specific biomedical retrieval (terminology, abbreviations) | BioBERT-based RAG |
| 3 | Trust-aware multi-source combination with conflict detection + confidence calibration | Clinical Data Fusion agent |
| 4 | Adaptive feedback for re-querying and hypothesis refinement when confidence is low | Adaptive Optimizer agent |

This decomposition is the paper's most transferable idea. Specialisation (agent roles) + evidence freshness (web) + factual grounding (RAG) + decision calibration (confidence-guided feedback).

### 3.6 Stated contributions

1. Overcome fixed-rule limits — flexible multi-agent architecture that dynamically analyses patient data and optimizes itself over time.
2. Go beyond general-purpose solutions — specialized RAG ensuring correct interpretation of medical terminology and abbreviations.
3. Increase reliability — automatic merging and cross-checking of WEB and RAG agent data.

### 3.7 Honest novelty statement (quote this)

> *"This study does not present any truly new algorithmic paradigm, but it fills an important gap in the literature it combines existing pieces (biomedical RAG, web-based evidence retrieval, and adaptive feedback mechanisms) into a single agent-driven CDSS framework. What is original about our work is that we systematically integrate and interoperate these components in real-time within a modular, task-specific architecture."*

**Use this in your presentation.** Saying "we built an integration, not a new primitive" is a strength — it preempts "what's your novel contribution?" and redirects the panel to system design, where the paper actually holds up.

---

## 4. The architecture (Section II-A)

### 4.1 System-level flow

Built in **CrewAI** using its four primitives — `Agent`, `Crew`, `Process`, `Task`. Modular agents combine through **sequential process flows** plus **dynamic feedback loops**.

```
Raw clinical narrative
        │
        ▼
[1] Clinical Text Clarifier  (DeepSeek-R1)
        │  structured JSON
        ├──────────────────────┐
        ▼                      │
[2] Symptom RAG Analyzer  ────┤  (in parallel)
    (Gemini-2.0-Flash)         │
        │ ranked dx list       │
        ▼                      ▼
        └──────────► [3] Evidence-Based Scanner
                     (Qwen2-72B)
                            │ current web evidence
                            ▼
                     [4] Clinical Data Fusion
                        (GPT-4o-mini)
                            │ consolidated report + confidence
                            ▼
                     [5] Adaptive Optimizer
                        (Gemini-2.0-Flash)
                            │ confidence ≥ τ ? ──yes──► output
                            │ confidence < τ
                            ▼
                   feedback report ──► re-query [2] and [3]
                            └──────────► (max 3 iterations)

        Diagnostic results of the patient
```

**The paper's own architectural gloss:** "a two-tiered architecture" — a *static* tier (RAG over a frozen vector DB) and a *dynamic* tier (web scanning that continually updates what is known) — "repeatedly verified by a meta-controller agent-assisted iterative feedback mechanism."

### 4.2 Agent 1 — Clinical Text Clarifier (DeepSeek-R1)

**Problem it solves:** downstream agents cannot understand plain user text.

**Pipeline, in order:**

| Step | Operation | Detail |
|---|---|---|
| 1 | **Text Normalization** | Removes noise: abbreviations (e.g. "HT" → hypertension), spelling errors, stray punctuation. Converts to standard format. |
| 2 | **NER** | Identifies and labels clinical entities from a predefined label set: `Symptom`, `Disease`, `Medication`, `Test Result`. |
| 3 | **Association Extraction** | When entity recognition alone is insufficient, establishes semantic relations between entities. Example: symptom `Numbness` + anatomical region `Left Arm` → "numbness in the left arm". |
| 4 | **Ontology mapping** | Maps identified concepts to **SNOMED CT** or **ICD-10** codes. |
| 5 | **Serialization** | Emits semantically enriched structured output (JSON) with a **predefined schema**, consumable by other agents and other LLMs. |

Also colour-codes critical diagnostic information to make hastily written emergency notes legible. Runs **zero-shot**.

**Why DeepSeek-R1:** highest schema concordance, lowest error-field rate (especially for `Symptom–Anatomical Region` relationships), and low latency so it does not bottleneck stage 1.

**Complexity:** ≈ **O(L)** for single-pass normalization, L = input tokens.

> **Panel trap.** Step 3 ("association extraction") and step 4 ("SNOMED CT / ICD-10 mapping") are described as things the agent *does*, but the paper never says which NER model, which ontology version, or how the mapping is validated. It is unclear whether DeepSeek-R1 is generating codes or whether a real ontology service is wired in. Answer honestly: *"the paper doesn't specify; we implemented it as LLM-driven schema generation against a real SNOMED/ICD lookup and measured the code-mapping accuracy separately."*

### 4.3 Agent 2 — Symptom RAG Analyzer (Gemini-2.0-Flash)

Takes the Clarifier's structured entities → builds a semantically dense query vector → **cosine similarity** search against the custom vector DB → produces a preliminary ranked diagnosis list.

**Key semantic claim (quote this):** the vector DB is *not* a catalogue of diagnoses. It is a BioBERT–Qdrant index of textbook, guideline and epidemiology passages. The model proposes candidates, and **candidates are retained only when at least one retrieved passage explicitly supports them.** This is what makes every diagnosis traceable to a cited text span, and it is a genuinely good design decision.

**Complexity:** Qdrant ANN search scales sublinearly with collection size N and linearly with passages fetched k → ≈ **O(log N) + O(k)**. Ranked by cosine similarity between query and passage embeddings; only top-*k* become context.

**Why Gemini-2.0-Flash:** most balanced results for RAG context-consistent hypothesis generation plus fast function-calling flow. Required capabilities: long-context semantic alignment and decision justification grounded in RAG evidence.

### 4.4 Agent 3 — Evidence-Based Scanner (Qwen2-72B)

Its stated purpose is to *mitigate the risk of the static vector DB going stale* and to exploit new methods. Rather than answering a single patient query, it proactively keeps the knowledge base current.

**Mechanism:**
1. Develops **programmatic searches optimized per hypothesis**.
2. Submits them via a web search API (**Serper**) to scholarly indexes (**PubMed**, **Google Scholar**) and settled guideline databases (**NICE** — National Institute for Health and Care Excellence).
3. Interprets returned web pages to derive purpose-relevant text summaries.
4. **Evidence assessment / hierarchy weighting:** guidelines and systematic reviews > meta-analyses > case reports. Higher-value evidence is weighted more heavily.

**Why Qwen2-72B:** excels at both generating specific search queries *and* summarizing dense academic text; produces concise per-hypothesis evidence summaries.

> **Panel trap.** Qwen2-72B is an 18B-active-parameter model cited as [28] Tian et al. *"Think twice: Enhancing LLM reasoning by scaling multi-round test-time thinking"* — a test-time-scaling paper, not the Qwen2-72B model card. And the choice of Gemini-2.0-Flash is cited to a *bias* study [27]. Ask about this yourself before the panel does.

### 4.5 Agent 4 — Clinical Data Fusion (GPT-4o-mini)

Described as **the core synthesis component**; produces the final analytical output.

**Two input streams:** (1) static RAG evidence, (2) dynamic web evidence.

**Workflow:**
1. **Cross-validation and consistency check** between the two streams. Concrete example given: does a treatment procedure in the textbook the RAG relies on *still comply* with the latest crawled clinical guideline?
2. **Conflict detection protocol** to identify inconsistencies.
3. **Final calibrated confidence per diagnostic hypothesis.** Depends on: RAG similarity score, web evidence score, degree of concordance between the two streams.
4. **Explainable report** — structured synthesis summarizing evidence, source, and final confidence for each diagnosis.

**Why GPT-4o-mini:** scalable synthesis of clinical data from multiple sources.

> **This is the paper's single biggest methodological hole.** They call the confidence "calibrated" and use it to gate a feedback loop, but **they never give the formula, never give the weights, and never report any calibration metric.** See §21.2. This is exactly where our project beats them.

### 4.6 Agent 5 — Adaptive Optimizer (Gemini-2.0-Flash)

The workflow manager. When confidence `< τ`, it:
- identifies untrustworthy data
- intelligently reorganizes query parameters: **keyword changes, query expansion/reduction, other key points to investigate**
- generates a **feedback report**
- uses that report to issue better query recommendations to the Symptom RAG Analyzer and Evidence-Based Scanner, which then generate new queries

**Termination:** loop halts when (i) combined confidence `> τ = 0.65`, **or** (ii) max of **3 iterations** is reached. In practice converges in 2–3 cycles, mean response **3.7 minutes** on MedQA.

**Why Gemini-2.0-Flash:** supports real-time iteration with low latency and high call throughput — required for instantaneous requery and parameter adaptation.

### 4.7 Complexity summary

| Stage | Complexity | Paper's justification |
|---|---|---|
| Clarifier | **O(L)** | single-pass normalization, L tokens |
| RAG search | **O(log N) + O(k)** | Qdrant ANN, N collection size, k passages |
| Whole case | **O(k + R)** | k RAG passages, R crawled web pages — dominant in practice |
| Iterations | multiply by iteration count | 1→5 iterations ≈ linear in observed wall-clock |

---

## 5. The biomedical RAG architecture (Section II-B)

### 5.1 Vector database creation pipeline (Figure 3)

```
Collect diverse, reliable biomedical sources (PubMed articles + symptom-disease datasets)
        ▼
Download up to 1,000 articles in parallel, rank by Entrez "relevance"
        ▼
PDFs indexed locally → PyMuPDF text extraction → typo correction → standard format
        ▼
OCR path for the 28% of pages needing it → ICU Unicode normalization
        ▼
Regex/keyword filter strips references, captions, copyright, author info,
abstract, keywords, emails, publication-specific boilerplate
        ▼
RecursiveCharacterTextSplitter: chunk_size=512, chunk_overlap=64
   └─ if segment count inadequate → page-based fallback: 500 / 80
        ▼
Drop too-short / low-content / label-only segments
        ▼
Tag metadata: source filename, page number, text length
        ▼
Embed every chunk with BioBERT-mnli-snli-scinli-scitail-mednli-stsb → 768-d
        ▼
Load vectors + text + metadata into Qdrant (gRPC)
```

**Corpus sourcing detail:** search queries loaded with **15+ keywords** on symptom–diagnosis correlation, clinical findings patterns, multi-organ injury; filters **biased toward human clinical studies and diagnostic papers**. **~20 topics excluded** — named examples: AI applications, cancer biomarkers, pandemic vaccine effects. Entrez's **history option activated** to avoid overloading NCBI servers. Parallelization gave a **5.8× speedup** over one CPU core.

**Ingestion benchmarks claimed:**

| Claim | Value |
|---|---|
| PyMuPDF extraction rate | **65 pages/sec** |
| vs pdfplumber | **~9.6× faster** |
| Pages requiring OCR | **28%** |
| Parallelization speedup | **5.8×** |
| gRPC vs REST latency | **−32%** |
| Embeddings produced | **"18M embeddings from 200,000 chunks for 1,000 PDFs"** |
| Final index size | **2.1 GB** |
| Total processing time | **38 minutes** |

> See §21.4 — these numbers do not reconcile.

**Embedding model choice:** BioBERT (Lee et al., *Bioinformatics* 36(4):1234–1240, 2020) was chosen because it was **pre-trained on PubMed articles**, and this study uses a **multi-task BioBERT architecture trained on biomedical NLI datasets** to handle medical jargon and abbreviations — the explicit gap they claim over general-purpose embedding models used in prior RAG CDSS work like [21].

### 5.2 Corpus composition

| Source | Count | Rationale given |
|---|---|---|
| PubMed articles | 1,000 PDFs downloaded (intro says **889** publicly available medical articles) | common *and* rare conditions |
| Symptom-disease dataset [30] | added to boost semantic diversity | see below |
| **Total patient texts** | **5,630** | — |

**The symptom-disease dataset** is `dux-tecblic/symptom-disease-dataset` on HuggingFace. Open-label: thousands of symptoms with corresponding diagnoses. Chosen because:
1. It is **labeled**, so the system can generate more informative contextual embeddings.
2. It reflects real-world patient and symptom diversity.
3. It contains clinical conditions **poorly represented in the literature** — important for primary care where clinical variability is normal.

Each data point is enriched so that the same diagnosis appears under many different semantic phrasings, making it accessible through different vector-space representations.

**Per-point metadata:** unique ID, associated text, **source** (`symptom-disease-dataset`, `medical-papers`) and **kind** of text (`diagnosis-symptom-pair`, `book-chunk`). This enables source-related filtering and analysis.

### 5.3 Figure 4 — embedding cluster visualization

A schematic conceptual decomposition of symptom–diagnosis embeddings. Clusters shown: **blue = `headache–migraine`**, **green = `chest pain–angina`**, **orange = `fever–influenza`**. Purpose is illustrative only — to show the model picks up semantic similarity between symptoms and diagnoses in 3D space. **No clustering metric (silhouette, ARI) is reported.**

### 5.4 Customized search tool for LLM agents

This is contribution #2, and it is the most technically interesting part.

**Fully parametrically defined** — directly invoked and customizable by agents w.r.t. vector similarity threshold, max number of results, and context-based filtering options.

**Medical term boosting.** Enhances the content weight of specific clinical terms by encouraging repeated occurrence of high-priority symptoms in the query so they gain importance in the embedding. Named examples: *"shortness of breath"*, *"chest pain"* get higher vector-space scores so prioritized documents surface.

**Three-level fallback strategy** when a query does not succeed (Figure 5):

```
Start → Query Entry → Query Preprocessing → Medical Term Boosting
        ▼
Qdrant Vector Search (threshold = 0.65)
        ▼
  Was a satisfactory result found?
        ├── NO ──► Strategy 1: Threshold Reduction (lower similarity threshold, larger matches)
        │              ▼
        │        Is the result sufficient?
        │          ├── YES ──► Results Integration (Score | Source Info | Document ID | …)
        │          └── NO  ──► Strategy 2: Simplify Query (linguistically simplify, re-embed)
        │                         ▼
        │                   Is the result sufficient?
        │                     ├── YES ──► Results Integration
        │                     └── NO  ──► Strategy 3: Remove Symptoms / reconstruct from
        │                                    stored clinical semantic content → new
        │                                    semantic cluster → alternative query
        └── YES ──► Results Integration
```

**The stated innovation:** Strategy 3 is *not* constructed like traditional top-*k* rotation. The system converts initial symptom patterns into a **new semantic cluster** to build an alternative query, which "reveals learned behavior when handling inexact queries."

**Verifiable decision traces.** Detailed metadata is maintained per query: source, document ID, page, fragment length, retrieval time. At the end of each query the system returns high-confidence-scoring results to the agent.

### 5.5 Retrieval protocol — the three guarantees

| Guarantee | Mechanism |
|---|---|
| **Multi-Source Validation** | Symptom inferences cross-validated against **both** the vector DB and academic web sources. |
| **Contextual Confidence Score** | Every recommendation scored by the Clinical Data Fusion agent; **data below the threshold is selected and not made available to the agent.** |
| **Semantic Fusion** | RAG + web information combined into a single aggregate — the **"best data"** — through a consistency check. |

Reported: on MedQA the **fused system reached 94.7%**, beating **RAG-only (65.0%)** by **+29.7** points and **web-only (70.0%)** by **+24.7** points.

> **Panel trap.** "RAG-only 65.0%" is the number from Table 7's *ablation* row for removing the Symptom RAG Analyzer, and "web-only 70.0%" is the ablation row for removing the Evidence-Based Scanner. The paper relabels ablation results as *configurations* and computes a delta against a *different* 94.7% figure than the 94.0% in Table 4. Push on this.

---

## 6. End-to-end clinical validation protocol (Section II-C)

Three key agents drive this flow: Symptom RAG Analyzer, Evidence-Based Scanner, Clinical Data Fusion — using the custom Qdrant vector search utility and the Serper web query module — to produce an overall analysis from literature *and* live medical resources.

### Figure 6 — the worked pneumonia case (study this carefully)

This is the only concrete demonstration in the paper. Know it.

**Input prompt (raw patient text):**
> 45 y male, fever over 38.5 deg C x3 d. Persistent dry cough + dyspnea. Recent travel abroad (≤14 d). Labs: WBC 15090/mm3, CRP 20 mg/L. Troponin-I negative. CXR: left lower-lobe hazy opacity. CT: LLL consolidation + pleural fluid.

**[1] Clinical Text Clarifier →**
| Field | Value |
|---|---|
| DEMOGRAPHICS | 45 y, Male |
| SYMPTOMS | Fever 38.5 °C 3 d |
| TRAVEL | International trip ≤ 2 w |
| LAB | WBC 15k ↑, CRP 20 ↑ |
| IMAGING | CXR hazy LLL |
| STRUCTURED DX | Lower-lobe pneumonia |

**[2] Symptom RAG Analyzer → TOP DIAGNOSES**
| Rank | Diagnosis | Score | Tier |
|---|---|---|---|
| 1 | Pneumonia | 0.75 | High |
| 2 | Pulmonary Embolism | 0.57 | Mod |
| 3 | Empyema | — | — |
| 4 | Q fever | 0.39 | Low |

KEY FEATURES: WBC high, focal LLL, travel.
NEXT STEPS: sputum or blood culture; consider CT angiogram if PE likely.

**[3] Clinical Data Fusion → BEST ARTICLES**
- *StatPearls, Bacterial Pneumonia* — high — "matches fever, cough, WBC up, LL infiltrate"
- *CHEST 2028, PE with negative trop* — "suggests D-dimer + age-adjusted cut-off"

**GAP / CONFLICT flagged:** troponin normal yet dyspnea present → ≥3 cardiac less likely.

**[3] refined output:**
- RAG TUNE: + pulmonary embolism, D-dimer − bacterial pneumonia, viral
- WEB SEARCH: `"pulmonary embolism" dyspnea neg trop`
- DECISION: **if D-dimer < age-adjusted → rule out PE; else order CT angiography**

**[5] Adaptive Optimizer:** the fused data *failed* the confidence check, so it issued improved RAG TUNE / WEB SEARCH guidance — the agents then re-queried along those lines, and the loop ran to a maximum of **3 cycles**. Any more "would cost and take longer."

**Note on Table 6's iteration=3 wall-clock of 3.7 min** — that matches the §II-A5 claim of "mean response time of 3.7 minutes on the MedQA benchmark." Internal consistency check: **passes.**

---

## 7. The adaptive feedback loop (Section II-D)

Described as **one of the key innovations**. It generates feedback reports that let the system "learn from its own outputs" and optimize for gaps in search queries or scenarios needing investigation.

**Three building blocks:**

1. **Scoring.** The Adaptive Optimizer evaluates each analysis with a confidence score (τ = 0.65 = acceptance threshold for reliability). Results below threshold are prompted for feedback due to possible missing or incorrect information.
2. **Iteration.** The feedback mechanism prompts re-investigation by providing new guidance reports to the agents, up to a specified maximum number of iterations.
3. **Semantic structure.** Feedback is *not* mechanical repetition — it consists of **semantically structured recommendations** generated by contextual evaluation of the outputs.

**Resulting capabilities:**
- Automatically restructure queries that fail initial analysis
- Generate more appropriate alternative queries for ambiguous/missing symptoms
- Exhibit **reflexive behaviour** — optimizing itself by learning from case similarities over time

**The convergence failure mode the authors themselves identify (very useful for us):**

> *"The reason the feedback loop begins to repeat the same suggestions is the main reason for the lack of a significant increase in accuracy with further iterations."*

And from Table 6's discussion: gains beyond iteration 3 are marginal; *"Minimal changes were observed in the fourth and fifth cycles."*

> **This is our single best engineering attack surface.** The authors diagnose their own ceiling — a feedback loop that re-emits identical suggestions — and then never fix it. A loop with (a) an iteration-similarity penalty that forces *diversified* suggestions, (b) query-space novelty scoring, and (c) an explicit "no-progress" detector that escalates strategy instead of repeating, should push accuracy past their 94.7% ceiling. Details in §22 and in the implementation plan.

---

## 8. The prototype (Section II-C, end + §III)

Deliberately built so the paper is not "just theoretical design."

| Aspect | Implementation |
|---|---|
| Interface | single-page web UI taking symptom/history input |
| Execution | agent structure executed via **REST APIs** |
| Output | annotated report; per run appends the **source used**, **potential sources of variation**, and a **calibrated confidence score** |
| Feedback trigger | cases below the confidence threshold automatically trigger the feedback loop for reanalysis |
| Latency | real-time; typical case analysis **2–4 minutes** |
| Auditability | **all interactions retained in auditable logs** |
| Privacy | testing with **synthetic, de-identified cases only**; **no persistent patient data retained**; **role-based access control** |

---

## 9. Experimental setup (Section III)

### 9.1 Hardware

| Component | Spec |
|---|---|
| Machine | HP Victus 16 laptop |
| CPU | Intel Core i5-14500HX, 2.60 GHz, 14 cores |
| RAM | 16 GB, 5600 MT/s |
| GPU | NVIDIA GeForce RTX 4050, **8 GB VRAM** |
| Storage | 954 GB SSD |
| OS | Windows 11, 64-bit |
| Libraries | latest available PyTorch, Transformers, Qdrant |

> **Panel trap.** 8 GB VRAM cannot host Qwen2-72B, so the evidence scanner is necessarily an **API call**, not local inference. This matters for latency (the paper acknowledges "computational cost is higher than that of single models" as a limitation) and it means the entire system's speed is bounded by hosted-API latency.

### 9.2 Datasets

Three biomedical NLU benchmarks, chosen to span knowledge types and reasoning levels and to allow comparison against literature:

| Dataset | Nature | Format | n evaluated |
|---|---|---|---|
| **MedQA** | Postgraduate medical knowledge; diagnostic reasoning | Multiple choice **A–E**; **exact-match** accuracy | **50** |
| **PubMedQA** | Derived from biomedical articles; infer answers from research question + abstract | **Free-text**; **semantic-similarity** correctness | **50** |
| **MedBullets** | Applied/basic medicine, case-based short questions | **Free-text**; **semantic-similarity** correctness | **50** |

**Why these three:** together they give an accurate view of performance across different knowledge types and reasoning levels, and they are widely used for comparative purposes, enabling objective comparison with other solutions.

**Comparability statement:** to keep the number of data points equal for objective comparison, the authors evaluated **50 questions per dataset under the same settings**. They acknowledge that some compared works used different sample sizes, and cite [31] for the 50-sample figure for MDAgents, Reconcile and AutoGen.

### 9.3 How correctness is computed

**MedQA — per-question one-vs-all reduction.** The one correct option is the positive class; the four remaining options are negatives.

| Prediction | TP | FP | TN | FN |
|---|---|---|---|---|
| Correct | 1 | 0 | 4 | 0 |
| Incorrect | 0 | 1 | 3 | 1 |

Summed over all questions. Under this formulation **micro-precision, micro-recall and micro-F1 equal accuracy by construction**, and specificity quantifies the power to reject the four non-gold options.

**I verified Table 5's MedQA row is arithmetically correct:**

```
47 correct, 3 incorrect
TP = 47,  FP = 3,  FN = 3
TN = 47×4 + 3×3 = 188 + 9 = 197
Precision = 47/(47+3) = 0.940  ✓ paper: 0.94
Recall    = 47/(47+3) = 0.940  ✓ paper: 0.94
F1        = 0.940               ✓ paper: 0.94
Specificity = 197/(197+3) = 0.985  ✓ paper: 0.985
```

The authors are competent on arithmetic. **Note the nuance to raise if asked:** per-question accuracy is 47/50 = 0.94, but per-*option-decision* accuracy over all 250 option-slots is (47+197)/250 = **0.976**. Both are "correct"; the paper reports the per-question one.

**PubMedQA / MedBullets — fixed semantic-similarity rule.** Let `φ(·)` be the sentence-embedding function, `u = φ(ŷ)` for the prediction and `v = φ(y)` for the reference:

```
s(ŷ, y) = cos(u, v) = (u · v) / (‖u‖ · ‖v‖),    s ∈ [−1, 1]
```

Threshold **θ = 0.80** fixed **a priori** to reflect high semantic agreement and to balance precision–recall on a held-out development split. The authors state lower values inflated false positives while higher values substantially reduced recall. Under this binary criterion micro-P = micro-R = micro-F1 = accuracy, reported anyway "for completeness."

> **Panel trap.** θ = 0.80 is a *self-selected grading knob*. Semantic similarity is not correctness. A prediction of "the patient likely has community-acquired pneumonia with parapneumonic effusion" scores high cosine similarity to a gold answer about the same thing while being clinically wrong in a detail that matters. Always report **exact-match alongside** semantic similarity — the paper promises this in Table 6's discussion but never actually shows the exact-match column.

---

## 10. All tables, transcribed

### Table 1 — Feature-based comparison of CDSS approaches

| Feature | Traditional LLMs [17] | Single Model | KG4Diagnosis [16] | Medical Agents [17] | **This Study** |
|---|---|---|---|---|---|
| **Type** | Limited | Limited | Multi-Agent | Multi-Agent | Multi-Agent |
| **Dynamic Web Access** | No | No | No | No | **Yes** |
| **Multi-Source Data Fusion** | No | No | No | No | **Yes** |
| | | Limited | Limited | Limited | **Yes** |
| **Prominent Limitation** | Static information, risk of hallucinations | Static information, risk of hallucinations | A structure closed to real-time data and the web | Integration of external current information is weak | — |

*OCR caveat: the citation numbers in the Table 1 headers are inconsistent with the paper's reference list (the KG4Diagnosis row cites [16], which is actually the radiology-trial paper, and the Medical Agents row cites [17], which is actually KG4Diagnosis). The table's captions and reference numbers appear misaligned.*

> **Useful slide.** This table is the paper's positioning diagram. The "This Study" column is the only one with both dynamic web access and multi-source fusion.

### Table 2 — Parameters used in the analysis

| Component | Analysis Parameter | Value |
|---|---|---|
| RAG (retrieval) | Similarity criterion | **Cosine similarity** |
| RAG (vector dimension) | Embedding dimension | **768-D embedding** |
| RAG (text segmentation) | chunk_size / overlap | **512 / 64** (fallback: **500 / 80**) |
| RAG (text segmentation) | Term boosting | **Active (baseline)** |
| Adaptive feedback | "Medical term boosting" | **Active (baseline)** |
| Adaptive feedback | Acceptance threshold | **τ = 0.65** |
| Adaptive feedback | Max. iterations | **3** |

### Table 3 — Dataset summaries and number of tests

| Dataset | Short technical specification | Number of tests |
|---|---|---|
| MedQA | Postgraduate medical knowledge; multiple choice (A–E); diagnostic reasoning measure | **n = 50** |
| PubMedQA | Derived from biomedical articles; inference from research question + abstract; free-text answer | **n = 50** |
| MedBullets | Applied/basic medicine and case-based short questions; free-text response | **n = 50** |

### Table 4 — Comparison of models on MedQA, PubMedQA and MedBullets

| Category | Model | MedQA | PubMedQA | MedBullets | Related study |
|---|---|---|---|---|---|
| **No agent** | Llama 3 | 78.1 | 66.3 | 68.5 | [16], [31], [33] |
| **No agent** | GPT-4 | 79.6 | 75.4 | 66.2 | [31], [32] |
| **No agent** | OpenBioLLM-70B | 75.1 | 79.3 | 58.4 | [34] |
| **No agent** | DeepSeek-R1 | **92.0** | 76.2 | 79.2 | [18] |
| **No agent** | QWQ-32B | 78.6 | 77.8 | 54.2 | [18] |
| **Adaptive** | MDAgents | 88.7 | 75.0 | **80.8** | [18] |
| **Multi-agent (single-model)** | ER | 81.9 | 56.0 | 76.0 | [18] |
| **Multi-agent (multi-model)** | Medprompt | 82.4 | 51.8 | 71.0 | [18] |
| **Multi-agent (multi-model)** | Reconcile | 81.3 | **79.7** | 59.5 | [18] |
| **Multi-agent (multi-model)** | AutoGen | 60.6 | 77.3 | 55.3 | [18] |
| **Multi-agent (multi-model)** | DyLAN | 64.2 | 73.6 | 57.3 | [18] |
| — | **This Study** | **94.0** | **88.0** | **84.0** | This |

**Read this table carefully — it tells the whole story.** Almost every baseline number is *borrowed from another paper* (the "Related study" column). Only "This Study" was run by these authors. Therefore:

- The comparison is **not** apples-to-apples. Different papers, different prompts, different sample counts, different evaluation dates, different hardware.
- The paper is transparent about this, which earns it credit, but the deltas it quotes are therefore not defensible as "improvements over SOTA."
- Note that the baselines come overwhelmingly from **[18] MDAgents** — a single NeurIPS paper. And MDAgents' own reported MedQA figure is 88.7%, but that number appearing in six different rows means it is one experimental campaign, not six.

**Independent re-analysis of "This Study" vs each MedQA baseline (all n = 50, two-proportion z-test):**

| Baseline | Accuracy | Δ vs ours | z | p | Significant at α = 0.05? |
|---|---|---|---|---|---|
| DeepSeek-R1 | 0.920 | +0.020 | 0.39 | **0.695** | ❌ **No** |
| MDAgents | 0.880 | +0.060 | 1.05 | **0.295** | ❌ **No** |
| ER / Medprompt / Reconcile | 0.820 | +0.120 | 1.85 | **0.065** | ❌ No (marginal) |
| GPT-4 | 0.800 | +0.140 | 2.08 | 0.037 | ✅ Yes |
| Llama 3 / QWQ-32B | 0.780 | +0.160 | 2.31 | 0.021 | ✅ Yes |
| OpenBioLLM-70B | 0.740 | +0.200 | 2.73 | 0.006 | ✅ Yes |
| DyLAN | 0.640 | +0.300 | 3.68 | 0.0002 | ✅ Yes |
| AutoGen | 0.600 | +0.340 | 4.04 | 0.0001 | ✅ Yes |

| Dataset | Ours | Best borrowed baseline | z | p | Significant? |
|---|---|---|---|---|---|
| MedQA | 0.940 (47/50) | 0.920 (46/50) | 0.39 | 0.695 | ❌ No |
| PubMedQA | 0.880 (44/50) | ≈0.797 (40/50) | 1.09 | **0.275** | ❌ **No** |
| MedBullets | 0.840 (42/50) | ≈0.792 (40/50) | 0.52 | **0.603** | ❌ **No** |

**Conclusion: only MedQA vs the weak baselines is significant. The paper's three headline claims are statistically indistinguishable from the best published numbers.**

### Table 5 — Overall and micro-averaged metrics

| Dataset | MedQA | PubMedQA | MedBullets |
|---|---|---|---|
| N (evaluated) | 50 | 50 | 50 |
| Correct | 47 | 44 | 42 |
| Incorrect | 3 | — | 8 |
| **Accuracy** | 0.94 | 0.88 | 0.84 |
| Precision (micro) | 0.94 | 0.88 | 0.84 |
| Recall (micro) | 0.94 | 0.88 | 0.84 |
| F1-score (micro) | 0.94 | 0.88 | 0.84 |
| **Specificity** | **0.985** | 0.88 | 0.84 |

**Why this table is nearly uninformative:** in all three columns precision, recall, F1 and (except MedQA) specificity are **identical to accuracy**. The paper even concedes they "coincide with accuracy" and are reported "for completeness." Reporting four identical numbers as four metrics reads as padding. The single non-trivial number is MedQA specificity = 0.985, which does add information.

### Table 6 — System performance at different iteration numbers

*Setting: 20 classically prepared MedQA questions.*

| Iteration | Accuracy | Processing time |
|---|---|---|
| 1 | **86.3%** | 2.5 minutes |
| 2 | **91.5%** | 3.0 minutes |
| 3 | **94.7%** | **3.7 minutes** |
| 4 | **95.1%** | 4.4 minutes |
| 5 | **95.5%** | 5.5 minutes |

*Parenthetical deltas in the original table were garbled in extraction; the arithmetic deltas are +5.2, +3.2, +0.4, +0.4 points.*

**The honest reading:** iteration 1 → 3 delivers +8.4 points; iteration 3 → 5 delivers **+0.8 points total for 46% more wall-clock time**. The authors cap at 3 for exactly this reason. The value of the feedback loop is real but front-loaded.

**Statistical check:** at n = 20, iteration 1 (86.3%, ≈17.3 correct) vs iteration 3 (94.7%, ≈18.9 correct) gives z = 0.91, **p = 0.37**. Not significant. (Note also 86.3% × 20 = 17.26 correct — a non-integer, so the reported precision exceeds what 20 questions can express. One question = 5 points. Every value in this table should be a multiple of 5%.)

### Table 7 — Ablation: system performance with components turned off

*Setting: 20 MedQA test questions, same questions, one component removed at a time. Full-system reference on this subset is ≈94.7%.*

| Component removed | Accuracy | Deficiency observed in the system |
|---|---|---|
| **Clinical Text Clarifier** | **80.0%** | Semantic distortion in input texts made symptom detection difficult. |
| **Symptom RAG Analyzer** | **65.0%** | The production of literature-based diagnoses has weakened significantly. |
| **Evidence-Based Scanner** | **70.0%** | Current clinical information could not be obtained from the web, so old information was used. |
| **Clinical Data Fusion** | **60.0%** | Conflicting data could not be combined, and inconsistency in recommendations was observed. |
| **Adaptive Optimizer** | **70.0%** | Low-confidence answers were not flagged / the flagging ratio dropped sharply. |
| **Medical Term Boosting** (feature disabled) | **80.0%** | Prioritization of critical clinical expressions degraded, reducing diagnostic accuracy. |

**This is the paper's strongest evidence.** Statistical check at n = 20:

| Comparison | Δ | z | p | Significant? |
|---|---|---|---|---|
| Full (19/20) vs Fusion removed (12/20) | −34.7 pts | 2.62 | **0.0088** | ✅ Yes |
| Full (19/20) vs RAG Analyzer removed (13/20) | −29.7 pts | 2.34 | **0.019** | ✅ Yes |

**Component importance ranking (accuracy retained when removed):**

```
Data Fusion      60.0%   ████████████░░░░░░░░░  ← most critical
RAG Analyzer     65.0%   █████████████░░░░░░░░░
Web Scanner      70.0%   ██████████████░░░░░░░░
Adaptive Optim.  70.0%   ██████████████░░░░░░░░
Text Clarifier   80.0%   ████████████████░░░░░
Term Boosting    80.0%   ████████████████░░░░░
```

**The key architectural lesson:** fusion matters most. Removing the ability to *reconcile* evidence is worse than removing any individual evidence source. That is a genuine, transferable insight and it is the one result here with statistical backing.

**Two internal inconsistencies to be ready for:**
1. The text says disabling *"Medical Term Strengthening"* reduced accuracy to 80.0%, but Table 7's 80.0% row is *"Clinical Text Clarifier"* — and 80.0% also appears for the *Medical Term Boosting* row. Same number, three candidate components.
2. §II-C relabels the ablation rows as *configurations*: "RAG-only (65.0%) and web-only (70.0%)." But removing the Symptom RAG Analyzer does not produce a "RAG-only" system — it produces a system with **no** RAG. The labels are wrong.

---

## 11. Timings and cost (Section III)

### 11.1 Measured component latencies

| Component | Latency |
|---|---|
| RAG agent — issuing the request | ~1 s |
| RAG agent — vector lookup response | ~1 s |
| RAG agent — processing and deciding on the response | ~10 s |
| Inter-agent output transfer | **< 1 s** (negligible) |
| Web agent — **per single page** | **~45 s** |
| Web agent — typical total (1–2 additional pages) | ~1–2 minutes |
| Model evaluation/decision time (model differences) | ~8–12 s average |

### 11.2 End-to-end

| Configuration | Wall-clock |
|---|---|
| Single iteration (RAG call + web crawl + fusion) | **~2–3 minutes** |
| 2–3 iterations | **~3–4 minutes** |
| Mean response, 3 iterations, MedQA benchmark | **3.7 minutes** |
| 50-question evaluation incl. review and annotation | **~2–3 hours** |

### 11.3 Where the time goes — the key insight

> *"These measurements show that the system's temporal cost is largely due to web access and iterative optimization steps, while inter-agent communication does not contribute significantly to the total delay."*

**Concretely: the web crawler is ~85–90% of the latency.** RAG + LLM reasoning is ~12 s. One web page is ~45 s. This single fact drives most of our latency optimization work (§22.6).

**Total cost is never reported.** Four different frontier APIs are called multiple times per case. If you are asked "what does one case cost?", the honest answer is *"the paper does not report API cost at all — only wall-clock — which is itself a limitation for any deployable CDSS."* Then give your own measured cost/case.

---

## 12. Results interpretation (Section III, end)

**The authors' three-part attribution of the gains:**

1. Domain-specific retrieval
2. The fusion controller that cross-checks RAG and web evidence
3. The feedback loop that re-queries low-confidence cases

**Their three reasons the performance level was reached:**

1. Generating qualified queries from vectors containing multi-layered semantic relationships is crucial
2. The optimization approach combining these queries with agent algorithms supporting incremental learning improves accuracy and efficiency
3. A rational division of labour among agents, plus selecting the most appropriate LLM per task, deepened each agent's expertise

**Their framing of what the experiments confirm:**

1. **Adaptive optimization** — accuracy 86.3% → 94.7% after 3 iterations on MedQA (Table 6); removing the Adaptive Optimizer drops accuracy to 70.0% (Table 7).
2. **Customized BioBERT RAG** — removing the Symptom RAG Analyzer drops accuracy to 65.0%; disabling "Medical Term Strengthening" drops it to 80.0%.
3. **WEB + RAG fusion** — removing the Clinical Data Fusion Agent drops accuracy to 60.0%, the most significant degradation, "demonstrating that cross-checking and data fusion are essential for reliability."

**Qualitative ablation observations worth quoting:**
- Removing the Evidence-Based Scanner → "more stale and out-of-scope recommendations"
- Removing the Adaptive Optimizer → shuts down the ability to scan and optimize over time; "the precision in the decisions being made was severely lessened"
- Removing the Clinical Data Fusion module → impaired convergence of knowledge across sources, "resulting in either varied or downright wrong outputs"

**Their closing claim on synergy:** the system is *not* composed of independently operative pieces but becomes a unified whole because of the competence developed by each agent in its specific area of responsibility.

---

## 13. Limitations (Section IV) — the authors' own

| # | Limitation | Their proposed remedy |
|---|---|---|
| 1 | Evaluation on **three generic datasets**; none widely cover the range of real-world patients. Rare cases and multi-disease complexity untested. | Later clinical studies should investigate |
| 2 | Web/literature benefit **depends on quality and accessibility** of the actual information sources used. | Hybrid solutions integrating official clinical guidelines + evaluating source reliability |
| 3 | Multi-agent architecture has **higher computational cost** than single models; limiting in time-critical clinical environments. | More efficient models or cloud-based integration |
| 4 | The system is **a support tool, not a standalone diagnostic tool**; the conclusion for the course of action lies ultimately with the expert physician. | — (framing as a feature) |
| 5 | **Hallucination risk cannot be removed entirely.** Data fusion + feedback loop minimize it significantly. | — |
| 6 | Additional limitations: extra computational load in real-time clinical environments, repeated process resource utilization, **variability of model performance in rare clinical environments**, hallucination risk. | Multicenter clinical testing, expert interactive assessment protocols, stronger ethical/regulatory dimension |

**Conclusion text:** future work should focus on multicenter clinical validation, resource-efficient optimization, and strengthening the ethical/regulatory dimension.

> **The strategic read.** Limitations 1, 3 and 6 are *directly attackable by your project*. Limitation 1 (n=50, generic datasets) → you run the full public splits. Limitation 3 (computational cost) → you measure and report latency/cost properly and optimize. Limitation 6 (variability on rare cases) → you add a rare-case / multi-disease stress set. **Framing your work as "we make this claim rigorous and this limitation tractable" is much stronger than "we built something different."**

---

## 14. Conclusion (Section V)

- Proposes an architecture of multiple LLM agents to make CDSS more flexible, precise and contextually responsive.
- Unlike rigid solutions, involves **semantic analysis, contextual inference, and adaptive feedback loops**.
- Accuracy targets achieved: 94% MedQA, 88% PubMedQA, 84% MedBullets — "all of which represent a substantial improvement from the previously defined benchmarks."
- Gains attributed to: combined access to specialized biomedical RAG to enrich dynamic web searches, plus continuous adaptation from the feedback loop which increased accuracy from 86.3% to 94.7% on MedQA.
- **"Clinical intelligence partner"** framing: builds upon a physician's cognitive schema by synthesizing a multi-faceted complex patient presentation, bridging symptoms to the evidence base, establishing connection to the scope of decision-making by providing an always-current knowledge framework. "This capacity may be especially relevant in high-stakes clinical encounters."
- Emphasizes **LLMs as dynamic support systems combined with clinical expertise, not merely data processing tools.**

---

## 15. Independent audit — what I verified and what is wrong

This section is the analytical core. Everything here was computed from the paper's own reported numbers.

### 15.1 The sample-size problem (fatal for the headline claims)

With **n = 50**, a 95% Wilson confidence interval is roughly **±10 points**. The paper reports none.

| Dataset | Reported | Count | **Wilson 95% CI** | Width |
|---|---|---|---|---|
| MedQA | 94.0% | 47/50 | **[83.8%, 97.9%]** | 14.1 pts |
| PubMedQA | 88.0% | 44/50 | **[76.2%, 94.4%]** | 18.2 pts |
| MedBullets | 84.0% | 42/50 | **[71.5%, 91.7%]** | 20.2 pts |

**The MedQA interval [83.8%, 97.9%] contains the accuracy of eight of the eleven baselines in Table 4** — including MDAgents (88.7%), GPT-4 (79.6% is just below), QWQ-32B (78.6%), ER/Medprompt/Reconcile (82.0%). It even extends down toward ER/Medprompt/Reconcile at 82.0%. You cannot claim superiority from an interval that swallows your competitors.

### 15.2 The "statistically significant" claim has no test behind it

The Conclusion states "Empirical evaluations demonstrate the system to have **statistically significant performance improvement** over state-of-the-art methods." **No statistical test appears anywhere in the paper.** There is no p-value, no confidence interval, no test name, no standard deviation, no seed, and no repeated run.

My independent re-analysis (see §10, Table 4): **none of the three headline claims is significantly better than the strongest published baseline** (p = 0.695, 0.275, 0.603).

If the panel asks "is your improvement statistically significant?", the answer is **no**, and the graceful response is: *"You're right that the paper doesn't test it. When we ran the power analysis at n=50 we found a 2-point gap over DeepSeek-R1 would need several thousand paired items to detect. That's why our work runs the full public splits with McNemar tests — the architecture claim held up, but the paper's evidence for it was under-powered."*

### 15.3 Precision/recall/F1 are not independent metrics here

The paper's own text says micro-P = micro-R = micro-F1 = accuracy *by construction*. Reporting three copies of accuracy as separate metrics invites the question "did you report four numbers or one?" Only specificity 0.985 carries information.

Additionally, a 0.65 confidence threshold that is never calibrated means **precision in the clinical sense — "when the system says it's confident, how often is it right?" — is never measured.** That is the metric that matters for a CDSS, and it is exactly what a system that claims "calibrated confidence" should report.

### 15.4 The vector database numbers do not reconcile

Stated: "18M embeddings from 200,000 chunks for 1,000 PDFs", "final index size 2.1 GB", "38 minutes".

| Claim | Arithmetic check | Verdict |
|---|---|---|
| 200,000 chunks → 18M embeddings | 200k chunks produce 200k embeddings (1:1) | ❌ **Impossible** unless "18M" means something unstated (tokens? passages across re-indexing?) |
| 1,000 PDFs → 18M chunks | 18,000 chunks/PDF. But 200,000/1,000 = **200 chunks/PDF** | ❌ **90× discrepancy** |
| 200,000 × 768 × fp32 | = **0.61 GB** | ❌ Paper says 2.1 GB |
| 200,000 × 768 × fp64 | = 1.23 GB | ❌ Still not 2.1 GB |
| Chunk size 512 tokens ≈ 2 KB | 200,000 × 2 KB = **0.4 GB** raw text | 2.1 GB is ~5× plausible *only* if you include HNSW graph + payload duplication at some fp setting |
| 38 minutes for 18M embeddings on one RTX 4050 | BioBERT-base ≈ ~1,500 sentences/s batched on a 4050 → 18M / 1500 ≈ 3.3 hours | ❌ Not feasible; 200k chunks would take ~2 min |

**Conclusion:** the throughput, size and count figures were not measured together. Most likely 200,000 chunks is right and 18M / 2.1 GB are wrong (or refer to a different configuration). Note also the corpus is described two ways: **"889 publicly available medical articles"** (§I) vs **"1,000 articles"** (§II-B).

**This is a low-risk, high-value presentation line:** *"I re-derived the paper's ingestion figures and they don't reconcile — 200k chunks of 768-d float32 is 0.61 GB, not 2.1 GB. We report measured pages/sec, measured index bytes and a content-hash manifest for our own corpus so the numbers are auditable."*

### 15.5 "Logarithmic" processing-time growth is linear

Text: *"improvement in accuracy slowed down with an increase in the number of iterations, and thus the processing times were increased logarithmically."*

Table 6: 2.5 → 3.0 → 3.7 → 4.4 → 5.5 minutes.

| Iteration | Time | Δ | ratio vs iteration |
|---|---|---|---|
| 1 | 2.5 | — | — |
| 2 | 3.0 | +0.5 | 1.20× |
| 3 | 3.7 | +0.7 | 1.23× |
| 4 | 4.4 | +0.7 | 1.19× |
| 5 | 5.5 | +1.1 | 1.25× |

Uniformly ~1.2× per iteration — **linear with a constant factor**, because each iteration re-runs the same fixed set of agents. Note the confusion: the paper's `O(L)`, `O(log N)+O(k)` and `O(k+R)` expressions are *per-iteration*, and multi-iteration cost is simply `iterations × per-iteration cost`. There is no logarithm in iterations.

### 15.6 The "14.2% improvement" does not follow

Text: *"improving output quality by 14.2%."* The cited evidence is 86.3% → 94.7%.

```
Absolute:  94.7 − 86.3 = +8.4 percentage points
Relative:  8.4 / 86.3  = +9.7%
```

Neither is 14.2%. Perhaps it was computed against a different baseline (e.g. 82.4 → 94.7 = +15.0%, or 82.8 → 94.7 = +14.4% — close but not exact). **The number is not derivable from the evidence cited.**

### 15.7 "Consistency increased 1.8 times" is undefined

No consistency metric is defined, computed, or reported anywhere in the paper. Given the context (the sentence also covers the 0.65-threshold reprocessing), the likely intent is "the fraction of low-confidence outputs that got reprocessed increased 1.8×," but that is inference on our part. **Treat it as an unsupported claim.**

### 15.8 n = 20 cannot express Table 6's precision

One MedQA question = 5.0 percentage points. All Table 6 accuracies should therefore be multiples of 5%: 85.0, 90.0, 95.0, 95.0, 95.0. Reported values are 86.3, 91.5, 94.7, 95.1, 95.5. **86.3% implies 17.26 correct out of 20** — impossible.

Either (a) more than 20 questions were actually used, contradicting the text, or (b) the values are averaged over multiple partial runs, which is never stated. This is a genuine, checkable inconsistency.

### 15.9 94% vs 94.7% — which is the result?

| Location | Value | n |
|---|---|---|
| Abstract | "accuracies of 94%, 88%, and 84%" | 50 |
| §III main text | "94% on MedQA, 88% on PubMedQA, 84% on MedBullets" | 50 |
| Table 4 | "This Study" MedQA = **94.0** | 50 |
| Table 5 | 47/50 = **0.94** | 50 |
| §II-C | "the fused system reached **94.7%**" | — |
| Table 6 | **94.7%** | 20 |
| §V Conclusion | "increased our accuracy from 86.3% to **94.7%** on the MedQA dataset" | 20 |

So there are two figures: **94.0%** (the 50-question main evaluation, which is what Table 4 and Table 5 use) and **94.7%** (the 20-question iteration study). The §II-C "+29.7 / +24.7 point" improvement deltas are computed against the 20-question figure while being presented as the headline result. **If asked for "the" number, answer 94% and explain the 94.7% is the 20-question ablation subset.**

### 15.10 Minor citation and labelling issues

| Issue | Detail |
|---|---|
| Table 1 headers misaligned | KG4Diagnosis row cites [16] (which is the radiology paper); Medical Agents row cites [17] (which is KG4Diagnosis). Captions shifted by one. |
| Ref [27] mismatch | Cited to justify choosing Gemini-2.0-Flash for RAG, but Balestri (2025) is *"Gender and content bias in large language models: A case study on Google Gemini 2.0 flash experimental."* A bias study is not a capability benchmark. |
| Ref [29] mismatch | Cited for GPT-4o-mini, but Siddiky et al. is *"Optimizing AI language models: A study of ChatGPT-4 vs. ChatGPT-4o"* — GPT-4o, not 4o-mini. |
| Ref [28] mismatch | Cited for Qwen2-72B, but Tian et al. is *"Think twice: Enhancing LLM reasoning by scaling multi-round test-time thinking"* — a test-time-scaling paper, not a model report. |
| Ref [6] | Öğdü et al., cultural-heritage damage detection by deep learning — unrelated to the clinical CDSS; appears in the LLM-background citation cluster. |
| Ref [30] access date | "Accessed: Aug. 22, 2023" for a 2023 dataset — plausible but the citation is formatted inconsistently with the rest of the list. |
| Table 7 conflict | Text says "Medical Term Strengthening" → 80.0%; Table 7 has both "Clinical Text Clarifier" and "Medical Term Boosting" at 80.0%. |
| "RAG-only / web-only" | Ablation rows relabelled as configurations in §II-C. Removing the RAG Analyzer ≠ RAG-only. |

None of these are fatal, and naming them signals real critical reading. **Do not be smug about it** — frame as "these are the checks a reviewer should run."

### 15.11 What the paper gets genuinely right

Be able to say this too — it is your foundation.

1. **Fusion > any single source.** The ablation is the one statistically sound result and it is the most interesting finding in the paper.
2. **Retrieval grounding with a support requirement.** "Candidates are retained only when at least one retrieved passage explicitly supports them" makes every diagnosis traceable to a cited span. That is the right design and it is rarely done.
3. **Deterministic retrieval tier + live web tier.** A frozen, auditable index for reproducibility plus a live layer for freshness is a sound two-tier architecture.
4. **Role-specialized models.** Assigning a different model per task is empirically sensible, even if the supporting citations are mismatched.
5. **Query-time adaptation over weight updates.** No fine-tuning means the knowledge base is updated by re-indexing, which is far cheaper and more auditable for clinical governance.
6. **Self-diagnosed convergence ceiling.** The authors identified and stated their own iteration-saturation problem. Most papers hide that.
7. **Honest positioning.** Explicitly stating "we do not present a new algorithmic paradigm" and listing competitors' limitations in Table 1 is good scientific hygiene.
8. **Privacy-by-design.** Synthetic de-identified cases only, no persistent patient data, role-based access. Appropriate for a clinical prototype.

---

## 16. Datasets — deeper reference

### 16.1 MedQA
- **What it is:** USMLE-style multiple-choice questions on postgraduate medical knowledge. Designed as a diagnostic-reasoning benchmark.
- **Format:** one stem + N options, single correct answer. The paper uses **A–E (5 options)**.
- **Note:** the original `MedQA-USMLE` release is 4-option (A–D); the 5-option configuration is a distinct release. The paper does not state which. **Confirm this before reproducing.**
- **Scoring:** exact-match on the label.
- **Public sizes:** test split in the order of ~1,270 items — roughly **25× the 50 evaluated**.
- **Known weaknesses:** contaminated by web-scraped training data for frontier LLMs; recall-heavy and US-centric (Step 1/Step 2 CK); does not test multimodal input; does not test long-horizon clinical reasoning.

### 16.2 PubMedQA
- **What it is:** question–abstract pairs from PubMed with yes/no/maybe answers. Artificially constructed from real biomedical papers' title (question) and abstract (context). Ground truth came from expert annotation.
- **Format:** **free-text answer**, then binarized by cosine similarity ≥ **θ = 0.80**.
- **Ground truth labels are yes / no / maybe.** This matters: a sensible alternative is to score exact-match against the label directly, which is fully objective and removes the θ knob entirely.
- **Public sizes:** the `pqa_labeled` test split is ~450 items — **9× the 50 evaluated**.
- **Known weaknesses:** heavily skewed towards "yes" (~60%+); "no" answers are often underdetermined from title alone; it is a *reading comprehension* task more than a *diagnosis* task.

### 16.3 MedBullets
- **What it is:** a question bank for applied/basic medicine and case-based short questions, built for board-exam prep.
- **Format:** short case vignettes, free-text responses per the paper.
- **Label access:** MedBullets has historically been a closed/licensed corpus with a public API under terms of use. **The paper does not say how it obtained the labelled 50.** Reproducing this dataset may be blocked. **Verify access early** — it is the single biggest dataset risk in the project.
- **Known weaknesses:** mixed difficulty and mixed specialities in a small sample; original labels are multiple-choice, so forcing free-text grading is the authors' own methodological choice, not a property of the dataset.

### 16.4 Why this trio is a weak evaluation suite

| Dimension | Coverage | Gap |
|---|---|---|
| Knowledge type | MC recall, abstract comprehension, case vignettes | No longitudinal records, no labs-over-time, no differential reasoning chains |
| Reasoning depth | Mostly single-shot QA | No multi-step reasoning, no tool-use necessity, no hypothesis elimination |
| Answer space | Closed (MC) + short free text | No calibration-friendly numeric/lab-estimation tasks, no ICD/SNOMED coding tasks |
| Clinical realism | Exam-style vignettes | **Zero** real EHR data, zero rare/complex multi-system presentations |
| Dosing/actionability | None | No treatment-selection, contraindication, or interaction tasks |

**This is exactly the authors' stated Limitation 1, and it is our opportunity.**

---

## 17. Metric-by-metric: how we beat them

### 17.1 The master comparison table

| Their metric | Their value | Why it is beatable | Our concrete method | Expected effect |
|---|---|---|---|---|
| **MedQA accuracy** | 94.0% (47/50), CI [83.8, 97.9] | n=50 gives ±10 pts; not significant vs DeepSeek-R1's 92% (p=0.70); MedQA has a public test split 25× larger | Evaluate the full public split with deterministic seeds, 5 repeats; report Wilson CI + **McNemar** paired test vs baselines we run ourselves | Narrow CI to roughly ±1.5 pts; either a genuinely significant lead, or an honest finding that the architecture saturates near 92–94% |
| **PubMedQA accuracy** | 88.0% (44/50) | Graded by self-chosen cosine threshold θ=0.80; the dataset has yes/no/maybe labels, so exact match is free | Score **exact-match against the gold label** as primary; report semantic similarity as secondary; sweep θ ∈ {0.70…0.95} and show the accuracy-vs-θ curve | Removes the grading-knob objection; likely a higher or equal number that is *defensible* |
| **MedBullets accuracy** | 84.0% (42/50) | Small sample, unclear provenance, free-text grading | Verify/licence the dataset first; if blocked, substitute a documented public substitute and say so | Either a real comparison or an honest, clearly-flagged substitution |
| **Micro P/R/F1** | = accuracy (degenerate) | Informative only under multi-class/imbalanced scoring | Report **macro-F1**, per-class F1 on yes/no/maybe, and **AUROC** for the yes-vs-rest decision | Genuinely new information; directly answers "your P/R/F1 are just accuracy" |
| **Specificity** | 0.985 MedQA, 0.88 others | Only computed under the degenerate binary reduction; no abstention analysis | Report **selective prediction**: an accuracy-vs-coverage curve as the confidence threshold varies, plus **risk-coverage (AURC)** | This is the metric a real CDSS needs; we are the first to report it |
| **τ = 0.65 confidence** | asserted, never calibrated | No formula, no weights, no calibration metrics | Implement the formula the paper omits: `c = w₁·norm(rag_sim) + w₂·evidence_grade + w₃·concordance − w₄·conflict`, then **temperature-scale on a held-out dev split**; report **ECE and Brier before vs after** | Turns their weakest claim into our strongest contribution |
| **Iteration gain** | 86.3 → 94.7 on n=20 (+8.4 pts), p=0.37 | n=20 → 1 question = 5 pts; the reported values are not achievable at n=20; the loop repeats identical suggestions past iteration 3 (their own diagnosis) | Re-run on **n ≥ 300** with CI; add **iteration-diversity penalty** + **novelty score** + **no-progress escalation detector** so iterations keep finding new evidence | Push past their 94.7% plateau; the plateau is an artifact they identified but never fixed |
| **Ablation** | 60–80% on n=20 | n=20; only p<0.02 for two rows; one row label conflicts with the text | Re-run every ablation at **full n, 5 seeds**, report mean ± std and CIs; add our own components (reranker, calibrator, abstention, safety agent) | Turns the paper's best result into a much stronger claim and adds new ablations they never ran |
| **Retrieval quality** | **never measured** | They built a vector DB and reported zero retrieval metrics | Report **recall@k, MRR@10, nDCG@10** against gold supporting passages; ablate BioBERT vs **MedCPT / BioClinicalBERT / SPECTER2 / BGE-M3 / OpenAI text-embedding-3**; ablate dense-only vs **hybrid BM25+dense** vs **+ cross-encoder reranker** | Our retrieval section alone may be a publishable contribution — this is a total white space in the paper |
| **Latency** | 2.5–5.5 min; web ≈45 s/page (85–90% of total) | Live web crawling is the entire bottleneck, plus zero caching | **Snapshot** web evidence offline for reproducible eval (keep live mode for the demo); parallelize the scanner across hypotheses; cache Serper + LLM responses; stream to UI | 5–20× latency reduction; p50/p95 reported. Turns their Limitation 3 into our headline engineering result |
| **Corpus** | 889 or 1,000 PDFs + 5,630 patient texts; internally inconsistent counts | Numbers don't reconcile; no manifest, no hashes | 5,000–10,000 documents incl. NICE/WHO/StatPearls/guidelines; emit a **SHA-256 manifest**, measured pages/sec, measured index bytes | Auditable, and a corpus large enough for meaningful retrieval gains |
| **Embedding model** | BioBERT-mnli-…-stsb, 768-d | BERT-base from 2019; weak on clinical text, no cross-encoder training, max 512 tokens, not retrieval-tuned | Embedding bake-off + hybrid retrieval + reranker (above) | Typically +3–8 pts on retrieval-dependent tasks |
| **Evidence grading** | qualitative hierarchy (guidelines > SR > case reports) | Never quantified, never validated | Explicit weighted scheme with an **Oxford/CEBM level** per source; report it; test whether down-weighting case reports helps | Makes the "evidence assessment process" a real, testable component |
| **Ontology grounding** | "maps to SNOMED CT or ICD-10" — unspecified | No version, no accuracy, unclear if real | Wire a **real UMLS/SNOMED/ICD-10 lookup**, measure **entity-linking F1** as its own metric | A metric the paper has no value for at all; easy points |
| **Safety / hallucination** | "risk significantly minimized" | No metric | Add an **abstention agent** + a **red-team set** (leading-question, contraindication, dosage, out-of-scope, fabricated-vitals) + **harmful-omission rate** and **unsafe-recommendation rate** | Answers the panel's ethics question with numbers instead of adjectives |
| **Determinism** | none; results vary with the live web | Cannot be reproduced | Fully deterministic offline mode with pinned seeds + cached evidence; live mode optional | Reproducibility is a reviewer's first question; having both modes is a strength |
| **Baselines** | borrowed from other papers, different n | The core comparison is invalid | **Run every baseline ourselves at matched n, matched prompts, matched budget, same machine** | Single biggest defensibility gain in the whole project |

### 17.2 Realistic expectation setting

Be honest with yourself about what is achievable:

- **We will very likely not beat 94% on MedQA by much.** MedQA is near-saturated and contaminated; frontier models already reach low 90s. **The win is statistical rigor, not the point estimate.**
- **We will likely win clearly on rigor**: CIs, paired tests, ablations at full n, retrieval metrics, calibration metrics, latency, cost.
- **We have a real chance of a large win on PubMedQA and MedBullets**, because the paper's free-text grading with θ=0.80 is soft and our objective scoring + better retrieval should show a genuine gap.
- **The best publishable contributions are arguably not accuracy at all** — they are the confidence-calibration method and the retrieval evaluation. Consider leading with those in your writeup.

---

## 18. Trade-offs and open tensions

These are genuine, unavoidable tensions in this project. Decide them explicitly; do not discover them mid-implementation.

### 18.1 Faithful reproduction vs. beating the paper

| | `faithful` config | `improved` config |
|---|---|---|
| **Goal** | Reproduce Öğdü et al. exactly so the head-to-head is valid | Maximize accuracy and rigor |
| **Retrieval** | BioBERT dense-only, top-*k*, cosine | Hybrid BM25+dense, reranker, tuned embedder |
| **Confidence** | Paper's un-specified formula, τ=0.65 | Calibrated formula + temperature scaling |
| **Web** | Live Serper crawl | Snapshotted evidence (deterministic) |
| **Feedback** | Plain loop, max 3 | Diversity-penalized, novelty-scored, escalation |

**Trade-off:** we must ship **both**. A single improved system cannot be honestly compared to their numbers, and a single faithful system cannot beat them. Cost: roughly 1.6× total engineering and eval time.
**Resolution:** one codebase, two config profiles. `configs/faithful.yaml` and `configs/improved.yaml`. Everything else identical.
**Risk:** scope creep. **Mitigation:** build `faithful` first and freeze it before touching `improved`. Freeze = no commits to that code path without a note in `docs/DECISIONS.md`.

### 18.2 Free-tier compute vs. evaluation scale — the binding constraint

Full-scale eval: 500 q × 3 datasets × ~6 LLM calls × up to 3 iterations ≈ **27,000 calls**.

| Provider | Free limit | Can it absorb 27k calls? |
|---|---|---|
| Groq free | ~30 RPM, ~1,000–14,400 RPD depending on model | ❌ Days to weeks |
| OpenRouter free | ~200 RPD | ❌ ~4.5 months |
| Gemini free tier | Project-scoped, generous but throttled | ⚠️ Maybe, with backoff |
| **Ollama local** | **Unlimited** | ✅ Yes — this is the answer |

**Trade-off:** local 7–8B models will score materially lower than DeepSeek-R1/GPT-4-class models on MedQA, so "our best accuracy" will not be produced by the free engine.
**Resolution — split the two goals across engines:**
- **Local Ollama** = the **bulk statistical evaluation** (CIs, ablations, iteration sweeps, retrieval experiments). Unlimited, free, deterministic. Its absolute numbers are lower but every *comparison* is internally valid because the engine is held fixed.
- **Free cloud tiers** = the **headline matched-50/50/50 run** and the demo, where we need the paper's model class.
- **Paid tier (optional, if the panel requires it)** = one confirmation run on the paper's exact models to prove the result transfers.
**Always report which engine produced which number.** Mixing engines across a comparison invalidates it — this is exactly the mistake the paper makes with its borrowed Table 4.

### 18.3 Model substitution vs. exact reproduction

Gemini-2.0-Flash and GPT-4o-mini are legacy/retired. DeepSeek-R1 and Qwen2-72B are available but not free-tier.

| Paper agent | Paper model | Free substitute | Fidelity risk |
|---|---|---|---|
| Clinical Text Clarifier | DeepSeek-R1 | Qwen3 / Qwen2.5-7B-Instruct (Groq/Ollama) | Low — this is schema extraction, easy task |
| Symptom RAG Analyzer | Gemini-2.0-Flash | Gemini free tier, else Qwen3-27B | Low–Medium |
| Evidence-Based Scanner | Qwen2-72B | Qwen3-8B/27B | **High** — dense-evidence summarization is exactly where size matters |
| Clinical Data Fusion | GPT-4o-mini | gpt-oss-20b / Qwen3-8B | **Medium–High** — conflict detection needs strong reasoning |
| Adaptive Optimizer | Gemini-2.0-Flash | Gemini free tier, else Qwen3-27B | Medium |

**Trade-off:** lower engine quality will depress absolute numbers.
**Resolution:** (1) keep the **role split identical** — the architecture claim is about roles, not model brands; (2) record every model ID + version + date in every results row; (3) run a **model-sensitivity ablation**: swap each agent's model across ≥3 options and report the spread. That converts a weakness into a finding, and it is something the paper never did despite making model selection its central claim.
**Hard rule:** never report a number without its model manifest. Non-negotiable.

### 18.4 Determinism vs. live web freshness

| | Offline snapshot | Live web |
|---|---|---|
| Reproducible | ✅ Yes | ❌ No |
| Current evidence | ❌ Snapshot date | ✅ Live |
| Latency | ✅ Cached, fast | ❌ ~45 s/page |
| Demonstrates the architecture | Partially | ✅ Fully |

**Trade-off:** a snapshot undercuts contribution #1 (dynamic web access) in the demo.
**Resolution — support both, clearly labelled.** Evaluation runs on the snapshot (with a snapshot date and hash manifest). The live demo shows real-time NICE/PubMed retrieval. In any table, mark which mode produced the number. The paper conflates the two, which is part of why its numbers are not reproducible.

### 18.5 Rigor vs. time-to-presentable

Full rigor = matched-50 + 500/dataset × 3 datasets + 6 ablations × 5 seeds + retrieval bake-off + calibration + safety set + UI. That is weeks of wall-clock on free tiers.

**Resolution — a three-tier evidence ladder.** Ship tier 1 for the presentation, keep climbing if time allows.

| Tier | Scope | Cost | When to stop |
|---|---|---|---|
| **T1 — Presentation-ready** | Matched 50/50/50, faithful config, CIs + McNemar, 1 ablation sweep, retrieval metrics on one dataset, latency/cost table, working UI | Days | Minimum for a credible panel session |
| **T2 — Strong** | T1 + 300/dataset improved config + full ablation at n=300, 3 seeds + calibration ECE/Brier + embedding bake-off | ~2 weeks | Recommended target |
| **T3 — Publishable** | T2 + full public splits (1273/450) + 5 seeds + safety red-team + selective-risk curves + live demo | ~4–6 weeks | If you intend to write a paper |

**Recommendation: target T2, present T1, and state the ladder openly in the presentation.** Showing the panel your own rigour ladder is itself impressive.

### 18.6 BioBERT vs. modern retrievers — the honesty tension

BioBERT 768-d was chosen in 2019 for PubMed-domain pretraining. It is now clearly suboptimal for retrieval. Replacing it will produce a large gain — **but it means you are no longer reproducing the paper.**

**Resolution:** keep `faithful` on BioBERT permanently as the reference arm. Run the embedder bake-off inside `improved`. Report both. The narrative becomes *"we reproduced their retriever, showed it is the weakest component, and quantified by how much"* — a better paper than either reproduction or replacement alone.

### 18.7 Improvement claims vs. possible regression

Real risk: we improve rigor and *lose* accuracy. Small free models + a bigger eval set can easily mean our faithful config scores below 94%.

**This is fine, and it must be pre-accepted.** The finding would be *"the reported 94% does not replicate at scale"* — which is a legitimate, valuable, publishable result, and arguably the most important one. **Never let sunk-feelings pressure you into a favourable framing.** Pre-write the framing: *"we measure X at Y% with CI [a,b]; the paper's point estimate falls inside our interval, which means the architecture claim survives but the precision claim does not."*

### 18.8 Scope: what to cut if behind schedule

Cut in this order, and only this order:
1. **T3 items** (full splits, safety red-team, live demo polish) — cut first.
2. **MedBullets** if licensing is blocked — substitute and flag.
3. **Embedding bake-off** beyond 3 models.
4. **The UI** beyond a functional trace viewer — never cut the eval harness.
5. **Never cut:** matched-50 baseline runs, confidence intervals, McNemar, retrieval metrics. Those are the project.

### 18.9 Ethical and regulatory positioning

A CDSS is a medical device under FDA/EU MDR. Non-negotiables in anything you demo or publish:
- Synthetic/de-identified data only. **No real patient data, ever**, without IRB approval.
- Explicit "not a diagnostic device; for clinician decision support; verify independently" in every output.
- Logged, attributable decision traces (the paper does this — keep it).
- An abstention path — the system must be able to say "insufficient evidence."
- Bias analysis across demographics, since the underlying models have documented bias issues (§10, ref [27]).
- Never a dosing recommendation without a dose + units + a citation.

---

## 19. Panel Q&A bank

Answers are written to be spoken. Where the paper is weak, the answer concedes cleanly and redirects — that reads as competence, not defensiveness.

### 19.1 Motivation and problem

**Q. Why do we need yet another CDSS? Aren't there rule-based systems like MYCIN and OntoDiabetic?**
Those work well on closed, well-scoped problems — that's exactly why OntoDiabetic exists and works. The problem is clinical data is heterogeneous, time-varying and noisy, and evidence changes. Fixed rule bases can't adapt to that. This paper's bet is that role-specialised LLM agents plus a grounding layer plus a self-monitoring loop can adapt where static rules cannot.

**Q. The paper cites 44,000–98,000 preventable deaths per year. That's an IOM 2000 figure. Isn't it stale?**
Yes, and I'd treat it as an order-of-magnitude motivation rather than a current statistic. The underlying point — diagnostic error is a large, materially human-caused problem — still holds. Our work doesn't depend on the specific number.

**Q. What's genuinely new here?**
The authors are honest: it's not a new algorithmic primitive, it's a systematic integration. The genuinely useful contribution is the *evidence ordering* from their ablation — fusion matters more than any individual evidence source — plus the architecture for auditable, cited, self-monitoring output. Our contribution builds on that by making the confidence claim real.

**Q. Isn't this just RAG with extra steps?**
Structurally, yes — and the paper's own framing is that it's an integration. What RAG-with-extra-steps buys you here: two independent evidence tiers that must agree, a hard traceability requirement (a diagnosis is kept only if a retrieved passage supports it), and a confidence gate that triggers re-investigation. That combination is what the ablation says matters.

### 19.2 Architecture

**Q. Why five agents? Why not one strong model with a long prompt?**
The ablation says multi-agent helps, but it doesn't isolate *why*. Plausible reasons: context isolation (the fusion agent sees only evidence, not the raw note, which reduces anchoring), role-specific prompting without prompt bloat, and the ability to swap models per task. I think context isolation is the underrated one and it's testable — worth an ablation the paper didn't run.

**Q. Is a different LLM per agent actually necessary?**
The paper claims it is ("rational division of labour... deepened expertise") but provides **no evidence**. Every baseline in Table 4 is a single model. I consider this an untested claim and I test it directly in my model-sensitivity ablation.

**Q. How much does CrewAI actually contribute, versus just being an orchestration library?**
It supplies Agent/Crew/Process/Task primitives and sequential flow plus manual re-entry for the feedback loop. It is engineering convenience, not scientific contribution. Nothing in the architecture depends on it.

**Q. Why is the Adaptive Optimizer on the same model as the Symptom RAG Analyzer?**
Coincidence of capability match — both need fast, high-throughput function calling. The paper's own reason is latency and call throughput. It's a reasonable operational choice, not a principle.

### 19.3 RAG and retrieval

**Q. Why BioBERT? It's from 2019.**
Because it was pre-trained on PubMed and handles biomedical jargon and NLI, unlike general-purpose embedders. That was a real gap in 2020-era RAG CDSS work. But in 2026 it's clearly not state of the art, and it's a 512-token, non-retrieval-tuned BERT-base encoder. I treat it as the weakest component and I quantify that.

**Q. What is "medical term boosting" and does it work?**
It repeats high-priority clinical phrases in the query so they dominate the embedding — "shortness of breath", "chest pain". Their ablation says disabling it drops 94.7% → 80.0%, which is a huge effect for such a simple trick. I want to reproduce that number before believing it, because repeating tokens can also distort the embedding. I suspect part of that effect is a retrieval-recall artefact that would show up immediately in a recall@k measurement — which is exactly why the paper should have reported retrieval metrics and didn't.

**Q. How does the three-level fallback differ from ordinary top-k retry?**
The third level is genuinely different: instead of loosening thresholds or rephrasing, it reconstructs the query from the *stored* clinical semantic content — converting the symptom pattern into a new semantic cluster and building an alternative query. That's a learned re-embedding strategy, not a parameter sweep. Levels 1 and 2 are fairly conventional.

**Q. Did they evaluate retrieval quality at all?**
No. No recall@k, no MRR, no nDCG. That's the single biggest evaluation gap in the paper — contribution #2 is asserted and never measured. It's the first thing I added to my evaluation harness.

**Q. 200,000 chunks from 1,000 PDFs — but the paper says 18M embeddings. Which is it?**
It can't be both. 200,000 chunks produce 200,000 embeddings at one vector each. And 200,000 × 768 × 4 bytes is 0.61 GB, not the stated 2.1 GB. The 18M figure would imply 18,000 chunks per PDF, which contradicts the 200 chunks per PDF implied by the other pair. I could not reconcile them from the paper and had to re-measure. Their ingestion benchmark table also isn't self-consistent with an 8 GB GPU doing 18M sentence embeddings in 38 minutes.

**Q. Is the vector database a catalogue of diagnoses?**
The paper explicitly says no — it's textbook, guideline and epidemiology passages, and a candidate diagnosis is kept only if a retrieved passage explicitly supports it. That design point is good: it's what makes outputs traceable to spans.

### 19.4 Fusion and confidence

**Q. This is the key question: what exactly is the confidence score? How is it computed?**
The paper says it depends on the RAG similarity score, the web evidence score, and the degree of concordance between the two streams. It never gives a formula, weights, or normalisation. I cannot reproduce it from the paper. I'd have to invent something, and anything I invent would not be *their* number. So my confidence score is explicitly my own design, reported as such.

**Q. Is the confidence actually calibrated?**
No evidence. They call it "calibrated" and use τ = 0.65 to gate the feedback loop, but there is no ECE, no Brier score, no reliability diagram, and no stated derivation of 0.65. Calling a score "calibrated" without a calibration metric is a definitional, not empirical, claim. **This is the biggest gap and it's what my project fixes first.**

**Q. Why 0.65?**
Not stated. Presumably tuned on their 20-question subset — which would make it overfit to the test set.

**Q. The ablation says removing fusion is worst, at 60%. Why?**
My reading: the fusion agent isn't just an aggregator, it's the only place where *agreement between independent evidence tiers* is checked. Remove it and you have two agents emitting confident-looking but unvalidated diagnoses side by side, and nothing catches a hallucinated one. It is also the only component that produces a cited, structured, explainable output — so removing it degrades the output's *form*, not just its content. Either way, the result says the system's reliability lives in the reconciliation step, not in the retrieval.

**Q. Precision, recall and F1 all equal accuracy in your table. Is that meaningful?**
No, and the paper admits it — those metrics coincide with accuracy *by construction* under that binary reduction. It's four copies of one number. The informative number is MedQA specificity at 0.985. What you'd actually want for a CDSS is calibration and selective-risk: if the system says 90% confident, how often is it right? I report an accuracy-vs-coverage curve instead.

**Q. Precision at 0.88 on PubMedQA — how does that square with an 88% accuracy?**
Under the binary reduction they used, precision and recall are forced to equal accuracy. And that whole reduction is questionable for PubMedQA, which has yes/no/maybe labels. Grading free text at a self-chosen cosine threshold of 0.80 is a knob, not a ground truth. I'd score exact match against the gold label.

### 19.5 Feedback loop

**Q. The feedback loop improves accuracy from 86.3% to 94.7%. That's a big gain.**
On twenty questions. One question is five percentage points. 86.3% of 20 is 17.26 correct, which isn't an integer — so either more than 20 questions were run, or the numbers are averaged over partial runs, and neither is stated. At n=20 the gain is not statistically significant; the p-value is about 0.37.

**Q. The paper claims the loop raises consistency 1.8×. What is consistency?**
It's never defined. I could not find a consistency metric, a computation, or a table anywhere in the paper. I treat it as unsupported.

**Q. The paper claims a 14.2% quality improvement. Where does 14.2% come from?**
I can't derive it. 86.3 → 94.7 is +8.4 points or +9.7% relative. Neither is 14.2%.

**Q. Does the loop keep improving?**
No — it saturates at iteration 3, which the authors acknowledge. They even explain why: the loop starts repeating the same suggestions. So the ceiling is a known limitation they identified and didn't fix. Their iteration-4 and 5 results, 95.1% and 95.5%, are within noise and cost 48% more time.

**Q. Processing times "increase logarithmically" — correct?**
The table shows 2.5, 3.0, 3.7, 4.4, 5.5 minutes — uniformly about 1.2× per iteration. That's linear with a constant factor. I think the authors conflated the per-iteration complexity expressions, O(L) and O(log N) + O(k), with growth *across* iterations, which is just linear in the iteration count.

**Q. Isn't it dangerous for a system to change its mind three times in a diagnostic loop?**
In the paper's framing no, because every iteration is evidence-gated and logged. But the paper doesn't study the failure mode where iteration makes it *worse*. It only reports monotonically increasing accuracy. A system that can revise itself needs to be measured on revision quality, not just final accuracy — for example, how often does a later iteration overturn a correct earlier answer? That needs tracking, which the paper doesn't do.

### 19.6 Datasets and evaluation

**Q. Fifty questions per dataset — isn't that too small?**
Yes, and it's the paper's central weakness. With n=50 the 95% confidence interval on MedQA is [83.8%, 97.9%], which contains eight of the eleven baselines in their own Table 4. Every headline claim becomes statistically indistinguishable from the best published number: p = 0.70 on MedQA against DeepSeek-R1, p = 0.28 on PubMedQA, p = 0.60 on MedBullets.

**Q. The conclusion claims "statistically significant performance improvement." Where's the test?**
There isn't one. No p-value, no confidence interval, no test statistic, no standard deviation, no seed, and no repeated run appears anywhere in the paper. When I ran the power analysis, a two-point gap over DeepSeek-R1 on MedQA needs on the order of a few thousand paired items to detect reliably. The claimed significance isn't established.

**Q. How is the comparison in Table 4 fair, if the baseline numbers come from other papers?**
It isn't fully fair, and the authors say as much. Baselines used different sample counts, prompts, dates and hardware. Most of them trace back to a single paper, MDAgents, so it's really one experimental campaign rather than eleven independent results. Their strongest rival on MedQA, DeepSeek-R1 at 92%, is a plain model with no agents at all — so the multi-agent architecture buys two points over just calling a good model, which is well inside noise.

**Q. 94% or 94.7%?**
94.0% is the main result: 47/50 on Table 5 and 94.0 on Table 4. The 94.7% comes from the 20-question iteration study in Table 6, and it leaks into §II-C where it's used as the reference for the "+29.7 and +24.7 point" improvement claims. The 50-question number is the one to quote.

**Q. How does the MedQA metric work?**
Per-question one-vs-all: the correct option is the positive class and the four others are negatives, so a correct answer contributes one true positive and four true negatives, and an incorrect one contributes one false positive, one false negative and three true negatives. Summing gives precision, recall and F1 all equal to accuracy, with specificity adding information — 0.985, which I verified arithmetically. Worth noting the per-option accuracy over all 250 option slots is 244/250 = 0.976, higher than the per-question 0.94.

**Q. How are free-text answers scored?**
Cosine similarity between sentence embeddings of the prediction and the reference, thresholded at 0.80, fixed a priori. Lower values inflate false positives, higher values reduce recall, they say. My concern is that semantic similarity isn't correctness — a verbose answer that agrees with the gold on the surface but is clinically wrong on a consequential detail still scores above threshold.

**Q. Is MedBullets publicly available?**
MedBullets has historically been a licensed corpus with API access under terms of use, and the paper doesn't state how it obtained the labelled 50. That's a reproduction risk I'd verify before promising that comparison.

**Q. What's the right way to evaluate a CDSS?**
Beyond accuracy: calibration (ECE, Brier, reliability diagrams), selective prediction (accuracy versus coverage), retrieval quality (recall@k, nDCG), clinical safety (harmful omission rate), latency percentiles, and cost per case. The paper reports only accuracy, a degenerate set of classification metrics, and mean latency.

### 19.7 Ablation

**Q. The ablation is on 20 questions. Can we trust it?**
It's the strongest evidence in the paper, and it's the only claim with statistical support behind it. Fusion removal is p ≈ 0.009 and RAG Analyzer removal p ≈ 0.019 at n=20. But twenty questions is still twenty questions, there's one ambiguous row, and no repeated runs.

**Q. There's a labelling conflict in Table 7.**
Yes. The text says disabling "Medical Term Strengthening" drops accuracy to 80.0%. Table 7 has 80.0% for removing the Clinical Text Clarifier *and* 80.0% for disabling Medical Term Boosting. Same number, three candidate components, so the mapping is ambiguous.

**Q. Removing the RAG Analyzer gives 65%. The paper calls that "RAG-only."**
That label is wrong. Removing the RAG Analyzer leaves a system with no RAG. It's an ablation row being relabelled as a configuration, and it's then used to claim "+29.7 points over RAG-only" — a delta computed against a different accuracy figure (94.7%) than the headline (94.0%).

**Q. Which component matters most?**
Fusion, at 60% retained. Then the RAG Analyzer at 65%, then the web scanner and the adaptive optimizer tied at 70%, then the clarifier and term boosting tied at 80%. The interesting finding is that removing *reconciliation* costs more than removing any individual evidence source.

### 19.8 Cost, deployment, and limits

**Q. Is 2–4 minutes per case acceptable clinically?**
For non-emergency, outpatient, pre-visit decision support, yes — the authors make exactly that argument, and 3.7 minutes is the stated mean. For emergency medicine, no. And the bottleneck is entirely the web crawler at roughly 45 seconds per page, so it's an engineering problem rather than a fundamental one. Cache and snapshot the evidence and it drops to seconds.

**Q. What does a case cost in API fees?**
The paper doesn't say — it reports only wall-clock, never token counts or cost, for a system that calls four different frontier models. That's a real gap for any deployable system.

**Q. The paper's GPU has 8 GB of VRAM. How do you run a 72B model?**
It doesn't run locally — it's an API call. Which means the whole system's latency is bounded by hosted API response time, and that's a cost/reproducibility dependency the paper doesn't discuss.

**Q. What if the web evidence is wrong or low quality?**
The paper's answer is a hierarchy weighting — guidelines and systematic reviews above meta-analyses above case reports — but it's qualitative and unvalidated. My version assigns explicit evidence levels and measures whether down-weighting case reports actually helps.

**Q. Does this replace the physician?**
No, and the paper says so. It's decision support; the clinician retains the course of action. My version adds an abstention path so the system can say the evidence is insufficient rather than guessing.

**Q. What about hallucination?**
Reduced, not eliminated — the paper concedes this. I don't think there's a hallucination *metric* in the paper at all, which is a measurement gap. My version adds a red-team set and reports harmful-omission and unsafe-recommination rates.

**Q. How would you deploy this in a hospital? Biggest concerns?**
Three: (1) governance — every output needs a cited trace and a versioned model/prompt manifest, otherwise it's unauditable; (2) validation — benchmark accuracy is not clinical validation, and multicenter prospective study is the only acceptable evidence; (3) regulatory — this is a medical device under FDA and EU MDR, and no benchmark result substitutes for that pathway.

**Q. Why should I believe your results over theirs?**
Because I'm reporting the things that make a result checkable: sample sizes large enough to have narrow intervals, paired significance tests against baselines I ran myself on the same machine with the same prompts, retrieval and calibration metrics alongside accuracy, and a content-hash manifest of the corpus. Their central number has a confidence interval fourteen points wide.

### 19.9 Suggested opening and closing

**Opening (30 seconds):** *"This paper proposes a five-agent clinical decision support system where each agent runs a different LLM — a text normaliser, a BioBERT-Qdrant retrieval agent, a live web evidence scanner, a fusion/confidence agent, and an adaptive optimizer that re-queries when confidence is low. It reports 94%, 88% and 84% on MedQA, PubMedQA and MedBullets. My work both reproduces it and tests it rigorously. The architecture holds up — the ablation is sound and it's their best result. But the headline accuracy numbers don't survive a power analysis, and I can show you exactly which ones don't and why."*

**Closing:** *"The paper's real contribution is the evidence ordering from its ablation: reconciling evidence matters more than any single evidence source. Where it overreaches is statistics — n equals twenty to fifty with no confidence intervals and a 'statistically significant' claim with no test behind it. My contribution is making that architecture measurable: full-split evaluation with paired tests, retrieval metrics for a RAG contribution that was never measured, real confidence calibration for a score they call calibrated, and a latency breakdown that attacks the 90% web-crawl bottleneck."*

---

## 20. Appendix — reference map

| Ref | Short name | Role in the paper |
|---|---|---|
| [1] | Musen, Middleton & Greenes, *Clinical Decision Support Systems* | Textbook framing |
| [3] | Ohta et al., *J Patient Safety* 2019 | 70% of errors human-caused |
| [4] | Greenes, *Clinical Decision Support: The Road to Broad Adoption* | Limits of rule-based systems |
| [5] | Achiam et al., GPT-4 technical report | LLM background |
| [6] | Öğdü et al., cultural-heritage damage detection | Unrelated; self-citation |
| [7] | Matarazzo & Torlone, LLM survey | LLM background |
| [8] | Arslanoğlu & Karaköse, trustworthy chatbot analysis | Author-internal |
| [9] | Nayak et al., *JAMA Intern Med* 2023 | Chatbot vs residents, HPI summaries |
| [10] | Ayers et al., *JAMA Intern Med* 2023 | Chatbot vs physician responses |
| [11] | Karaköse et al., medical deepfake detection | Adjacent imaging work |
| [12] | Sherimon & Krishnan, *OntoDiabetic* | Ontology-based CDSS baseline |
| [13] | Ji et al., LLM internal states / hallucination | Hallucination background |
| [14] | Levra et al., *Eur J Intern Med* 2025 | Single-model LLM CDSS, syncope in ED |
| [16] | Öğdü et al., LLM CDSS in healthcare | Radiology trials: GPT-4 98%, Llama-3.3-70B 96%, GatorTron −12% |
| [17] | Zuo et al., **KG4Diagnosis** | Hierarchical multi-agent + KG |
| [18] | Kim et al., **MDAgents**, NeurIPS 2024 | Adaptive multi-agent collaboration |
| [19] | Mehandru et al., LLMs as agents in the clinic | Agent-based CDSS |
| [20] | Wilk et al., *J Med Syst* 2016 | MAS for interdisciplinary teams |
| [21] | Ong et al., RAG CDSS for medication safety | RAG CDSS, 12 specialties |
| [22] | Chen et al., RAG benchmarking, AAAI 2024 | RAG background |
| [23] | Lee et al., **BioBERT**, *Bioinformatics* 2020 | Embedding model |
| [24] | McMurry et al., symptom identification multicenter | LLM symptom extraction |
| [25] | Levine et al., *Lancet Digit Health* 2024 | GPT-3 diagnostic/triage accuracy |
| [26] | Guo et al., **DeepSeek-R1** | Agent 1 model |
| [27] | Balestri, *Front Artif Intell* 2025 | Bias study of Gemini 2.0 Flash — cited for capability |
| [28] | Tian et al., test-time scaling | Cited for Qwen2-72B — mismatch |
| [29] | Siddiky et al., GPT-4 vs GPT-4o | Cited for GPT-4o-mini — mismatch |
| [30] | dux-tecblic/symptom-disease-dataset | RAG corpus |
| [31] | Chen, Fang, Singla & Dredze, NAACL 2025 | Benchmarking LLMs on medical QA |
| [32] | Feng et al., MedRxiv 2024 | LLM on biomedical NLU/reasoning |
| [33] | Huang et al., M1 test-time scaling | Baseline source |
| [34] | Tang et al., **MedAgentsBench** | Baseline source |

---

## 21. Quick reference card

```
PAPER      IEEE Access 13:167390-167404 (2025) · DOI 10.1109/ACCESS.2025.3613340
AUTHORS    Öğdü, Arslanoğlu*, Karaköse — Fırat University, Elâzığ, Türkiye
FRAMEWORK  CrewAI (Agent / Crew / Process / Task)
AGENTS     1 Text Clarifier      DeepSeek-R1      normalize→NER→relation→SNOMED/ICD→JSON
           2 Symptom RAG Analyz. Gemini-2.0-Flash BioBERT→Qdrant cosine top-k, ranked dx
           3 Evidence-Based Scan Qwen2-72B        Serper→PubMed/Scholar/NICE, evidence tiers
           4 Clinical Data Fusion GPT-4o-mini      cross-validate, conflict detect, confidence
           5 Adaptive Optimizer   Gemini-2.0-Flash τ=0.65 gate, feedback report, max 3 iters
RAG        BioBERT-mnli-snli-scinli-scitail-mednli-stsb, 768-d, Qdrant gRPC
           chunk 512/64 (fallback 500/80), cosine, threshold 0.65, 3-level fallback
CORPUS     ~1000 PubMed PDFs + 5,630 patient texts (dux-tecblic/symptom-disease-dataset)
           term boosting + metadata (source, doc_id, page, len, retrieval time)
DATA       MedQA / PubMedQA / MedBullets — n=50 each
SCORING    MedQA exact-match; free-text cosine ≥ θ=0.80; one-vs-all micro reduction
RESULTS    94.0 / 88.0 / 84.0    (iteration study: 86.3→91.5→94.7 on n=20)
ABLATION   Fusion 60 · RAG 65 · Web 70 · Optimizer 70 · Clarifier 80 · TermBoost 80
LATENCY    1 iter 2-3 min · 3 iter 3.7 min · web 45 s/page = ~85-90% of total

STRONGEST  Fusion is the most critical component (p≈0.009). Architecture idea is sound.
WEAKEST    n=50 with no CI; "statistically significant" with no test; θ=0.80 grading knob;
           undefined "calibrated" confidence; 18M-vs-200k embeddings contradiction;
           "logarithmic" timing is linear; 14.2% not derivable; undefined "1.8× consistency";
           n=20 cannot express 86.3%; 94% vs 94.7% ambiguity; no retrieval metrics at all.
OUR EDGE   Full public splits + Wilson/McNemar · calibration (ECE/Brier) · selective risk ·
           recall@k, MRR, nDCG + embedder bake-off + hybrid + reranker · diversity-aware
           feedback loop past their plateau · deterministic offline snapshot + live mode ·
           latency & cost optimisation (attack the web crawl) · ablation at full n, 5 seeds ·
           safety/abstention red-team set
```

---

*Prepared for presentation and implementation. Companion document: `docs/IMPLEMENTATION_PLAN.md`.*