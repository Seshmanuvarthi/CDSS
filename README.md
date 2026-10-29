# CDSS: Adaptive Multi-Agent Clinical Decision Support System

An adaptive multi-agent clinical decision support system integrating biomedical dense vector retrieval (BioBERT + Qdrant), dynamic literature validation (web intelligence), and confidence-gated adaptive optimization.

Based on the research paper:
> **Öğdü, E. et al. (IEEE Access, 2025)**  
> *"An Adaptive Multi-Agent LLM-Based Clinical Decision Support System Integrating Biomedical RAG and Web Intelligence"*

---

## 🏛️ Architecture Overview

The system operates as a closed-loop multi-agent reasoning framework consisting of **5 specialized autonomous agents**:

1. **Agent 1: Clinical Text Clarifier** (`cdss/agents/clarifier.py`) — Normalizes noisy clinical narratives into structured JSON (`ClinicalNote`), handling symptom negations, vitals, lab biomarkers, and diagnostic imaging.
2. **Agent 2: Symptom RAG Analyzer** (`cdss/agents/rag_analyzer.py`) — Queries a local Qdrant vector database using 768-dimensional BioBERT embeddings to retrieve biomedical literature passages and propose ranked candidate diagnoses ($H_1, H_2, H_3$).
3. **Agent 3: Evidence-Based Scanner** (`cdss/agents/evidence_scanner.py`) — Scans live clinical practice guidelines (NICE, ACC/AHA, ESC) and recent PubMed literature via Google Serper / local Level-A guideline corpus.
4. **Agent 4: Clinical Data Fusion** (`cdss/agents/fusion.py`) — Performs multi-source synthesis, evaluates semantic concordance, checks for biological contraindications, and computes calibrated confidence via Paper Eq. 3.
5. **Agent 5: Adaptive Optimizer** (`cdss/agents/optimizer.py`) — Governs the feedback loop. If calibrated confidence $< 0.65$, formulates targeted clinical keywords and redirects upstream re-querying (up to 3 iterations) before halting or safely escalating to attending physician review.

---

## 🚀 Quickstart Guide

### 1. Backend Setup (FastAPI + Agent Pipeline)

```zsh
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure Environment Keys in .env:
# OPENROUTER_API_KEY="your_openrouter_key"
# SERPER_API_KEY="your_serper_key" (Optional, for live web guidelines)

# Start Backend Server
uvicorn cdss.api.server:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend Setup (Angular Dashboard)

```zsh
cd frontend
npm install
npm start
```

Open `http://localhost:4200` in your browser.

---

## 🧪 Benchmark Datasets & Verification

The system includes a **Live Benchmark Verification Runner** supporting:
* 🩺 **MedQA (USMLE Step 1 & 2)**: 1,273+ clinical examination vignettes.
* 🔬 **PubMedQA**: 1,000 labeled biomedical research study abstracts (`Yes`, `No`, `Maybe`).
* 💊 **MedBullets**: 50 clinical residency scenarios covering pharmacotherapy and interventions.

Each test run dynamically samples a random 10 cases without replacement and streams real-time diagnostic execution traces and verdicts into the live UI log console.

---

## 📚 Documentation

For exhaustive technical specifications, mathematical confidence proofs, corpus ingestion details, and file references, see:
* [`docs/CDSS_ARCHITECTURE_AND_IMPLEMENTATION_GUIDE.md`](docs/CDSS_ARCHITECTURE_AND_IMPLEMENTATION_GUIDE.md)
* [`docs/PAPER_DEEP_DIVE.md`](docs/PAPER_DEEP_DIVE.md)
