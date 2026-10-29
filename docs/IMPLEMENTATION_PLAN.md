# Implementation Plan — Reproducing and Extending *An Adaptive Multi-Agent LLM-Based CDSS Integrating Biomedical RAG and Web Intelligence*

> Build plan for a from-scratch Python + TypeScript implementation that (a) faithfully reproduces Öğdü et al. (IEEE Access 2025), (b) re-evaluates its claims with statistically valid methodology, and (c) extends it to beat every one of its reported metrics.
> Companion document: `docs/PAPER_DEEP_DIVE.md` (paper teardown, audit, Q&A bank).

---

## 0. Executive summary

**Two configs, one codebase.**

| | `faithful` | `improved` |
|---|---|---|
| Purpose | Reproduce the paper so the head-to-head is valid | Beat the paper |
| Retrieval | BioBERT dense-only, top-*k*, cosine | Hybrid BM25+dense, tuned embedder, cross-encoder reranker |
| Confidence | Paper's unspecified formula, τ = 0.65 | Explicit formula + temperature scaling, ECE/Brier reported |
| Web | Live Serper crawl | Deterministic offline snapshot (live mode retained for demo) |
| Feedback | Plain loop, max 3 iterations | Diversity-penalized, novelty-scored, escalation-on-no-progress |

**Ten phases**, each with a hard acceptance test that must pass before the next begins.

**The binding constraint is free-tier compute.** 500 q × 3 datasets × ~6 calls × 3 iterations ≈ 27k calls, versus ~200 req/day on OpenRouter free. The plan therefore splits engines: **local Ollama for bulk statistics, free cloud tiers for the headline runs.** See §7 and §9.

**Evidence ladder** — ship progressively, never ship nothing:

| Tier | Scope | Target |
|---|---|---|
| **T1** | Matched 50/50/50, `faithful`, CIs + McNemar, one ablation sweep, retrieval metrics, latency/cost table, working UI | Presentation-ready |
| **T2** | T1 + 300/dataset `improved` + full ablation at n=300/3 seeds + calibration + embedder bake-off | **Recommended target** |
| **T3** | T2 + full public splits + 5 seeds + safety red-team + selective-risk curves + live demo | If writing a paper |

---

## 1. Repo layout

```
Major/
├── README.md
├── pyproject.toml                  # uv-managed, src layout, ruff + mypy + pytest
├── uv.lock
├── Makefile                        # make ingest / index / eval / ui / all
├── .env.example
├── docs/
│   ├── PAPER_DEEP_DIVE.md          # ✅ written
│   ├── IMPLEMENTATION_PLAN.md      # this file
│   ├── DECISIONS.md                # append-only decision log (ADR-lite)
│   └── RESULTS.md                  # generated results tables
├── configs/
│   ├── faithful.yaml               # paper-faithful
│   ├── improved.yaml
│   ├── models.local.yaml           # Ollama
│   ├── models.groq.yaml
│   ├── models.gemini.yaml
│   └── models.openrouter.yaml
├── backend/                        # Python package
│   └── src/cdss/
│       ├── config.py               # pydantic-settings
│       ├── logging_utils.py        # structured JSONL + run IDs
│       ├── llm/
│       │   ├── base.py             # LLMClient protocol, Message, LLMResponse
│       │   ├── registry.py         # model catalog + role→model binding
│       │   ├── providers/{gemini,groq,openrouter,ollama,openai}.py
│       │   ├── ratelimit.py        # token bucket, org-wide, retry/backoff
│       │   ├── cache.py            # disk-backed prompt cache
│       │   └── ledger.py           # token/cost/latency accounting
│       ├── corpus/
│       │   ├── pubmed.py           # Entrez E-utilities
│       │   ├── symptom_disease.py  # HF dataset
│       │   ├── manifest.py         # SHA-256 corpus manifest
│       │   └── queries.py          # 15+ keyword query sets, 20-topic exclusion
│       ├── ingest/
│       │   ├── extract.py          # PyMuPDF, OCR path, ICU normalization
│       │   ├── clean.py            # regex boilerplate stripper
│       │   ├── chunk.py            # 512/64 → 500/80 fallback
│       │   └── bench.py            # measured pages/sec, chunk counts
│       ├── retrieval/
│       │   ├── embedder.py         # BioBERT + modern alternatives
│       │   ├── index.py            # Qdrant gRPC + HNSW
│       │   ├── term_boosting.py    # medical term boosting
│       │   ├── fallback.py         # 3-level fallback strategy
│       │   └── search.py           # custom search tool
│       ├── agents/
│       │   ├── clarifier.py        # agent 1
│       │   ├── rag_analyzer.py     # agent 2
│       │   ├── evidence_scanner.py # agent 3
│       │   ├── fusion.py           # agent 4
│       │   ├── optimizer.py        # agent 5
│       │   ├── schemas.py          # pydantic JSON schemas
│       │   └── prompts/            # versioned .md prompt templates
│       ├── orchestration/
│       │   ├── crew.py             # CrewAI assembly
│       │   ├── pipeline.py         # sequential + feedback loop
│       │   ├── confidence.py       # ★ the formula the paper omits
│       │   ├── calibration.py      # temperature scaling, ECE, Brier
│       │   └── conflicts.py        # conflict detection protocol
│       ├── evidence/
│       │   ├── serper.py           # live web search
│       │   ├── snapshot.py         # deterministic offline snapshot
│       │   └── grading.py          # Oxford/CEBM evidence levels
│       ├── eval/
│       │   ├── datasets/{medqa,pubmedqa,medbullets}.py
│       │   ├── scorers.py          # exact-match, one-vs-all, cosine
│       │   ├── metrics.py          # Wilson, Clopper-Pearson, McNemar, bootstrap
│       │   ├── calibration_metrics.py  # ECE, Brier, reliability diagram
│       │   ├── retrieval_metrics.py    # recall@k, MRR@10, nDCG@10
│       │   ├── runner.py           # main eval driver
│       │   ├── ablations.py        # component removal at full n
│       │   ├── iterations.py       # iteration sweep
│       │   └── safety.py           # red-team, abstention
│       └── api/
│           ├── server.py           # FastAPI + SSE streaming
│           └── schemas.py
├── frontend/                       # React + Vite + TypeScript
│   └── src/
│       ├── pages/{CaseInput,CaseTrace,Report,Results}.tsx
│       ├── components/{AgentTrace,EvidencePanel,ConfidenceBadge}.tsx
│       └── api/client.ts
├── data/                           # gitignored; manifest.json is committed
├── artifacts/                      # vectors, snapshots, eval outputs
└── tests/
    ├── unit/ integration/ eval/
```

---

## 2. Phase 0 — Scaffold (0.5 day)

**Deliverable:** `uv sync` succeeds; `make test` green; config loads.

**Files:** `pyproject.toml`, `Makefile`, `.env.example`, `src/cdss/config.py`, `src/cdss/logging_utils.py`, `configs/*.yaml`.

**Key design:**
- `uv` for dependency management; `src/` layout; `ruff` + `mypy --strict` + `pytest`.
- Pydantic v2 `BaseSettings` for config; YAML profiles loaded over env.
- Structured JSONL logging with a `run_id` threaded through every agent call. This is what makes "auditable logs" a real feature rather than a claim.
- `.env.example` lists every key with an empty value and a comment on which are optional.

**Acceptance test:** `make bootstrap && make test` exits 0; `python -c "from cdss.config import load; load('configs/faithful.yaml')"` prints the merged config.

**Risk:** none. Do this first and get it right.

---

## 3. Phase 1 — LLM abstraction & free-tier strategy (1–2 days)

This is the highest-leverage phase and the one that determines whether the project is feasible on a zero budget.

### 3.1 The `LLMClient` interface

```python
class LLMClient(Protocol):
    async def complete(self, req: LLMRequest) -> LLMResponse: ...

@dataclass(frozen=True)
class LLMRequest:
    messages: list[Message]
    model: str
    temperature: float = 0.0        # deterministic default
    max_tokens: int = 2048
    response_schema: type[BaseModel] | None = None   # structured output
    seed: int | None = None
    timeout_s: float = 60.0

@dataclass(frozen=True)
class LLMResponse:
    text: str
    parsed: BaseModel | None
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str
    cached: bool
    provider: str
```

Every provider adapter implements this and nothing else. Everything above it — agents, orchestrator, eval — is provider-agnostic.

### 3.2 Model discovery — mandatory, not optional

Model IDs and free-tier availability churn monthly. Hardcoding them guarantees breakage. So:

```bash
make discover-models            # hits /v1/models on every configured provider
```

Writes `artifacts/models/<provider>_<date>.json` and prints a table of candidates per **role** (clarifier / rag_analyzer / evidence_scanner / fusion / optimizer) with context window, tool-calling support, and structured-output support. `configs/models.*.yaml` reference candidates; the registry resolves against the discovery snapshot at runtime and **fails loudly if a configured model is unavailable**.

### 3.3 Free-tier provider matrix

| Provider | Base URL | Free limits (verify at runtime) | Best role fit |
|---|---|---|---|
| **Google AI Studio** | Gemini SDK | project-scoped RPM/RPD, generous context | rag_analyzer, optimizer (paper's actual models) |
| **Groq** | `https://api.groq.com/openai/v1` | ~30 RPM org-wide; ~1k–14.4k RPD per model; **multiple keys do NOT raise org limits** | rag_analyzer, clarifier (very fast) |
| **OpenRouter** | `https://openrouter.ai/api/v1` | ~200 RPD (`:free` suffix) | overflow only |
| **Ollama (local)** | `http://localhost:11434` | **unlimited** | bulk eval, ablations, sweeps, dev |
| OpenAI (optional) | — | paid | confirmation run on GPT-4o-mini |

**Critical operational rules:**
1. **Rate limits are organization-wide, not per-key.** Rotating keys does not help. Design for the limit, don't try to evade it.
2. **Token-bucket limiter per provider**, plus `tenacity`-style exponential backoff with jitter on `429`/`5xx`.
3. **Disk-backed prompt cache** keyed on `sha256(model + messages + params + schema_version)`. Cached responses are free. Re-running the eval after a code change costs almost nothing if the LLM path is unchanged — this is what makes iteration affordable.
4. **Token/cost/latency ledger** appended to every run: input tokens, output tokens, wall-clock, provider, model. Answers "what does a case cost?" — which the paper never does.
5. **Concurrency ceiling per provider**, configurable. Bulk eval defaults to Ollama; cloud runs default to low concurrency with backoff.

### 3.4 Two-engine strategy (see §9.2 for the trade-off)

| Purpose | Engine | Why |
|---|---|---|
| Development, unit tests, CI | Ollama or a **recorded fixture** | Zero cost, deterministic, no network |
| Bulk statistics: CIs, ablations, sweeps, retrieval | Ollama local | Unlimited; comparisons internally valid because the engine is fixed |
| Headline matched-50/50/50 | Free cloud tiers | Needs the paper's model class |
| Live demo | Free cloud + live Serper | Shows the architecture working |

**Hard rule: never report a number without its model manifest.** Mixing engines across a comparison invalidates it — that is precisely the mistake the paper makes with its borrowed Table 4.

**Acceptance test:** a test that runs the same prompt through two providers and two mock providers; cache hit verified (second call returns `cached=True` and zero cost); rate limiter correctly delays under simulated 429; ledger totals reconcile with per-call records.

---

## 4. Phase 2 — Corpus acquisition (1–2 days)

### 4.1 PubMed via Entrez E-utilities

- `esearch` → `efetch`, with `tool=` and `email=` params (NCBI requires these; the paper mentions only the history option).
- **`HISTORY` server parameter + 8-iteration backoff**, as the paper does, to avoid overloading NCBI.
- Relevance ranking; **1,000 PDFs** by default in `faithful`, **5,000–10,000** in `improved`.
- **15+ keyword queries** on symptom–diagnosis correlation, clinical findings patterns, multi-organ injury.
- Filters biased to **human clinical studies and diagnostic papers** (publication type filters).
- **~20 excluded topics**, seeded and configurable: AI applications, cancer biomarkers, pandemic vaccine effects, plus ~17 more chosen by us and documented.

### 4.2 Symptom-disease dataset

`dux-tecblic/symptom-disease-dataset` from HuggingFace. Enrich each record into varied phrasings so the same diagnosis occupies multiple regions of vector space. Store as its own collection (or payload `kind`) so retrieval can filter by source.

### 4.3 Manifest — the reproducibility fix

`data/manifest.json` (committed, PDFs gitignored):

```json
{
  "generated_utc": "...",
  "sources": [{"name":"pubmed","n":1000,"query_set_hash":"..."},
              {"name":"symptom_disease","n":5630,"hf_revision":"..."}],
  "files": [{"path":"...","sha256":"...","bytes":...,"pages":...}],
  "excluded_topics": ["..."],
  "total_documents": 6630
}
```

This is the single most valuable artifact for defensibility, and it is exactly what the paper lacks.

**Acceptance test:** manifest generated; a second run over the same inputs produces byte-identical hashes; total document count is reconciled and *printed* (this is where the paper's 889-vs-1,000 inconsistency gets caught).

---

## 5. Phase 3 — Ingestion pipeline (1–2 days)

| Stage | Implementation | Paper's claim to reproduce |
|---|---|---|
| Extract | PyMuPDF, page-by-page | **65 pages/sec**, ~9.6× pdfplumber |
| OCR path | `ocrmypdf`/Tesseract; ICU Unicode normalization (`unicodedata.normalize('NFKC')`) | **28% of pages need OCR** |
| Clean | Regex/keyword filters strip references, figure/table captions, copyright, author info, abstract, keywords, emails, publication boilerplate | as described |
| Chunk | `RecursiveCharacterTextSplitter` `chunk_size=512`, `chunk_overlap=64`; if segment count inadequate → **page-based fallback 500/80** | as described |
| Filter | Drop too-short, low-content, label-only segments | as described |
| Tag | Metadata: source filename, page number, text length | as described |
| Embed | BioBERT-mnli-snli-scinli-scitail-mednli-stsb → **768-d**, batched | as described |
| Load | Qdrant, **gRPC**, HNSW, payload = text + metadata | **−32% latency vs REST** |

### 5.1 The measurement harness — where we beat them

`make ingest-bench` prints, for our own corpus:

- measured pages/sec (vs pdfplumber, head to head on the same 50 PDFs)
- measured OCR page fraction (paper: 28%)
- measured chunk count, mean/median chunk length, chunk count per PDF
- **measured index bytes on disk**, broken down: vectors / HNSW graph / payload / text
- total wall-clock, and pages/sec/GPU

This directly substitutes measured numbers for the paper's unreconcilable "18M embeddings from 200,000 chunks / 2.1 GB / 38 minutes". **This table is going in the paper we write.**

### 5.2 Qdrant collection schema

```python
PointStruct(
  id=uuid5(namespace, f"{doc_id}:{page}:{chunk_idx}"),
  vector=embedding_768d,
  payload={
    "text": str,
    "source": "medical-papers" | "symptom-disease-dataset",
    "kind": "book-chunk" | "diagnosis-symptom-pair",
    "doc_id": str, "page": int, "chunk_idx": int, "length": int,
    "evidence_level": "guideline"|"systematic-review"|"meta-analysis"|"case-report"|"n/a",
  },
)
```

**Acceptance test:** re-running `make ingest-bench` on 50 PDFs reproduces the paper's ordering of magnitude for pages/sec; our own arithmetic self-check passes (`n_chunks × 768 × 4 bytes ≤ measured_vector_bytes ≤ 3×`); a search returns top-*k* with all payload fields populated.

---

## 6. Phase 4 — Customized search tool (2 days)

Contribution #2, and the most technically interesting part.

### 6.1 Pipeline (Figure 5)

```
query → preprocess → medical term boosting → embed → Qdrant search (thr=0.65)
  ├─ satisfactory? → results + metadata
  ├─ no → L1: threshold reduction → sufficient? → yes: return | no ↓
  │        L2: linguistic simplification + re-embed → sufficient? → yes: return | no ↓
  │        L3: reconstruct from stored clinical semantic content
  │            → new semantic cluster → alternative query → search
```

All three thresholds, `max_results`, and context filters are **parametrically exposed** to the calling agent.

### 6.2 Medical term boosting

Repeat high-priority clinical phrases in the query so they dominate the embedding. The paper's named examples — "shortness of breath", "chest pain" — get higher vector-space weight.

Configurable priority lexicon: high-value symptom phrases, negation handling (`"no chest pain"` must not boost "chest pain" — a bug the paper never mentions and we should test for).

### 6.3 Level-3 fallback (the actual innovation)

Reconstruct the query from stored clinical semantic content: convert the symptom pattern into a **new semantic cluster** and build an alternative query from it. Genuinely different from top-*k* rotation.

### 6.4 Verifiable decision traces

Per query, retained: source, document ID, page, fragment length, retrieval time, plus **which fallback levels fired and why**. This is the "verifiable decision traces" claim, made real.

**Acceptance test:** a test that a query with no good match triggers L1→L2→L3 in order; a test that negation does not boost the negated term; a golden-trace test asserting every returned diagnosis cites at least one retrieved span (the paper's support requirement, §5.3 of the deep dive).

---

## 7. Phase 5 — The five agents (2–3 days)

### 7.1 Role → model binding

| Agent | Role | Paper model | Free substitute (verify at runtime) |
|---|---|---|---|
| 1 Clinical Text Clarifier | schema normalization | DeepSeek-R1 | Qwen3 / Qwen2.5-7B-Instruct |
| 2 Symptom RAG Analyzer | RAG hypothesis generation | Gemini-2.0-Flash | Gemini free tier → Qwen3-27B |
| 3 Evidence-Based Scanner | web evidence + grading | Qwen2-72B | Qwen3-8B/27B |
| 4 Clinical Data Fusion | synthesis + confidence | GPT-4o-mini | gpt-oss-20b / Qwen3-8B |
| 5 Adaptive Optimizer | feedback loop | Gemini-2.0-Flash | Gemini free tier → Qwen3-27B |

**Keep the role split identical** — the architecture claim is about roles, not model brands.

### 7.2 Agent 1 — Clinical Text Clarifier

Five ordered steps, each independently testable: text normalization → NER over `{Symptom, Disease, Medication, Test Result}` → association extraction → **real SNOMED CT / ICD-10 lookup** → strict JSON schema output.

Output schema (`schemas.py`):

```python
class ClinicalNote(BaseModel):
    demographics: Demographics | None
    symptoms: list[Symptom]          # {name, onset, duration, severity, negation}
    anatomical_regions: list[str]
    medications: list[str]
    labs: list[Lab]                  # {name, value, unit, ref_range, flag}
    imaging: list[str]
    test_results: list[str]
    ontology_codes: dict[str, str]   # concept -> SNOMED/ICD-10
    negations: list[str]
    raw_preserved: str
```

**Improvement over the paper:** wire a *real* ontology service and report **entity-linking F1** as its own metric. The paper claims SNOMED/ICD-10 mapping with no version, no accuracy, and unclear whether an LLM is generating the codes. We measure it.

### 7.3 Agent 2 — Symptom RAG Analyzer

Structured entities → dense query vector → cosine top-*k* → ranked preliminary diagnoses.

**Support requirement (keep this — it is the paper's best design decision):** a candidate diagnosis is retained only if at least one retrieved passage explicitly supports it. Enforced in code, not by prompt.

### 7.4 Agent 3 — Evidence-Based Scanner

Programmatic per-hypothesis search → Serper → PubMed / Google Scholar / NICE → per-hypothesis evidence summaries.

**Improvement:** explicit **Oxford/CEBM evidence levels** replacing the paper's qualitative hierarchy, stored in the payload and used as a numeric weight.

### 7.5 Agent 4 — Clinical Data Fusion

Cross-validation between the two evidence streams → conflict detection → calibrated confidence → structured explainable report.

### 7.6 Agent 5 — Adaptive Optimizer

τ gate → structured feedback report → requery guidance to agents 2 and 3 → max 3 iterations.

### 7.7 Strict output schemas everywhere

Every agent returns a Pydantic model. Parse failures are retried once with the validation error appended, then recorded as a failure rather than silently coerced. Silent coercion is how schema errors become invisible accuracy bugs.

**Acceptance test:** each agent has a unit test with a recorded fixture input and a golden output; each has a schema-violation test; each agent is individually runnable from CLI (`python -m cdss.cli.run_agent clarifier --fixture X`).

---

## 8. Phase 6 — Confidence, calibration, and the feedback loop (2 days)

### 8.1 The formula the paper omits — our strongest contribution

The paper says confidence "depends on the RAG similarity score, the web evidence score, and the degree of concordance" and then never writes it down. So we define it, and we report it as ours:

```
raw(d) = w_r · norm(rag_sim(d))
       + w_e · evidence_grade(d)
       + w_c · concordance(d)
       − w_k · conflict(d)
       − w_u · uncertainty(d)

confidence(d) = sigmoid(raw(d) / T)
```

| Term | Definition |
|---|---|
| `rag_sim(d)` | max cosine similarity among supporting passages for diagnosis *d* |
| `evidence_grade(d)` | weighted mean CEBM level of the supporting web evidence |
| `concordance(d)` | agreement between RAG-supported diagnoses and web-supported diagnoses (Jaccard-style) |
| `conflict(d)` | severity of detected contradictions between the two streams |
| `uncertainty(d)` | entropy / margin of the candidate ranking |

### 8.2 Then actually calibrate it

- Hold out a **development split**; fit temperature `T` (single scalar) and optionally `w` by maximizing NLL or minimizing ECE.
- Report **ECE**, **Brier score**, and a **reliability diagram** — before and after.
- Compare against `τ = 0.65` uncalibrated.

This turns the paper's weakest claim ("calibrated confidence" with no calibration) into our headline methodological contribution. **It directly answers the panel question we cannot answer for them.**

### 8.3 Conflict detection protocol

Beyond the paper's vague "conflict detection protocol":
- Contradictory diagnoses for the same symptom cluster
- Textbook recommendation vs current guideline (the paper's own example — generalize it)
- Evidence-level conflict (case report contradicting a guideline)
- Temporal conflict (older source superseding a newer one)

Each conflict is recorded with severity, both sources, and the resolution chosen. This feeds `conflict(d)` above.

### 8.4 Feedback loop — beating their plateau

The paper diagnoses its own ceiling: *"the feedback loop begins to repeat the same suggestions."* We fix it:

| Mechanism | Effect |
|---|---|
| **Iteration-diversity penalty** | Penalize a new suggestion whose cosine similarity to any previous suggestion exceeds θ_div → forces genuinely new queries |
| **Novelty score** | Reward suggestions that retrieve at least one passage not previously retrieved |
| **No-progress escalation detector** | If a suggestion set repeats, escalate strategy (widen query, drop a low-yield term, switch evidence tier) instead of re-issuing |
| **Diversity-aware early stopping** | Stop when confidence > τ **or** when expected marginal gain < ε — saves the wasted 4th and 5th iterations the paper burns |

**Expected effect:** accuracy that keeps improving past their iteration-3 plateau, with *fewer* total iterations.

### 8.5 Semantic Fusion — the "best data" aggregate

Combine RAG + web into a single aggregate via the consistency check, preserving per-source provenance so the final report can cite each claim.

**Acceptance test:** ECE after calibration < ECE before, on the dev split; reliability diagram roughly monotonic; feedback loop demonstrably terminates and does not repeat identical suggestions; a test that iteration 5's suggestions differ from iteration 3's.

---

## 9. Phase 7 — Evaluation harness (3–4 days) — the highest-value phase

**This is the phase that makes the project defensible. Do not cut it under any schedule pressure.**

### 9.1 Dataset loaders

| Dataset | Loader | Split | Scoring |
|---|---|---|---|
| MedQA | HF | full public test (~1,273) + matched 50 subset | **Exact-match** on A–E label |
| PubMedQA | HF `pqa_labeled` | full test (~450) + matched 50 subset | **Exact-match on yes/no/maybe** (primary) + cosine ≥ 0.80 (secondary, for comparability with the paper) |
| MedBullets | **verify access first** | matched 50 | as paper; substitute + flag if blocked |

**MedBullets is the biggest dataset risk.** Verify licensing before promising anything. If blocked, substitute a documented public equivalent (e.g. MedMCQA) and say so explicitly.

**Sampling protocol:** the matched-50 subsets must be the **same 50 questions the paper used**, as closely as reconstructable — otherwise the head-to-head is meaningless. Publish the sampled IDs in the repo.

### 9.2 Metrics — everything the paper reports, plus what it omits

**Reported (must reproduce):** accuracy, micro-P/R/F1, specificity, one-vs-all reduction, cosine ≥ θ.

**Added — statistical validity:**
- **Wilson** and **Clopper-Pearson** confidence intervals on every accuracy
- **McNemar test** — paired, because we evaluate on identical items. This is the *correct* test and the paper's unpaired comparison is not.
- **Bootstrap** CIs for deltas and for ablations
- **Multiple-comparison correction** (Holm–Bonferroni) when comparing against many baselines — the paper does not correct for 11 comparisons
- Effect sizes, not just p-values

**Added — clinical confidence:**
- **ECE**, **Brier**, reliability diagram
- **Selective prediction:** accuracy-vs-coverage curve as τ varies
- **AURC** (area under the risk-coverage curve)
- Abstention rate

**Added — retrieval (contribution #2, never measured by the paper):**
- **recall@k**, **MRR@10**, **nDCG@10** against gold supporting passages
- Embedder bake-off: BioBERT vs BioClinicalBERT vs MedCPT vs SPECTER2 vs BGE-M3 vs a hosted embedder
- Dense-only vs hybrid BM25+dense vs + cross-encoder reranker
- Effect of medical term boosting measured **directly in recall@k**, not via end-task accuracy

**Added — efficiency:**
- p50/p95/p99 latency per agent and end-to-end
- Web-crawl share of latency
- **Token cost per case**, per dataset
- Throughput under concurrency

### 9.3 Baselines — we run them ourselves

**The single biggest defensibility gain in the project.** Table 4 in the paper is eleven borrowed numbers. We run:

- Each agent's underlying model **zero-shot** (no architecture)
- RAG-only
- Web-only
- Fusion-removed
- Optimizer-removed
- Full `faithful`
- Full `improved`

All at **matched n, matched prompts, matched engine, same machine**. This makes the comparison valid in a way the paper's is not.

### 9.4 Ablation runner

Every Table 7 row re-run at **full n, 5 seeds**, reporting mean ± std and CIs:
Clinical Text Clarifier · Symptom RAG Analyzer · Evidence-Based Scanner · Clinical Data Fusion · Adaptive Optimizer · Medical Term Boosting · term-boosting-off · each of our new components (reranker, calibrator, abstention).

**Also fix the paper's ambiguity:** unambiguously name each removed component, and assert that removing exactly one thing changes exactly one thing (a test, not a claim).

### 9.5 Iteration sweep

Iterations 1..5 at n ≥ 300 with CIs, reporting accuracy **and** wall-clock **and** marginal gain **and** suggestion-diversity. Our curve should show a real plateau at a *different* place than theirs, and we can explain why.

### 9.6 Determinism

Fully deterministic offline mode: pinned seeds, snapshotted evidence, cached LLM responses. A second identical run must produce byte-identical results. Live mode is separate and clearly labelled.

**Acceptance test:** `make eval --config faithful --n 50` completes and emits a results table with CIs and p-values; a second run is byte-identical; the baseline table's numbers are all produced by us, with model manifests attached.

---

## 10. Phase 8 — Prototype UI (2 days)

| Requirement | Implementation |
|---|---|
| Single-page input | React + Vite + TS; symptom/history textarea |
| Real-time progress | **SSE stream** per agent event — paper used plain REST; SSE is strictly better |
| Agent trace | Timeline of all 5 agents: input, output, latency, model, tokens, cost |
| Evidence panel | Retrieved spans + web sources with CEBM level and clickable citation |
| Confidence badge | Per-diagnosis calibrated confidence, colour-coded against τ |
| Conflict panel | Detected conflicts with both sources and the resolution |
| Iteration panel | Feedback reports per iteration, with the diversity metric |
| Report | The paper's annotated report: source used, possible sources of variation, confidence |
| Audit export | Download the full run as JSONL |
| Privacy | Synthetic/de-identified data only; no persistent patient data; RBAC |

**Acceptance test:** a full case runs end to end in the UI, each agent's output appears as it completes, the exported trace replays offline, and no patient-identifying field is ever persisted.

---

## 11. Phase 9 — Extensions (ongoing, T2/T3)

- **Rare-case & multi-disease stress set** — directly answers the paper's Limitation 1
- **Safety red-team set** — leading questions, contraindication traps, dosage errors, out-of-scope requests, fabricated vitals. Report harmful-omission and unsafe-recommendation rates
- **Abstention agent** — "insufficient evidence" as a first-class output
- **Bias analysis** across demographics (the paper's own ref [27] documents bias in Gemini 2.0 Flash)
- **Cost/latency optimization** — parallel web scans, snapshot caching, smaller models for easy roles. Target: attack the 85–90% web-crawl bottleneck
- **Model-sensitivity ablation** — swap each agent's model across ≥3 options. Converts a weakness (we use substitutes) into a finding the paper never produced
- **Multilingual / demographic robustness** if time allows

---

## 12. Trade-offs — full register

The complete trade-off register lives in §18 of `docs/PAPER_DEEP_DIVE.md`. Condensed operational version:

### 12.1 Faithful vs. improved
Two configs, one codebase, `faithful` **frozen** before `improved` begins. A single improved system cannot honestly be compared to their numbers; a single faithful system cannot beat them.
**Cost:** ~1.6× engineering time. **Risk:** scope creep. **Mitigation:** `configs/`, code freeze, ADR log.

### 12.2 Free-tier compute vs. evaluation scale ★ binding constraint
27k calls needed vs ~200/day available on OpenRouter free.
**Resolution:** Ollama local for bulk statistics (unlimited, internally valid because the engine is fixed); free cloud for headline runs and the demo; optional paid confirmation run.
**Cost:** local models depress absolute accuracy. **Mitigation:** never compare across engines; always attach the model manifest.

### 12.3 Model substitution vs. exact reproduction
Gemini-2.0-Flash and GPT-4o-mini are retired. DeepSeek-R1 / Qwen2-72B aren't free-tier.
**Resolution:** keep the role split identical; record every model ID + version + date; run the model-sensitivity ablation to turn the substitution into a finding.
**Hard rule:** no number without a manifest.

### 12.4 Determinism vs. live web freshness
**Resolution:** both modes, clearly labelled. Evaluation on the snapshot (with date + hash); live demo shows real NICE/PubMed retrieval. Never conflate them in one table — the paper does, which is part of why its numbers don't reproduce.

### 12.5 Rigor vs. time-to-presentable
**Resolution:** the T1/T2/T3 evidence ladder (§0). Target T2, present T1, state the ladder openly. Showing the panel your own rigour ladder is itself impressive.

### 12.6 BioBERT vs. modern retrievers
**Resolution:** BioBERT stays permanently in `faithful` as the reference arm. The bake-off lives in `improved`. Narrative: *"we reproduced their retriever, showed it is the weakest component, and quantified by how much."* Better paper than either alone.

### 12.7 Improvement claims vs. possible regression ★
**Real risk:** we gain rigor and *lose* accuracy. Small free models + a larger eval set can easily put us below 94%.
**This is acceptable and must be pre-accepted.** The finding would be *"the reported 94% does not replicate at scale"* — legitimate, valuable, publishable, arguably the most important one.
**Never let sunk feelings pressure a favourable framing.** Pre-write it: *"we measure X at Y% with CI [a,b]; the paper's point estimate falls inside our interval, so the architecture claim survives but the precision claim does not."*

### 12.8 Scope cuts — in this order only
1. T3 items (full splits, safety red-team, demo polish)
2. MedBullets → substitute + flag, if licensed-blocked
3. Embedder bake-off → cap at 3 models
4. UI → functional trace viewer only
5. **Never cut:** matched-50 baselines, confidence intervals, McNemar, retrieval metrics

### 12.9 Ethical / regulatory — non-negotiables
Synthetic/de-identified data only; no real patient data without IRB. Explicit "not a diagnostic device; clinician decision support; verify independently" on every output. Logged attributable traces. Abstention path required. Bias analysis. No dosing recommendation without dose + units + citation.

---

## 13. Risks

| # | Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|---|
| 1 | **MedBullets not publicly licensed** | High | Medium | Verify in week 1; substitute MedMCQA + flag; MedQA + PubMedQA are sufficient |
| 2 | **Free-tier throughput too slow** | High | High | Ollama for bulk; disk cache; concurrency + backoff; T1/T2/T3 ladder |
| 3 | **Free-tier models too weak** | High | High | Engine-per-purpose split (§3.4); never compare across engines; model manifest always |
| 4 | **Paper's confidence formula unrecoverable** | Medium | Certain | Accepted. We define our own and label it ours. This *is* a contribution. |
| 5 | **Qdrant/BioBERT install friction on macOS** | Low | Medium | Use `qdrant-client` local mode or Docker; pin versions; document setup |
| 6 | **Serper rate limits / cost** | Low | Low | Snapshot mode for eval; live only for demo |
| 7 | **We fail to reproduce 94%** | Medium | **High** | Pre-accepted (§12.7). Reframe as a finding, not a failure. |
| 8 | **Scope creep across two configs** | Medium | High | Freeze `faithful` first; ADR log; weekly scope review |
| 9 | **Live web breaks determinism** | Medium | High | Snapshot-first; live is a separate, labelled mode |
| 10 | **Panel asks something not covered** | Low | Medium | `PAPER_DEEP_DIVE.md` §19 is the defence; keep it current |

---

## 14. Definition of done

**T1 — Presentation-ready**
- [ ] `faithful` config reproduces the 5-agent pipeline end to end
- [ ] Vector index built; measured pages/sec, chunk counts, index bytes reported
- [ ] Matched 50/50/50 run with Wilson CIs and McNemar p-values
- [ ] One ablation sweep (all 7 rows) at matched n
- [ ] Retrieval metrics: recall@5/10, MRR@10, nDCG@10
- [ ] Latency p50/p95 + token cost per case
- [ ] All baselines run by us, not borrowed
- [ ] Working UI with live agent trace
- [ ] Second identical run byte-identical

**T2 — Strong (recommended target)**
- [ ] n=300 per dataset, both configs
- [ ] Full ablation at n=300, 3 seeds, mean ± std
- [ ] ECE/Brier before vs after calibration + reliability diagram
- [ ] Embedder bake-off (≥3 models) + hybrid + reranker
- [ ] Diversity-aware feedback loop beating their iteration-3 plateau
- [ ] Selective-risk / accuracy-vs-coverage curve
- [ ] Model-sensitivity ablation
- [ ] `docs/RESULTS.md` with every table and its model manifest

**T3 — Publishable**
- [ ] Full public splits (MedQA ~1,273; PubMedQA ~450)
- [ ] 5 seeds everywhere
- [ ] Safety red-team + abstention metrics
- [ ] Rare-case / multi-disease stress set
- [ ] Live demo with real NICE/PubMed retrieval
- [ ] Latency optimized ≥5× vs paper's 2.5–5.5 min

---

## 15. Immediate next steps

1. **Verify MedBullets access** (risk 1 — cheapest to resolve, biggest impact if blocked)
2. `make bootstrap` — Phase 0 scaffold
3. `make discover-models` — see what free models are actually available today
4. Run the T1 smoke path on Ollama: 1 case, all 5 agents, full trace
5. Then Phase 2 in earnest

**Start with step 3 and 4 in parallel.** A single end-to-end case on a local model de-risks the entire project faster than any other activity — it proves the pipeline shape, the schema contracts, and the trace format, all before any dataset or index work.