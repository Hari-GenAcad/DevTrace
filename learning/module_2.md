# Module 2 — Corpus, Ingestion & Evaluation Dataset

**Status:** COMPLETE ✅  
**Gate:** 101/101 tests passing (39 M1 + 62 M2)  
**Runtime:** 0.74s, fully offline

---

## What Was Built

### 1. DevCore Corpus (`data/corpus/devcore_corpus.json`)

A controlled, fictional 30-document knowledge base covering realistic developer troubleshooting scenarios.

**Topics covered:**
| Topic | Document IDs |
|-------|-------------|
| Authentication (2.x & 3.x) | AUTH-001–005, TS-001, TS-004 |
| Rate Limiting | RATE-001–002, TS-005 |
| Webhooks (2.x & 3.x) | WEBHOOK-001–003, TS-002, TS-006 |
| SDK Migration 2.x→3.x | SDK-001 |
| SDK Configuration | SDK-002–004 |
| API Behavior | API-001–005 |
| Configuration/Env Vars | CFG-001–002 |
| Common Errors | ERR-001–003 |
| Performance | PERF-001–002 |
| Troubleshooting | TS-001–006 |
| Security | SECURITY-001 |

**Version coverage:**
- `>=2.0,<3.0` — SDK 2.x specific documents
- `>=3.0,<4.0` — SDK 3.x specific documents  
- `*` — Version-agnostic documents

**Deliberate traps planted (for DevTrace demonstration):**
- AUTH-001 (2.x AUTH_401 fix) vs AUTH-002 (3.x AUTH_401 fix) — same symptom, different fix
- WEBHOOK-001 (SHA1 signatures) vs WEBHOOK-002 (SHA256 signatures)
- API-002 (offset pagination) vs API-003 (cursor pagination)
- SDK-002 (2.x config) vs SDK-003 (3.x config)
- AUTH-005 (API key creation) — highly relevant but 2.x only

### 2. Corpus Loader (`src/ingestion/loader.py`)

**Responsibilities:**
- Reads `devcore_corpus.json` from disk
- Validates each entry (required fields: `doc_id`, `title`, `content`)
- Rejects duplicate doc IDs
- Builds `Document` and `EvidenceChunk` objects from M1 contracts
- Two modes: `strict=True` (raises on any malformed entry), `strict=False` (skips bad entries)

**Chunking strategy (M2):**  
One chunk per document. `chunk_id = "{doc_id}-C01"`. Keeps IDs stable and predictable. The `-C01` suffix convention leaves room for M3/M4 to introduce finer chunks if retrieval evaluation proves it necessary.

> DEFERRED TO M3/M4: Semantic or token-aware chunking, if needed based on retrieval evaluation.

### 3. Signal Extraction (`src/normalization/signals.py`)

Fully deterministic, regex-based extraction. **No LLM.**

**What is extracted:**

| Signal | Method | Examples |
|--------|--------|---------|
| Error codes | `r"\b([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\b"` | `AUTH_401`, `RATE_429`, `ERR_TIMEOUT` |
| Upgrade version pair | `"from X to Y"` pattern | `from 2.8 to 3.1` → previous=2.8, current=3.1 |
| Single current version | One non-historical version mention | `SDK 3.1` → current=3.1 |
| Historical version | `"used to run X"`, `"previously on X"` | Not treated as current |
| Ambiguous versions | Multiple mentions, no clear upgrade context | current=None (safe) |
| Technical terms | Keyword set lookup | `oauth`, `bearer`, `webhook`, `hmac`, etc. |

**Key design decision — historical markers:**  
The statement "I used to run 2.8" does NOT set `current_version = 2.8`. The system intentionally leaves `current_version = None` in ambiguous cases, which triggers the NEEDS_INFO path in M4. This prevents wrong-version diagnoses.

**Structured fields always win:**  
If `current_version="3.1"` is passed explicitly, it overrides any version that might be extractable from the description text.

### 4. Incident Normalizer (`src/normalization/normalizer.py`)

Wraps signal extraction and M1 contract construction into a single step.

**Input:** Free-text description + optional structured kwargs  
**Output:** `NormalizedIncident(incident: TroubleshootingIncident, signals: ExtractedSignals)`

`ExtractedSignals.to_dict()` produces a plain serializable dict — suitable for populating `Trace.extracted_signals` in M6.

### 5. Evaluation Dataset (`data/eval/eval_dataset.json`)

37 cases across 8 classes:

| Class | Count | Purpose |
|-------|-------|---------|
| `straightforward` | 8 | Single or obvious docs. Baseline cases. |
| `multi_document` | 8 | Require multiple applicable docs. |
| `version_conflict` | 5 | Highly relevant wrong-version doc traps. |
| `similar_error` | 4 | e.g. AUTH_401 vs AUTH_403, NOT_FOUND vs AUTH_403. |
| `near_miss` | 4 | Topic present but evidence insufficient. |
| `unsupported` | 2 | Corpus cannot answer (billing, third-party). |
| `missing_info` | 3 | AUTH_401/webhooks without version → NEEDS_INFO. |
| `contradictory_evidence` | 3 | Applicable docs partially conflict. |

**Each case includes:**
- `gold_doc_ids`: documents that legitimately support the expected answer
- `forbidden_doc_ids`: wrong-version/inapplicable documents that must NOT be cited
- `expected_outcome` and `expected_completeness`
- `notes` explaining the design intent

### 6. Evaluation Dataset Contract (`src/evaluation/dataset.py`)

`EvaluationCase` and `EvaluationDataset` Pydantic models with:
- `case_class` validated against controlled vocabulary
- completeness ↔ ANSWERED invariant (mirrors M1's FinalAnswer invariant)
- unique `case_id` enforced at collection level
- optional corpus integrity check: validates all gold/forbidden `doc_id` references exist in the loaded corpus

---

## Stable ID Summary

| Object | ID Pattern | Example |
|--------|-----------|---------|
| Document | Explicit from corpus | `AUTH-002` |
| EvidenceChunk | `{doc_id}-C01` | `AUTH-002-C01` |
| EvaluationCase | Explicit `EVAL-{CLASS}{NN}` | `EVAL-C01` |

All IDs are deterministic: loading the same corpus twice produces identical IDs.

---

## What M3 Can Now Do

M3 can immediately start with:
```
load_corpus_and_chunks() 
  → (documents, chunks)
  → build Chroma index on chunks
  → build BM25 index on chunks
```

All chunk IDs are stable and traceable. M3 does not need to touch the corpus data format.

## What M4 Can Now Do

M4 receives `NormalizedIncident` from M2 and can immediately:
- Read `incident.current_version` for applicability checking
- Use `signals.error_codes` for retrieval boosting
- Apply `document.applies_to` (PEP 440 string) for version range evaluation

## What M7 Can Now Do

M7 can load `load_eval_dataset(valid_doc_ids=corpus_doc_ids)` and immediately have 37 structured test cases with gold labels, forbidden documents, and expected outcomes.

---

## Explicitly NOT Here (Deferred)

- **DEFERRED TO M3:** Chroma index building, BM25 index, semantic embeddings, hybrid retrieval
- **DEFERRED TO M4:** Version range comparison (`packaging.version`), NEEDS_INFO decision logic
- **DEFERRED TO M5:** Evidence verification, citation validity checking
- **DEFERRED TO M7:** Metric computation, baseline evaluation runners
