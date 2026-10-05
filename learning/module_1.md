# Module 1 — Contracts, Core Types & Test Harness

**Status:** COMPLETE ✅  
**Gate:** 39/39 tests passing  
**Runtime:** Python 3.11.9, pytest 9.1.1

---

## What Was Built

### Project Scaffold

The repository was bootstrapped from an empty directory. The following top-level structure now exists:

```
DevTrace/
├── .env.example          # Template for secrets — never committed
├── .gitignore
├── pyproject.toml        # Build config + pytest settings
├── requirements.txt      # All project dependencies
├── venv/                 # Python 3.11.9 virtual environment
├── src/
│   ├── __init__.py
│   ├── config.py         # Centralised settings (pydantic-settings)
│   ├── errors.py         # Project-level exception hierarchy
│   ├── models/
│   │   ├── __init__.py   # Clean re-exports
│   │   ├── enums.py      # All shared enums
│   │   └── contracts.py  # All Pydantic data contracts
│   └── llm/
│       ├── __init__.py
│       ├── base.py       # Abstract LLMClient + LLMResponse
│       ├── fake.py       # FakeLLMClient (deterministic, offline)
│       └── gemini.py     # Production GeminiClient
├── tests/
│   ├── __init__.py
│   └── unit/
│       ├── __init__.py
│       └── test_m1_contracts.py   # 39 tests, all offline
├── data/
│   ├── corpus/           # (empty — populated in M2)
│   └── eval/             # (empty — populated in M2)
└── learning/
    └── module_1.md       # This file
```

---

## Key Design Decisions

### 1. `model_config = {"frozen": True}` on all contracts
All Pydantic models are immutable after construction. This prevents accidental mutation as objects flow through the pipeline across modules.

### 2. Structural invariants enforced in contracts
Two business-critical rules are enforced as Pydantic `model_validator`s rather than being left for runtime checks:
- `DiagnosisResult` must contain **exactly one `root_cause` claim** — validated at construction time.
- `FinalAnswer` with `outcome=ANSWERED` must have `completeness` set, and vice versa — also validated at construction time. This prevents a whole class of silent errors in M6.

### 3. Deliberate separation of Citation Validity vs Citation Correctness
The `DiagnosisResult` contract accepts any `evidence_ids` strings — it does not validate that they exist in the applicable bundle. That check (`cited_ids ⊆ applicable_ids`) is explicitly M5's job. The comment in the contract code makes this boundary explicit to future implementers.

### 4. FakeLLMClient has three modes
- **Fixed response** — repeats the same string on every call. Good for simple generation tests.
- **Sequence** — returns responses in order, with exceptions embeddable at specific positions. Essential for testing the one-shot retry in M6.
- **Global exception** — every call raises. Tests DEGRADED mode.
A `call_count` property lets tests assert exactly how many LLM calls a pipeline step made.

### 5. GeminiClient is lazily imported
The `import google.generativeai` statement lives inside `__init__`, not at module top-level. This means importing `src.llm.gemini` does not trigger the SDK import, and tests that never instantiate `GeminiClient` pay zero cost.

### 6. Config singleton vs per-test override
`src/config.py` exports a module-level `settings` singleton. Tests that need to override config use `monkeypatch.setattr("src.llm.gemini.settings", DevTraceConfig(...))` — this patches the reference inside the gemini module without affecting anything else.

---

## Contracts Summary

| Model | Purpose | Key constraints |
|-------|---------|----------------|
| `TroubleshootingIncident` | Incident input | `description` required, all else optional |
| `Document` | Corpus document | `applies_to` defaults to `"*"` |
| `EvidenceChunk` | Retrievable unit | Carries `doc_id` for traceability |
| `RetrievalResult` | M3 output | Score validated to `[0, 1]` |
| `ApplicabilityResult` | M4 output | `applicable: bool` + reason |
| `DiagnosisClaim` | Single LLM claim | Auto-UUID, enum-validated `role` |
| `DiagnosisResult` | Claims collection | Exactly one `root_cause` enforced |
| `VerificationVerdict` | M5 output | Per-claim PASS/FAIL with support/contradiction fields |
| `FinalAnswer` | User-facing answer | ANSWERED ↔ completeness invariant enforced |
| `Trace` | Full pipeline audit | All optional except `incident`; grows across M2–M6 |

## Enums Summary

| Enum | Values |
|------|--------|
| `SystemOutcome` | `ANSWERED`, `INSUFFICIENT_EVIDENCE`, `NEEDS_INFO`, `DEGRADED` |
| `AnswerCompleteness` | `FULL`, `PARTIAL` |
| `ClaimRole` | `root_cause`, `fix`, `explanation` |
| `RetrievalSource` | `semantic`, `keyword`, `hybrid` |
| `VerificationStatus` | `PASS`, `FAIL` |
| `SupportDecision` | `SUPPORTED`, `UNSUPPORTED`, `INSUFFICIENT` |
| `ContradictionDecision` | `NONE`, `CONTRADICTED`, `UNRESOLVED` |

---

## What Is Explicitly NOT Here (Deferred to Later Modules)

- **M2:** Version normalization/parsing, incident signal extraction, corpus ingestion, evaluation dataset
- **M3:** Embeddings, Chroma, BM25, hybrid retrieval
- **M4:** Version applicability checking, NEEDS_INFO logic, diagnosis generation prompts
- **M5:** Evidence verification, contradiction resolution, citation validity checking
- **M6:** Retry orchestration, FULL/PARTIAL outcome assembly
- **M7:** Evaluation runner, metrics, baselines
- **M8:** Streamlit UI
