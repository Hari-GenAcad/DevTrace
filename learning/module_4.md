# DevTrace — Module 4: Applicability, NEEDS_INFO & Baseline Diagnosis Pipeline

## What is Module 4?

Module 4 is the point where DevTrace first becomes a **real troubleshooting pipeline**.

After M1 (contracts), M2 (corpus & normalization), and M3 (retrieval), we now have:

- A structured incident with extracted signals
- Candidate evidence chunks ranked by relevance

But having _candidates_ is not the same as having _correct_ evidence.

Module 4 adds three critical layers:

1. **Deterministic NEEDS_INFO gate** — do we have enough information to even try?
2. **Deterministic applicability filtering** — is each retrieved chunk actually relevant to _this version_?
3. **Baseline diagnosis generation** — structured LLM reasoning from applicable evidence only

---

## Why Retrieval Alone is Insufficient

M3's hybrid retriever is good at finding _semantically and lexically similar_ documents.

But similarity is not the same as applicability.

Consider this incident:
```
"AUTH_401 started after upgrading from SDK 2.8 to 3.1."
```

M3 correctly retrieves:
- `AUTH-001` — "Resolving AUTH_401 for SDK 2.x" (score: 0.87)
- `AUTH-002` — "Resolving AUTH_401 for SDK 3.x" (score: 0.85)

Both documents are **highly relevant**. But only `AUTH-002` is **applicable**.

Why? Because the incident is happening on SDK 3.1, and `AUTH-001` describes a completely
different authentication system (ApiKey headers vs. OAuth Bearer tokens). Using `AUTH-001`
to diagnose a 3.x problem would produce **incorrect guidance** — it would tell the developer
to check their API key when the real problem is their OAuth token.

This is the core architectural insight: **retrieval finds candidates; applicability decides truth**.

---

## Why Applicability is Separate from Retrieval

There are two tempting shortcut approaches that DevTrace deliberately avoids:

### Shortcut 1: Filter Chroma by metadata during retrieval
You could add `where={"major_version": 3}` to the Chroma query.

**Problem:** The retrieval step wouldn't see 2.x documents at all.
But sometimes they're useful — for example, a migration guide covering `>=2.0,<4.0`
spans both versions and is valid for understanding what _changed_.

M3 retrieves broadly. M4 decides narrowly.

### Shortcut 2: Use LLM to decide applicability
You could ask Gemini "Does this doc apply to SDK 3.1?"

**Problem:** This is expensive, non-deterministic, and unnecessary.
Version applicability is a simple, explicit, rule-based calculation.
The `applies_to` field already contains the version range.

Deterministic rules are:
- Auditable
- Reproducible
- Fast
- Testable

---

## How Version Applicability Works

The DevCore corpus uses PEP 440-style version specifiers in `applies_to`:

| applies_to | Covers |
|---|---|
| `>=2.0,<3.0` | SDK 2.x (major version 2) |
| `>=3.0,<4.0` | SDK 3.x (major version 3) |
| `>=2.0,<4.0` | Both 2.x and 3.x |
| `*` | All versions (version-agnostic) |

### The Five Rules

**Rule 1 — Version-agnostic documents**
`applies_to="*"` → APPLICABLE to any incident, regardless of version.

_Example:_ `AUTH-003` (AUTH_403 Forbidden) applies to all versions because the cause
(wrong scope) is the same in 2.x and 3.x.

**Rule 2 — Exact major-version compatibility**
`current_version=3.1` + `applies_to=">=3.0,<4.0"` → major 3 ∈ {3} → **APPLICABLE**

**Rule 3 — Wrong major version**
`current_version=3.1` + `applies_to=">=2.0,<3.0"` → major 3 ∉ {2} → **NOT_APPLICABLE**

**Rule 4 — Missing current version**
`current_version=None` + `applies_to=">=3.0,<4.0"` → **UNKNOWN**

UNKNOWN is excluded from the evidence bundle (conservative choice).

**Rule 5 — Previous version is context only**
`previous_version=2.8`, `current_version=3.1` → a 2.x-only document is still **NOT_APPLICABLE**.

The previous version explains _where you came from_, not where you are now.

---

## Why current_version Matters More Than previous_version

The bug is happening **now**, in the **current environment**.

The developer's problem is: "Why does AUTH_401 occur on my 3.1 SDK?" not "Why did it work on 2.8?"

A document that says "In 2.x, API keys are formatted as X" describes history, not the solution.

However, **migration documentation** (like `SDK-001`, which covers the 2.8→3.1 migration
and spans `>=2.0,<4.0`) remains applicable because it explicitly addresses the transition.

---

## Why NEEDS_INFO Exists

Sometimes the system genuinely cannot proceed without more information.

Two situations trigger NEEDS_INFO:

### Case A: Missing version + competing evidence

If the retrieved evidence spans both 2.x and 3.x, and the incident has no version,
we face a **genuine ambiguity**:

- AUTH_401 in 2.x: wrong API key format
- AUTH_401 in 3.x: expired OAuth Bearer token

These are completely different problems with completely different fixes.

Guessing would be **worse than saying "I don't know"**, because an incorrect fix wastes the developer's time.

**NEEDS_INFO is not a failure — it's honesty.**

### Case B: No meaningful technical signal

If the incident says "something is wrong with our API", we have nothing to retrieve meaningfully.

The NEEDS_INFO check is NOT based on word count. A 3-word incident "AUTH_401 on 3.1" has
excellent signal. A 50-word incident "something seems to be having some sort of issue with
one of our things" has almost no signal.

We check for:
- Error codes (strongest signal)
- Version (scopes the evidence space)
- Technical terms (oauth, webhook, bearer, etc.)
- Specific description with identifiable technical content

---

## How Applicable Evidence is Constructed

```
M3 retrieval results (ranked by fused score)
    ↓
For each chunk:
    check_applicability(chunk, current_version)
    → APPLICABLE / NOT_APPLICABLE / UNKNOWN
    ↓
applicable_results = [chunks where decision == APPLICABLE]
```

All decisions are stored in `applicability_decisions` for full traceability.
Even NOT_APPLICABLE chunks remain visible in the trace — you can see exactly
what was retrieved and why it was excluded.

---

## How Gemini Generates Structured Claims

The diagnosis generator sends Gemini a structured prompt containing:

1. The normalized incident description
2. Extracted signals (error codes, versions)
3. **Only the applicable evidence** — NOT_APPLICABLE chunks are never shown

The prompt instructs Gemini to:
- Use only the provided evidence (no outside knowledge)
- Return structured JSON (not prose)
- Tag every claim with the chunk IDs that support it
- Distinguish root cause from fix from explanation
- Produce exactly one root_cause claim

The output JSON is parsed and validated against the M1 `DiagnosisResult` schema.

### Why the diagnosis is still "untrusted"

M4 generates a diagnosis. But Gemini might:
- Cite a chunk ID that doesn't actually support the claim
- Mis-read the evidence
- Make a logically valid-looking claim that isn't actually grounded

M5 will independently verify each claim by asking:
> "Does chunk `AUTH-002-C01` actually support this root_cause claim text?"

M4 does not perform this verification. Attempting to do so in M4 would be premature
(we'd need to re-read the evidence from the model's own output, creating a circular dependency).

---

## The Baseline Pipeline

```
normalize_incident(description, ...)
    ↓
retriever.retrieve_as_contracts(normalized)    ← M3 HybridRetriever
    ↓
needs_info_check(normalized, retrieval_results)
    → if triggered: return NEEDS_INFO
    ↓
filter_applicable(retrieval_results, current_version)
    → applicable_results, applicability_decisions
    ↓
generator.generate(normalized, applicable_results)  ← Gemini via M1 LLMClient
    ↓
BaselinePipelineResult
```

The pipeline is a **simple linear function** — not an agent, not a graph.

The orchestrator's only job is to connect the pieces in the right order
and pass the right data between them.

---

## File Structure

```
src/
    applicability/
        __init__.py          — public API (check_applicability, filter_applicable)
        checker.py           — version rules, _decide(), check_applicability()

    diagnosis/
        __init__.py          — public API
        needs_info.py        — NeedsInfoResult, needs_info_check()
        prompts.py           — build_diagnosis_prompt(), evidence block formatter
        generator.py         — DiagnosisGenerator, parse_diagnosis()

    pipeline/
        __init__.py          — public API
        baseline.py          — run_baseline_diagnosis(), BaselinePipelineResult

scripts/
    smoke_test_m4.py         — optional live Gemini smoke test (not in pytest)

tests/unit/
    test_m4_pipeline.py      — ~54 deterministic offline tests
```

---

## What M5 Will Receive from M4

M4's `BaselinePipelineResult` contains everything M5 needs:

| Field | M5 usage |
|---|---|
| `normalized` | Incident context for verification prompts |
| `applicable_results` | The evidence bundle M5 evaluates citations against |
| `applicability_decisions` | Audit trail for M6 trace assembly |
| `diagnosis` | The `DiagnosisResult` containing claims to verify |
| `diagnosis.claims[i].evidence_ids` | The IDs M5 will check for citation support |

M5's question for each claim will be:
> "Does the cited evidence (`evidence_ids`) actually support this claim (`text`)?"

M5 never receives NOT_APPLICABLE evidence — the boundary is enforced at M4.

---

## Running the Tests

```powershell
# All 219+ tests (M1 + M2 + M3 + M4)
.\venv\Scripts\python.exe -m pytest

# M4 tests only
.\venv\Scripts\python.exe -m pytest tests/unit/test_m4_pipeline.py -v

# Optional live smoke test (requires GEMINI_API_KEY)
.\venv\Scripts\python.exe scripts\smoke_test_m4.py
```

All normal pytest tests are offline and deterministic. No API key required.
