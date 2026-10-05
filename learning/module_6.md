# Module 6: Retry, Root-Cause Survival & Final Outcome Orchestration

## Overview

Module 6 is the final intelligence layer of DevTrace's core troubleshooting pipeline.

It takes the M4 baseline diagnosis and the M5 evidence verification result and
transforms them into a **reliable, trustworthy final answer** — or explicitly
declares that the evidence was insufficient, or that a system failure prevented
completion.

M6 introduces exactly two new concepts:

1. **Root-cause survival** — a boolean gate that separates a plausible diagnosis from a trustworthy one.
2. **One targeted retry** — a single, focused second attempt when the root cause fails.

---

## Why Root-Cause Survival Matters

The fundamental DevTrace thesis is:

> A plausible diagnosis is NOT enough.

M3 retrieval can surface semantically relevant documents.  
M4 applicability can narrow them to the correct version range.  
M4 diagnosis can produce a structurally valid JSON response.  
M5 can find valid citations and call the evidence "supported."

And yet, **the root cause claim can still be wrong**.

Consider:

```
Incident: AUTH_401 started after upgrading from SDK 2.8 to 3.1.

M4 diagnosis:
  root_cause: "The SDK uses API key authentication in 3.x." ← WRONG
  fix: "Add your API key to the X-API-Key header." ← WRONG (3.x uses OAuth Bearer)
  explanation: "API keys are deprecated in 3.x." ← WRONG

M5 verification:
  root_cause: citation_correct=True, sufficient=False → REJECTED
  fix: REJECTED
  explanation: REJECTED
```

Without root-cause survival, one might present a "partial answer" built on a
rejected root cause. M6 prevents this by treating the root cause as the
**critical claim** whose survival is a precondition for any final answer.

### The Rule

```
root_cause_survived = (root_cause_claim.verification == VERIFIED)
```

If this is False:
- No answer is presented.
- A targeted retry is attempted.
- If retry also fails → INSUFFICIENT_EVIDENCE.

If this is True:
- Final outcome is classified (FULL vs PARTIAL based on fix/explanation).
- Final answer is assembled from surviving verified claims only.

---

## Why a Failed Root Cause Triggers (at most) One Retry

The retry exists because LLM diagnosis is not deterministic. The first
attempt may produce a root cause claim that:

- Cited evidence at too high a level of abstraction.
- Failed to recognize the specific version-change signal in the evidence.
- Produced a claim that the verifier could not confirm as sufficient.

A targeted retry with **explicit verification failure feedback** gives the
model a concrete second chance:

> "Your first root cause was rejected because: 'Evidence does not establish
>  the causal link to the version upgrade.' Please reconsider using the
>  same applicable evidence."

This is not a loop. The spec is explicit:

```
max_retries = 1
```

After the retry:
- If root cause survives → classify and assemble the final answer.
- If root cause fails again → INSUFFICIENT_EVIDENCE. **STOP.**

There is no third attempt, no progressive relaxation, no fallback to a
less-strict verification. The architecture is designed to be safe by default.

---

## Why the Retry is Targeted

A naive retry would simply re-send the original diagnosis prompt and hope for
a different result. This is wasteful and unlikely to succeed.

The targeted retry:
1. Shows the model its first diagnosis.
2. Shows the verification verdict and rejection reason for each claim.
3. Explicitly instructs the model to focus on what the evidence **directly
   and explicitly** states about the root cause.
4. Uses the **same applicable evidence bundle** — no new retrieval.

What does NOT change between the initial attempt and the retry:
- The incident description.
- The applicable evidence (M4 boundary is not re-run).
- The version constraints.
- The required output schema.

The retry is a *focused revision*, not a fresh start with different data.

---

## Why Rejected Claims Cannot Enter the Final Answer

M6 assembles the final answer by collecting only VERIFIED claims:

```python
verified_ids = set(verification.verified_claim_ids)
for claim in diagnosis.claims:
    if claim.claim_id in verified_ids:
        # include in answer
```

If the fix claim was rejected (insufficient evidence), it is excluded even
if it appears in the retry diagnosis. The user never sees:

> Fix: Apply the authentication header. (← rejected claim, evidence not found)

They see only what is demonstrably supported by applicable evidence.

This applies at all levels:
- root_cause not verified → no answer (INSUFFICIENT_EVIDENCE).
- fix not verified → root_cause appears in answer, fix does not.
- explanation not verified → only root_cause + fix appear.

---

## PARTIAL vs INSUFFICIENT_EVIDENCE

Both look like "incomplete" answers, but they are fundamentally different:

| Outcome | Root cause | Fix | Why |
|---------|-----------|-----|-----|
| `ANSWERED_PARTIAL` | VERIFIED | REJECTED or absent | The root is established; the fix lacks evidence |
| `INSUFFICIENT_EVIDENCE` | REJECTED | Any | The root was never established |

`ANSWERED_PARTIAL` is **safe to present**. The user knows why the issue
occurs, even if the specific fix could not be confirmed from the corpus.

`INSUFFICIENT_EVIDENCE` is **not safe to present as a diagnosis**. Showing
the user any claimed root cause would be misleading.

### Classification rules

```
root_cause VERIFIED AND fix VERIFIED AND explanation VERIFIED → ANSWERED_FULL
root_cause VERIFIED AND fix VERIFIED (no/rejected explanation) → ANSWERED_PARTIAL
root_cause VERIFIED AND fix REJECTED/absent → ANSWERED_PARTIAL
root_cause REJECTED (no retry success) → INSUFFICIENT_EVIDENCE
```

---

## INSUFFICIENT_EVIDENCE vs DEGRADED

These must never be confused:

| Outcome | What happened | What to tell the user |
|---------|--------------|----------------------|
| `INSUFFICIENT_EVIDENCE` | Diagnosis quality failed; evidence doesn't support root cause | "We could not confirm the root cause from the available documentation" |
| `DEGRADED` | System failure (LLM API error, schema error, infra failure) | "The troubleshooting system encountered an error; please try again" |

Converting a system failure into INSUFFICIENT_EVIDENCE would be misleading:
the documentation may well contain the answer — the infrastructure just failed
to evaluate it.

Converting an evidence-quality failure into DEGRADED would be misleading:
the system worked fine; it just couldn't establish the root cause.

M6 enforces this by catching `LLMError` and `SchemaValidationError` and
routing them to DEGRADED, while routing verification failures (REJECTED claims)
to the retry/INSUFFICIENT_EVIDENCE path.

---

## Complete DevTrace Flow

```
Developer provides incident
        ↓
M2: normalize_incident()
        ↓
M3: HybridRetriever.retrieve_as_contracts()
    (semantic + BM25 + signal boosting)
        ↓
M4: NEEDS_INFO gate (deterministic signal check + version ambiguity check)
    ↓ (if triggered) → NEEDS_INFO response
        ↓
M4: filter_applicable() — removes wrong-version evidence
        ↓
M4: DiagnosisGenerator.generate() — LLM call #1
        ↓
M5: verify_diagnosis() — per-claim verification
    Stage 1: deterministic citation validity (no LLM)
    Stage 2: semantic verification (LLM call per claim)
        ↓
M6: root_cause_survived()?
    YES → classify_outcome() → assemble_answer() → ANSWERED_FULL/PARTIAL
    NO → retry_is_allowed()?
             NO → INSUFFICIENT_EVIDENCE
             YES → targeted retry
                    ↓
                M6: build_retry_prompt() — includes verification feedback
                    ↓
                LLM call (retry diagnosis)
                    ↓
                M5: verify_diagnosis() (retry verification)
                    ↓
                M6: root_cause_survived(retry)?
                    YES → classify + assemble → ANSWERED_FULL/PARTIAL
                    NO → INSUFFICIENT_EVIDENCE (STOP — no second retry)
```

---

## Concrete Example: Initial Failure → Retry → Verified Answer

**Incident**: `AUTH_401 started after upgrading from SDK 2.8 to 3.1.`

**M4 Applicable Evidence**:
- `AUTH-002-C01`: "SDK 3.x requires OAuth 2.0 Bearer token in Authorization header."
- `MIGRATION-001-C02`: "Header format changed from X-API-Key to Authorization: Bearer in 3.x."

**Initial diagnosis (M4 LLM call #1)**:
```json
{
  "claims": [
    {
      "role": "root_cause",
      "text": "The authentication configuration may need updating.",
      "evidence_ids": ["AUTH-002-C01"]
    }
  ]
}
```

**M5 verification of initial diagnosis**:
- `root_cause`: `citation_correct=True, sufficient=False → REJECTED`
  - Reason: "Claim is too vague. Evidence states a specific header change but claim doesn't establish the causal link."

**M6 root-cause survival check**: FAILED.

**M6 retry decision**: Applicable evidence exists → retry allowed.

**Retry prompt includes**:
- Same incident, same applicable evidence.
- First diagnosis and its rejection reason.
- Explicit instruction to focus on what the evidence explicitly states.

**Retry diagnosis (M4 LLM call #2)**:
```json
{
  "claims": [
    {
      "role": "root_cause",
      "text": "SDK 3.x changed authentication from X-API-Key header to OAuth 2.0 Bearer token in the Authorization header. Applications using the old API key header will receive AUTH_401.",
      "evidence_ids": ["AUTH-002-C01", "MIGRATION-001-C02"]
    },
    {
      "role": "fix",
      "text": "Update your authentication code to use Authorization: Bearer <token> instead of X-API-Key.",
      "evidence_ids": ["AUTH-002-C01"]
    }
  ]
}
```

**M5 verification of retry diagnosis**:
- `root_cause`: `citation_correct=True, sufficient=True, contradicted=False → VERIFIED`
- `fix`: `citation_correct=True, sufficient=True → VERIFIED`

**M6 retry root-cause survival**: PASSED.

**Final outcome**: `ANSWERED_PARTIAL` (root + fix verified, no explanation claim).

**Final answer text**:
```
Root cause: SDK 3.x changed authentication from X-API-Key header to OAuth 2.0
Bearer token in the Authorization header. Applications using the old API key
header will receive AUTH_401.

Fix: Update your authentication code to use Authorization: Bearer <token>
instead of X-API-Key.
```

The 2.x evidence (`AUTH-001-C01`) was never in the applicable bundle and
never appears in the final answer. The wrong-version evidence problem is solved.

---

## Architectural Drift Check

Before declaring M6 complete, confirmed against the DevTrace architecture:

| Module | Responsibility | Status |
|--------|---------------|--------|
| M3 | Retrieval (Chroma + BM25 + signal boosting) | Unchanged — M6 calls it via M4 pipeline |
| M4 | Applicability + baseline diagnosis | Unchanged — M6 calls it via `run_verified_diagnosis` |
| M5 | Evidence verification | Unchanged — M6 calls it via `verify_diagnosis()` for retry |
| M6 | Retry + final outcome orchestration | ✅ New in this module |
| M7 | Evaluation framework | Not implemented yet |
| M8 | Streamlit UI | Not implemented yet |

Key invariants confirmed:
- ✅ Wrong-version evidence problem remains central (M4 boundary + M5 citation guard + M6 assembly).
- ✅ No new retrieval algorithms introduced.
- ✅ No new applicability/version logic introduced.
- ✅ Retry is exactly one attempt (`max_retries = 1`, enforced by code structure not a counter).
- ✅ Final answers contain only verified claims.
- ✅ No agentic/orchestration framework (LangGraph, agents, etc.) introduced.
- ✅ No LLM knowledge injected as evidence during retry.

---

## Files Created in M6

```
src/orchestration/
├── __init__.py          — Public API (FinalOutcome, TroubleshootingResult, run_troubleshooting)
├── models.py            — FinalOutcome enum, VerifiedAnswer, TroubleshootingResult
├── assembly.py          — root_cause_survived(), classify_outcome(), assemble_answer_text()
├── retry_prompt.py      — build_retry_prompt() — targeted retry with verification feedback
└── orchestrator.py      — run_troubleshooting() — main M6 entry point

tests/unit/
└── test_m6_orchestration.py  — 64 tests covering all spec scenarios

learning/
└── module_6.md          — This document
```
