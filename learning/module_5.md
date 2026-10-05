# DevTrace — Module 5: Evidence Verification

## What is Module 5?

Module 5 is the **independent evidence verifier**.

After M4 generates a structured diagnosis, M5 asks a fundamentally different question:

> "Is the diagnosis actually *supported* by the evidence?"

The key architectural insight:

```
RETRIEVED ≠ APPLICABLE ≠ SUPPORTED
```

M3 retrieval finds *semantically similar* documents.
M4 applicability ensures only *version-correct* documents reach the generator.
M5 verification ensures each *generated claim* is actually *grounded* in the applicable evidence.

---

## Why Retrieval and Generation Are Not Enough

Consider this scenario:

**Incident**: `AUTH_401 after upgrading SDK from 2.8 to 3.1.`

**M3 retrieves** (applicable after M4):
- `AUTH-002-C01` — "SDK 3.x uses OAuth Bearer tokens. The token must be refreshed every 60 minutes."
- `AUTH-003-C01` — "AUTH_403 errors occur when OAuth scope is insufficient."

**M4 generates this diagnosis**:
```json
{
  "claims": [
    {
      "role": "root_cause",
      "text": "The API key format changed in 3.x.",
      "evidence_ids": ["AUTH-002-C01"]
    }
  ]
}
```

The citation is **valid** — `AUTH-002-C01` exists in the applicable bundle.
But the claim is **incorrect** — `AUTH-002-C01` says nothing about API key format changes.
The generator hallucinated a plausible-sounding but unsupported claim.

Without M5, this incorrect diagnosis would be returned to the developer as if it were evidence-grounded truth.

---

## The Four Verification Checks

M5 evaluates each claim on four independent dimensions:

### 1. Citation Validity (Deterministic)

**Question**: Are the cited evidence IDs actually present in the M4 applicable bundle?

**Why deterministic**: This is a pure set-membership check. No LLM needed.
- `evidence_id ∈ applicable_ids` → `citation_valid = VALID`
- `evidence_id ∉ applicable_ids` → `citation_valid = INVALID`

**What INVALID catches**:
- Hallucinated chunk IDs that don't exist anywhere
- IDs from NOT_APPLICABLE chunks (excluded by M4 but perhaps remembered from training)
- Typos or malformed IDs

If citation validity is INVALID or EMPTY, the claim is **immediately rejected** — no LLM call is wasted.

### 2. Citation Correctness (Semantic)

**Question**: Does the cited evidence actually *support* the claim?

A citation can be **valid** (the ID exists) but **incorrect** (the content doesn't support the claim).

```
AUTH-002-C01 exists in the applicable bundle: citation_valid = VALID
AUTH-002-C01 content: "OAuth Bearer token must be refreshed every 60 minutes."
Claim: "The API key format changed in 3.x."
→ citation_correct = False
```

Citation correctness requires semantic reasoning and is evaluated by an independent LLM call.

### 3. Sufficiency (Semantic)

**Question**: Is the evidence strong enough to *justify* the claim?

A claim may be directionally correct but weakly supported — the evidence merely *relates* to the claim rather than *establishing* it.

```
Evidence: "Authentication may fail in some SDK 3.x environments."
Claim: "The root cause is expired OAuth token."
→ sufficient = False (too vague to establish this specific root cause)
```

Sufficiency is distinct from correctness. An explanation claim might be citation-correct (the evidence mentions the topic) but insufficient (the evidence doesn't establish the claim strongly enough).

### 4. Contradiction (Semantic)

**Question**: Does any applicable evidence *conflict* with the claim?

Contradiction is checked across the entire applicable evidence bundle — not just the cited chunks.

```
Claim: "SDK 3.x does not require OAuth. Use API key headers."
Evidence (AUTH-002-C01): "SDK 3.x requires OAuth Bearer tokens. API key auth was removed in 3.0."
→ contradicted = True
```

Important: M4 has already resolved version applicability. M5 only receives applicable evidence.
If a 2.x document contradicts a 3.x document, but the incident is 3.x, the 2.x document never reaches M5.
**M5 does not implement version logic. M4's boundary enforces it.**

---

## The Verdict Rule

```
VERIFIED if and only if:
    citation_validity == VALID
    AND citation_correct == True
    AND sufficient == True
    AND contradicted == False

Otherwise: REJECTED
```

This is intentionally strict. A claim that is mostly right but has one weak dimension gets REJECTED.
M6 will later decide what to do with a mix of VERIFIED and REJECTED claims.

---

## Why Verification is Independent of Generation

The diagnosis generator and evidence verifier are **separate LLM calls** to independent instances.

This is deliberate:

**Generator bias**: The generator was prompted to diagnose. It may unconsciously construct reasoning that sounds supported even when the evidence is weak.

**Verifier independence**: The verifier is prompted to be skeptical. It is told: "Does this evidence actually support this claim?" — not "Given this evidence, explain the diagnosis."

The two-LLM pattern prevents the generator from being judge of its own output.

---

## Short-Circuit for Invalid Citations

When `citation_validity == INVALID` or `citation_validity == EMPTY`:

1. The claim is immediately `REJECTED`
2. No LLM call is made for semantic verification

This is both correct and efficient:
- Correct: a hallucinated citation can never support a claim
- Efficient: saves one LLM call per invalid citation

---

## Why Applicability Remains M4's Responsibility

M5 does **not** check version ranges. It does not know about `applies_to`.

The boundary is clean:

| Module | Responsibility |
|--------|---------------|
| M4     | Which chunks are applicable to this incident's version? |
| M5     | Does the applicable evidence support the generated claims? |

M5 trusts the `applicable_results` it receives. If a 2.x chunk is in `applicable_results`, M5 will treat it as valid evidence. This is correct — M4's job is to ensure that never happens.

---

## Why Retry is Deferred to M6

M5 returns verdicts. It does not act on them.

If a root_cause claim is REJECTED:
- M5 records: `verdict=REJECTED, citation_validity=INVALID, reason="..."`
- M5 returns the DiagnosisVerification
- M5 stops

M6 will decide:
- Should we retry with a more targeted prompt?
- Is the surviving evidence sufficient to answer partially?
- What is the final SystemOutcome (ANSWERED, INSUFFICIENT_EVIDENCE, DEGRADED)?

Deferring this to M6 keeps M5's responsibility narrow and testable.

---

## Examples

### Example 1: VERIFIED claim

```
Incident: AUTH_401 on SDK 3.1. Bearer token rejected after upgrade.
Claim (root_cause): "SDK 3.x requires OAuth Bearer token. Token expires every 60 minutes."
Citation: AUTH-002-C01

Evidence (AUTH-002-C01):
  "In SDK 3.x, all API calls require an Authorization: Bearer <token> header.
   Tokens expire after 60 minutes and must be refreshed using /auth/token."

Verification:
  citation_valid = VALID      (AUTH-002-C01 is in the applicable bundle)
  citation_correct = True     (evidence directly states the Bearer requirement)
  sufficient = True           (evidence explains cause and behavior clearly)
  contradicted = False        (no other applicable evidence conflicts)
  verdict = VERIFIED
```

### Example 2: REJECTED — Invalid citation

```
Claim (root_cause): "SDK 3.x changed the HMAC signing algorithm."
Citation: AUTH-HMAC-999

Evidence bundle (applicable): [AUTH-002-C01, AUTH-003-C01, SDK-001-C01]
AUTH-HMAC-999 is NOT in the applicable bundle.

Verification:
  citation_valid = INVALID    (AUTH-HMAC-999 does not exist in the bundle)
  citation_correct = False    (short-circuit — citation is invalid)
  sufficient = False          (short-circuit)
  contradicted = False        (short-circuit)
  verdict = REJECTED
  reason = "Claim cites AUTH-HMAC-999 which is not present in the applicable evidence bundle."
```

### Example 3: REJECTED — Valid citation, unsupported claim

```
Claim (root_cause): "The API key format changed from 32-char to 64-char in SDK 3.x."
Citation: AUTH-002-C01

Evidence (AUTH-002-C01):
  "In SDK 3.x, all API calls require an Authorization: Bearer <token> header.
   Tokens expire after 60 minutes and must be refreshed using /auth/token."

Verification:
  citation_valid = VALID      (AUTH-002-C01 is in the bundle)
  citation_correct = False    (evidence says nothing about API key format changes)
  sufficient = False
  contradicted = False
  verdict = REJECTED
  reason = "AUTH-002-C01 describes OAuth Bearer token requirements, not API key format changes."
```

### Example 4: Mixed outcome

```
Claims:
  root_cause → VERIFIED  (well-cited, correct, sufficient)
  fix        → VERIFIED  (well-cited, specific fix steps)
  explanation → REJECTED  (cites a chunk that says "may sometimes" — insufficient)

DiagnosisVerification.all_verified = False
DiagnosisVerification.verified_claim_ids = [root_cause.claim_id, fix.claim_id]
DiagnosisVerification.rejected_claim_ids = [explanation.claim_id]
```

M6 receives this and decides: root_cause and fix are solid. The explanation is weak. A partial answer is possible.

---

## File Structure

```
src/
    verification/
        __init__.py     — public API: verify_diagnosis, ClaimVerification, DiagnosisVerification
        models.py       — ClaimVerification, DiagnosisVerification, CitationValidity, VerificationVerdict
        citation.py     — deterministic citation validity (set-membership, no LLM)
        prompts.py      — verification prompt builder
        verifier.py     — _verify_claim, verify_diagnosis (main entry point)

    pipeline/
        verified.py     — run_verified_diagnosis (M4 + M5 combined pipeline)

tests/unit/
    test_m5_verification.py — ~48 deterministic offline tests
```

---

## Running the Tests

```powershell
# All tests (M1–M5)
.\\venv\\Scripts\\python.exe -m pytest

# M5 tests only
.\\venv\\Scripts\\python.exe -m pytest tests/unit/test_m5_verification.py -v

# No Gemini API key required — all tests use FakeLLMClient
```

---

## What M6 Receives from M5

| M5 Output | M6 Usage |
|-----------|----------|
| `DiagnosisVerification.claim_verifications` | Which claims survived? |
| `ClaimVerification.verdict` | VERIFIED/REJECTED per claim |
| `ClaimVerification.citation_validity` | Was the citation hallucinated? |
| `ClaimVerification.supporting_evidence_ids` | Which chunks support surviving claims? |
| `ClaimVerification.contradicting_evidence_ids` | What conflicts exist? |
| `ClaimVerification.reason` | Explanation for M6 retry prompt |
| `DiagnosisVerification.verified_claim_ids` | IDs to assemble into final answer |
| `DiagnosisVerification.rejected_claim_ids` | IDs to use for targeted retry (M6 decides) |
