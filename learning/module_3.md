# Module 3 — Hybrid Retrieval: Chroma + BM25 + Signal Boosting

## What problem does this module solve?

DevTrace needs to find the most relevant documentation chunks before it can diagnose a developer's incident. But finding the right evidence is harder than it looks.

Imagine a developer reports:

> "AUTH_401 errors started after upgrading the SDK from 2.8 to 3.1."

The DevCore corpus contains documentation about AUTH_401 for **both** SDK 2.x *and* SDK 3.x. A naive search system might only return one or the other. But DevTrace needs to see both — because M4 (Applicability) will decide which one actually applies, and M5 (Verification) will check that the chosen evidence supports the diagnosis.

Module 3's job is simple:

> **Retrieve a strong, broad candidate set of evidence chunks. Do not pre-judge which version applies. Let later modules do that.**

---

## Why is dense retrieval alone insufficient?

Dense retrieval works by converting text into a vector of numbers (called an **embedding**) that captures the *meaning* of the text. Two chunks with similar meanings get similar vectors, so a query like "authentication failure" would retrieve documents about "unauthorized access" even if they use different words.

But dense retrieval has a blind spot: **exact technical identifiers**.

Consider the error code `AUTH_401`. In a short 384-dimension embedding vector, this specific identifier competes with hundreds of other tokens. A document about `AUTH_403` might score almost as high as one about `AUTH_401`, simply because their overall *topics* (authentication errors) are similar.

In developer troubleshooting, exact technical identifiers matter enormously. `AUTH_401` and `AUTH_403` have completely different causes and fixes. Dense retrieval can accidentally conflate them.

---

## Why does BM25 matter for technical troubleshooting?

**BM25** (Best Match 25) is a classic *sparse* retrieval algorithm. It does not use embeddings. Instead, it counts how often query terms appear in documents, weighted by how rare those terms are in the corpus.

BM25 is perfect for:

- **Error codes**: `AUTH_401`, `RATE_429`, `ERR_TIMEOUT` — exact token matches.
- **Technical identifiers**: `X-DevCore-Signature-256`, `HMAC-SHA256`, `OAuth`.
- **Method names**: `DevCoreClient.authenticate()`, `hmac.compare_digest`.

When a developer says "I'm getting AUTH_401", BM25 finds every document that mentions exactly `AUTH_401`. Semantic embeddings might retrieve related-but-wrong documents; BM25 finds exact matches.

### Tokenizer design matters

Standard tokenisers split on underscores and hyphens, destroying identifiers like `AUTH_401` → `AUTH`, `401`. DevTrace uses a custom tokeniser that:

1. Detects all-caps identifiers (`AUTH_401`, `RATE_429`) and keeps them intact.
2. Detects mixed-case technical tokens (`OAuth`, `HMAC-SHA256`) and keeps them intact.
3. Falls back to normal word splitting for prose.

This ensures BM25 can actually match on the signals that matter most.

---

## Why hybrid retrieval?

Dense-only retrieval misses exact technical signals.  
BM25-only retrieval misses semantic meaning (synonyms, paraphrases).

**Hybrid retrieval** combines both:

```
Dense score  × 0.6   (weighted by semantic relevance)
BM25 score   × 0.4   (weighted by lexical relevance, normalised per-query)
Signal boost × up to 0.30   (additive, from incident signals)
─────────────────────────────
Fused score  (clamped to [0, 1])
```

The weights are configurable. M7 (Evaluation) will tune them based on benchmark results.

---

## What are embeddings doing here?

An **embedding** is a mathematical representation of text as a vector of numbers. Texts that mean similar things get vectors that are close together in space.

DevTrace uses `all-MiniLM-L6-v2` — a small, fast sentence-transformer model. It:

- Runs entirely **locally** (no API calls, no cost).
- Produces 384-dimensional vectors.
- Is fast enough for interactive use on a laptop.

When we build the Chroma index, we embed every corpus chunk once and store the vectors. At query time, we embed the incident description and compare it to every stored vector using **cosine similarity** (how closely aligned the two vectors are). Similarity of 1.0 = identical meaning; 0.0 = completely unrelated.

---

## What does Chroma do?

**Chroma** is a local vector database. It:

1. Stores the embedding vectors for all corpus chunks.
2. Allows fast nearest-neighbour search — given a query vector, find the `k` most similar stored vectors.
3. Stores chunk metadata (doc_id, applies_to, topic) alongside each vector.
4. Persists the index to disk so we don't re-embed every startup.
5. Supports `upsert()` — inserting again with the same ID updates rather than duplicating.

Chroma runs entirely locally. No cloud service, no authentication, no network.

---

## How does signal boosting work?

After retrieving candidates from dense and BM25, DevTrace applies a small additive **signal boost** based on how well each chunk matches the incident's extracted signals.

Signal boosts are:

| Signal | Boost |
|---|---|
| Exact error code match (in content or metadata) | +0.20 |
| Technical term match (oauth, bearer, webhook, etc.) | +0.05 |
| Version range match (chunk's `applies_to` overlaps incident version) | +0.08 |
| **Total cap** | **0.30** |

**Example**: For an incident with `AUTH_401` and current_version `3.1`:

- A chunk about `AUTH_401` in SDK 3.x: `error_code_boost=0.20` + `version_boost=0.08` = **0.28** boost.
- A general chunk about networking with no signals: **0.00** boost.

The chunk with the signal match gets pushed higher in the ranking. But the general chunk is still in the results — boosting is not filtering.

---

## Why don't we filter by version yet?

This is one of the most important design decisions in DevTrace.

Consider: "AUTH_401 errors started after upgrading from SDK 2.8 to 3.1."

If M3 filtered out 2.x documents, it would only retrieve the 3.x auth guide. But the 2.x guide is crucial — it explains the **old** authentication model that the developer was using, which helps M4 reason about whether the old pattern is the root cause.

The correct trace looks like:

```
Retrieved by M3:
    AUTH-001 (2.x auth guide)  ← contains old API key format
    AUTH-002 (3.x auth guide)  ← contains new OAuth/Bearer format
    SDK-001  (migration guide) ← explains what changed

Applicability decision by M4:
    AUTH-001 → rejected (2.x doc, incident is on 3.x)
    AUTH-002 → applicable
    SDK-001  → applicable (version-agnostic migration guide)
```

If M3 had filtered AUTH-001, M4 would never see it. But AUTH-001's existence in the results provides important diagnostic context — it tells M4 that the developer was likely using the old format.

**Version matching in M3 is a ranking boost, not a filter.**

---

## How is the final ranking produced?

1. **Dense hits** from Chroma are retrieved (up to `dense_top_k`).
2. **BM25 hits** are retrieved (up to `bm25_top_k`).
3. Results are **merged by chunk_id** — a chunk that appeared in both channels is kept once, with both scores recorded.
4. BM25 scores are **normalised** to [0, 1] by dividing by the maximum BM25 score in the batch.
5. **Signal boosts** are computed for each candidate.
6. Scores are **fused**: `dense_weight × dense_score + bm25_weight × norm_bm25 + signal_boost`.
7. Results are **sorted descending** by fused score, with chunk_id as a stable tie-breaker.
8. Slice to `final_top_k`.

The ranking is **completely deterministic** — same corpus, same incident, same configuration → identical ordering every time.

---

## What happens when dense and BM25 disagree?

Dense retrieval retrieves by semantic meaning; BM25 by lexical overlap.

Scenario: A document talks about authentication token expiry in casual language with no explicit error codes. Dense finds it (semantic match). BM25 misses it (no exact token match).

Result: The document appears with `source=SEMANTIC`, `bm25_score=None`. Its fused score only counts the dense component. It may still rank highly if the dense score is strong.

Scenario: A document contains `AUTH_401` 5 times but is for an unrelated product. BM25 ranks it high (exact token frequency). Dense may rank it lower (meaning diverges from the query).

Result: BM25's contribution is weighted at 0.4; dense at 0.6. The fusion naturally down-weights the BM25-only match if semantic relevance is low. M4 will also apply applicability logic to catch any mismatch.

---

## What does M4 receive from M3?

M4 receives a list of `RetrievalResult` objects (the M1 contract), each containing:

- `chunk_id` and `doc_id` — for tracing back to the source document.
- `score` — the final fused relevance score ∈ [0, 1].
- `source` — `SEMANTIC`, `KEYWORD`, or `HYBRID`.
- `metadata` — includes:
  - `content` — the actual chunk text.
  - `applies_to` — the version range string (e.g. `>=3.0,<4.0`).
  - `topic` — e.g. `authentication`, `webhooks`.
  - `dense_score`, `bm25_raw_score`, `bm25_norm_score` — channel scores.
  - `error_code_boost`, `technical_term_boost`, `version_boost`, `signal_boost_total` — why this chunk ranked here.

M4 uses `applies_to` and the incident's `current_version` to make applicability decisions. M3 does not make those decisions.

---

## Files created in Module 3

```
src/retrieval/
    __init__.py       Public surface (HybridRetriever, build_index)
    config.py         RetrievalConfig with documented scoring formula
    index.py          Chroma index construction (build_index, get_or_create_collection)
    dense.py          SentenceTransformer + Chroma query (DenseRetriever)
    sparse.py         BM25 index + custom tokeniser (BM25Retriever)
    boosting.py       Signal boost computation (compute_boost)
    hybrid.py         Fusion, deduplication, ranking (HybridRetriever)

tests/unit/
    test_m3_retrieval.py  ~60 M3 unit tests
```

---

## Key properties of the M3 retrieval system

| Property | How it's achieved |
|---|---|
| Deterministic | Fixed weights, stable sort (chunk_id tie-break), no randomness |
| Offline | Local sentence-transformer, local Chroma, local BM25 |
| Transparent | Every score component is logged and stored in metadata |
| No version filtering | Boost only; applicability is M4's job |
| No reranker | Dense + BM25 + signal boosts is the complete system |
| No agents | Pure Python orchestration, no LangGraph, no MCP |
| Idempotent indexing | Chroma upsert; BM25 rebuilt per retriever instance |
| Configurable | All weights, top-k values, model name are in RetrievalConfig |
