"""
DevTrace — Module 4: Optional Gemini live smoke test.

Usage:
    Set GEMINI_API_KEY in your .env file or environment, then run:

        .\venv\Scripts\python.exe scripts\smoke_test_m4.py

This script is NOT part of the normal pytest suite.
It demonstrates the full M4 pipeline with a real Gemini API call.

The pipeline steps shown:
    1. Normalize incident
    2. M3 hybrid retrieval
    3. NEEDS_INFO gate
    4. Applicability filtering
    5. Gemini diagnosis generation

Output is printed to stdout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on the path when run directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings
from src.ingestion.loader import load_corpus_and_chunks
from src.llm.gemini import GeminiClient
from src.pipeline.baseline import run_baseline_diagnosis
from src.retrieval.hybrid import HybridRetriever


# ---------------------------------------------------------------------------
# Smoke test incident — a clear version-conflict case
# ---------------------------------------------------------------------------

INCIDENT = {
    "description": (
        "AUTH_401 on all API requests after upgrading from SDK 2.8 to SDK 3.1. "
        "We were sending 'Authorization: ApiKey <key>' but now every request fails."
    ),
    "current_version": "3.1",
    "previous_version": "2.8",
    "error_codes": ["AUTH_401"],
    "product": "DevCore SDK",
}


def main() -> None:
    print("=" * 60)
    print("DevTrace M4 — Live Gemini Smoke Test")
    print("=" * 60)

    # Check API key.
    if not settings.gemini_api_key:
        print(
            "\n[ERROR] GEMINI_API_KEY is not set.\n"
            "Copy .env.example to .env and provide your API key, then retry."
        )
        sys.exit(1)

    print(f"\nGemini model: {settings.gemini_model}")
    print(f"Temperature: {settings.gemini_temperature}")

    # Load corpus + build retriever.
    print("\nLoading DevCore corpus and building retrieval index...")
    _, chunks = load_corpus_and_chunks()
    retriever = HybridRetriever(chunks)
    retriever.load()
    retriever.build_vector_index()
    print(f"Index built: {len(chunks)} chunks.")

    # Build real Gemini client.
    llm_client = GeminiClient()

    # Run pipeline.
    print(f"\nIncident:\n{json.dumps(INCIDENT, indent=2)}")
    print("\nRunning baseline pipeline...")

    result = run_baseline_diagnosis(
        description=INCIDENT["description"],
        current_version=INCIDENT["current_version"],
        previous_version=INCIDENT["previous_version"],
        error_codes=INCIDENT["error_codes"],
        product=INCIDENT["product"],
        retriever=retriever,
        llm_client=llm_client,
    )

    print("\n" + "-" * 60)
    print(f"Pipeline outcome: {result.outcome}")

    if result.needs_info and result.needs_info.triggered:
        print(f"\nNEEDS_INFO: {result.needs_info.reason}")
        print(f"Hint: {result.needs_info.unblock_hint}")
        return

    print(f"\nRetrieval results: {len(result.retrieval_results)}")
    print(f"Applicable results: {len(result.applicable_results)}")
    print(f"Excluded by applicability: {len(result.retrieval_results) - len(result.applicable_results)}")

    print("\nApplicability decisions:")
    for dec in result.applicability_decisions:
        status = "✓ APPLICABLE" if dec.applicable else "✗ NOT APPLICABLE"
        print(f"  {status}: {dec.doc_id} ({dec.document_range!r})")
        print(f"    Reason: {dec.reason[:80]}...")

    if result.error:
        print(f"\n[DEGRADED] Error: {result.error}")
        return

    if result.diagnosis:
        print("\nGenerated diagnosis claims:")
        for claim in result.diagnosis.claims:
            print(f"\n  [{claim.role.value.upper()}]")
            print(f"  {claim.text}")
            if claim.evidence_ids:
                print(f"  Evidence: {', '.join(claim.evidence_ids)}")

    print("\n" + "=" * 60)
    print("Smoke test complete.")
    print(
        "\nNote: The diagnosis above is UNVERIFIED (M5 will check citations)."
        "\nDo not trust these claims until M5 verification has run."
    )


if __name__ == "__main__":
    main()
