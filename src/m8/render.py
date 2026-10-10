"""
DevTrace — Module 8: UI rendering helpers.

Pure functions that transform existing M6 result contracts into
Streamlit-renderable data structures.

Design constraints:
  - Contains ZERO reasoning logic.
  - Does NOT call any LLM or retriever.
  - Does NOT re-implement applicability, verification, or retry logic.
  - Renders only what the pipeline result already contains.
  - All functions accept and return standard Python types or Streamlit calls.

Functions:
  render_final_answer       — top-level outcome and answer text
  render_pipeline_trace     — count-level pipeline funnel visualization
  render_evidence_section   — retrieved vs applicable evidence display
  render_verification_section — claim-level verification results
  render_retry_section      — retry status display
  render_needs_info_section — NEEDS_INFO explanation display
  render_degraded_section   — safe DEGRADED failure display
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Only imported for type hints — avoids hard Streamlit dependency in tests
    pass

from src.models.contracts import ApplicabilityResult, DiagnosisClaim, RetrievalResult
from src.models.enums import ClaimRole
from src.orchestration.models import FinalOutcome, TroubleshootingResult, VerifiedAnswer
from src.verification.models import ClaimVerification, VerificationVerdict


# ---------------------------------------------------------------------------
# Outcome helpers
# ---------------------------------------------------------------------------

def outcome_label(outcome: FinalOutcome) -> str:
    """Return a human-readable label for a FinalOutcome."""
    return {
        FinalOutcome.ANSWERED_FULL: "✅ Answered (Full)",
        FinalOutcome.ANSWERED_PARTIAL: "🟡 Answered (Partial)",
        FinalOutcome.INSUFFICIENT_EVIDENCE: "🔍 Insufficient Evidence",
        FinalOutcome.NEEDS_INFO: "❓ More Information Needed",
        FinalOutcome.DEGRADED: "⚠️ System Error",
    }.get(outcome, str(outcome.value))


def outcome_color(outcome: FinalOutcome) -> str:
    """Return a Streamlit status color for a FinalOutcome."""
    return {
        FinalOutcome.ANSWERED_FULL: "success",
        FinalOutcome.ANSWERED_PARTIAL: "warning",
        FinalOutcome.INSUFFICIENT_EVIDENCE: "warning",
        FinalOutcome.NEEDS_INFO: "info",
        FinalOutcome.DEGRADED: "error",
    }.get(outcome, "info")


def claim_role_label(role: ClaimRole) -> str:
    """Return a human-readable label for a ClaimRole."""
    return {
        ClaimRole.ROOT_CAUSE: "Root Cause",
        ClaimRole.FIX: "Fix",
        ClaimRole.EXPLANATION: "Explanation",
    }.get(role, role.value.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Pipeline trace data (pure computation, no Streamlit calls)
# ---------------------------------------------------------------------------

def build_pipeline_trace_data(result: TroubleshootingResult) -> list[dict[str, Any]]:
    """
    Build the pipeline funnel data from a TroubleshootingResult.

    Returns a list of step dicts suitable for UI rendering:
      [{"label": str, "value": str | int, "detail": str | None}, ...]
    """
    steps: list[dict[str, Any]] = []

    # Retrieval
    n_retrieved = len(result.retrieval_results)
    steps.append({
        "label": "Retrieved",
        "value": n_retrieved,
        "detail": f"{n_retrieved} chunk(s) from hybrid retrieval",
    })

    # Applicability
    n_applicable = len(result.applicable_results)
    n_filtered = n_retrieved - n_applicable
    steps.append({
        "label": "Applicable",
        "value": n_applicable,
        "detail": (
            f"{n_applicable} passed applicability filter "
            f"({n_filtered} excluded by version/scope)"
        ),
    })

    # NEEDS_INFO early exit
    if result.final_outcome == FinalOutcome.NEEDS_INFO:
        steps.append({"label": "Outcome", "value": "NEEDS_INFO", "detail": "Missing required information"})
        return steps

    # DEGRADED
    if result.final_outcome == FinalOutcome.DEGRADED and result.initial_diagnosis is None:
        steps.append({"label": "Outcome", "value": "DEGRADED", "detail": "System error prevented completion"})
        return steps

    # Initial diagnosis
    initial_claims = len(result.initial_diagnosis.claims) if result.initial_diagnosis else 0
    steps.append({
        "label": "Diagnosis",
        "value": f"{initial_claims} claim(s)",
        "detail": "Initial structured diagnosis generated",
    })

    # Initial verification
    if result.initial_verification:
        n_verified = len(result.initial_verification.verified_claim_ids)
        n_total = len(result.initial_verification.claim_verifications)
        steps.append({
            "label": "Verified",
            "value": f"{n_verified}/{n_total}",
            "detail": "M5 evidence verification of initial diagnosis",
        })

    # Retry
    if result.retry_attempted:
        retry_claims = len(result.retry_diagnosis.claims) if result.retry_diagnosis else 0
        steps.append({
            "label": "Retry",
            "value": f"{retry_claims} claim(s)",
            "detail": f"Targeted retry triggered: {result.retry_reason or 'root cause not verified'}",
        })
        if result.retry_verification:
            n_rv = len(result.retry_verification.verified_claim_ids)
            n_rt = len(result.retry_verification.claim_verifications)
            steps.append({
                "label": "Retry Verified",
                "value": f"{n_rv}/{n_rt}",
                "detail": "M5 evidence verification of retry diagnosis",
            })
    else:
        steps.append({
            "label": "Retry",
            "value": "Not required",
            "detail": "Initial root cause survived verification",
        })

    # Final outcome
    steps.append({
        "label": "Outcome",
        "value": result.final_outcome.value,
        "detail": outcome_label(result.final_outcome),
    })

    return steps


# ---------------------------------------------------------------------------
# Evidence rendering data
# ---------------------------------------------------------------------------

def build_evidence_display_data(result: TroubleshootingResult) -> list[dict[str, Any]]:
    """
    Build display data for the evidence section.

    Returns a list of evidence item dicts:
      [{
        "chunk_id": str,
        "doc_id": str,
        "score": float,
        "applicable": bool,
        "reason": str,
        "content": str,
        "applies_to": str,
        "topic": str | None,
        "cited_in_final": bool,
      }, ...]
    """
    # Build lookup sets
    applicable_ids: set[str] = {r.chunk_id for r in result.applicable_results}

    # Cited chunk IDs: from the final verified claims
    cited_ids: set[str] = set()
    for claim in result.final_answer.verified_claims:
        cited_ids.update(claim.evidence_ids)

    # Build applicability decision lookup
    app_decision_map: dict[str, Any] = {}
    for decision in result.applicability_decisions:
        if hasattr(decision, "chunk_id"):
            app_decision_map[decision["chunk_id"] if isinstance(decision, dict) else decision.chunk_id] = decision

    items: list[dict[str, Any]] = []
    for rr in result.retrieval_results:
        is_applicable = rr.chunk_id in applicable_ids
        is_cited = rr.chunk_id in cited_ids

        # Get applicability reason
        app_reason = ""
        raw_decision = app_decision_map.get(rr.chunk_id)
        if raw_decision is not None:
            if isinstance(raw_decision, dict):
                app_reason = raw_decision.get("reason", "")
            else:
                app_reason = getattr(raw_decision, "reason", "")

        metadata = rr.metadata or {}
        items.append({
            "chunk_id": rr.chunk_id,
            "doc_id": rr.doc_id,
            "score": rr.score,
            "source": rr.source.value if hasattr(rr.source, "value") else str(rr.source),
            "applicable": is_applicable,
            "applicability_reason": app_reason,
            "cited_in_final": is_cited,
            "content": metadata.get("content", ""),
            "applies_to": metadata.get("applies_to", "*"),
            "topic": metadata.get("topic"),
            "title": metadata.get("title", rr.doc_id),
        })

    # Sort: applicable + cited first, then applicable, then not applicable
    def _sort_key(item: dict) -> tuple:
        return (
            not (item["applicable"] and item["cited_in_final"]),
            not item["applicable"],
            -item["score"],
        )

    return sorted(items, key=_sort_key)


# ---------------------------------------------------------------------------
# Verification rendering data
# ---------------------------------------------------------------------------

def build_verification_display_data(result: TroubleshootingResult) -> list[dict[str, Any]]:
    """
    Build display data for the verification section.

    Returns claim verification records in role order (root_cause, fix, explanation),
    sourced from whichever verification round produced the final answer.
    """
    # Determine which diagnosis and verification to show
    # If retry was attempted and produced a result, prefer retry data for final claims
    final_claims = result.final_answer.verified_claims

    # Build a lookup from claim_id to ClaimVerification
    verification_map: dict[str, Any] = {}

    # Collect from initial verification
    if result.initial_verification:
        for cv in result.initial_verification.claim_verifications:
            verification_map[cv.claim_id] = cv

    # Retry verification takes priority (overwrites same claim_id if re-used)
    if result.retry_verification:
        for cv in result.retry_verification.claim_verifications:
            verification_map[cv.claim_id] = cv

    # Also gather unverified claims from initial diagnosis
    all_claims_map: dict[str, Any] = {}
    if result.initial_diagnosis:
        for claim in result.initial_diagnosis.claims:
            all_claims_map[claim.claim_id] = claim
    if result.retry_diagnosis:
        for claim in result.retry_diagnosis.claims:
            all_claims_map[claim.claim_id] = claim

    items: list[dict[str, Any]] = []
    for claim_id, claim in all_claims_map.items():
        cv = verification_map.get(claim_id)
        verified = claim_id in {c.claim_id for c in final_claims}

        items.append({
            "claim_id": claim_id,
            "role": claim_role_label(claim.role),
            "role_enum": claim.role,
            "text": claim.text,
            "evidence_ids": list(claim.evidence_ids),
            "in_final_answer": verified,
            "verification_available": cv is not None,
            "verdict": cv.verdict.value if cv else "NOT_VERIFIED",
            "citation_validity": cv.citation_validity.value if cv else "UNKNOWN",
            "citation_correct": cv.citation_correct if cv else False,
            "sufficient": cv.sufficient if cv else False,
            "contradicted": cv.contradicted if cv else False,
            "reason": cv.reason if cv else "",
            "supporting_evidence_ids": list(cv.supporting_evidence_ids) if cv else [],
            "contradicting_evidence_ids": list(cv.contradicting_evidence_ids) if cv else [],
        })

    # Sort: root_cause first, then fix, then explanation
    role_order = {ClaimRole.ROOT_CAUSE: 0, ClaimRole.FIX: 1, ClaimRole.EXPLANATION: 2}

    def _sort_key(item: dict) -> int:
        return role_order.get(item["role_enum"], 99)

    return sorted(items, key=_sort_key)


# ---------------------------------------------------------------------------
# Streamlit rendering functions
# ---------------------------------------------------------------------------

def render_final_answer(result: TroubleshootingResult, st: Any) -> None:
    """
    Render the final answer section in Streamlit.

    Displays outcome, answer text, and NEEDS_INFO/DEGRADED explanations.
    All content comes from result.final_answer — no new reasoning.
    """
    fa = result.final_answer
    outcome = result.final_outcome

    label = outcome_label(outcome)
    # Streamlit status containers accept only running/complete/error. Semantic
    # UI colors such as success/info/warning are used by other render helpers
    # but are not valid values for this API.
    status_state = "error" if outcome == FinalOutcome.DEGRADED else "complete"
    st.status(f"**{label}**", state=status_state)

    if outcome in (FinalOutcome.ANSWERED_FULL, FinalOutcome.ANSWERED_PARTIAL):
        if fa.answer_text:
            st.markdown(fa.answer_text)
        else:
            st.info("Answer assembled from verified claims below.")

        if outcome == FinalOutcome.ANSWERED_PARTIAL:
            st.caption(
                "⚠️ Partial answer: the root cause was verified but some supporting "
                "claims (fix or explanation) could not be verified against available evidence."
            )

    elif outcome == FinalOutcome.NEEDS_INFO:
        render_needs_info_section(result, st)

    elif outcome == FinalOutcome.INSUFFICIENT_EVIDENCE:
        st.warning(
            "DevTrace could not verify a root cause for this incident. "
            "The available evidence in the knowledge base does not sufficiently "
            "support any diagnosis claim, even after one targeted retry."
        )

    elif outcome == FinalOutcome.DEGRADED:
        render_degraded_section(result, st)


def render_needs_info_section(result: TroubleshootingResult, st: Any) -> None:
    """
    Render the NEEDS_INFO explanation section.

    Only uses the pipeline-generated reason. Does NOT create new NEEDS_INFO rules.
    """
    st.info("More information is required to complete troubleshooting.")
    reason = result.final_answer.needs_info_reason
    if reason:
        st.markdown(f"**Reason from DevTrace:**\n\n{reason}")
    else:
        st.markdown(
            "The incident description does not contain enough information "
            "to select the correct version-specific documentation. "
            "Please provide the current SDK or API version."
        )


def render_degraded_section(result: TroubleshootingResult, st: Any) -> None:
    """
    Render a safe DEGRADED failure message.

    Never exposes API keys, stack traces, or internal configuration values.
    """
    st.error(
        "DevTrace could not complete the troubleshooting workflow due to a "
        "system error. Please check your configuration and try again."
    )
    # Only show sanitized error detail from the pipeline result — never raw exceptions
    if result.system_errors:
        with st.expander("Developer details (system errors)", expanded=False):
            for i, err in enumerate(result.system_errors, 1):
                # Strip any potential secret patterns before displaying
                safe_err = _sanitize_error(err)
                st.code(f"[{i}] {safe_err}", language=None)


def render_pipeline_trace(result: TroubleshootingResult, st: Any) -> None:
    """
    Render the pipeline trace visualization section.

    Shows the RETRIEVED → APPLICABLE → VERIFIED → OUTCOME funnel
    with real counts from the pipeline result.
    """
    steps = build_pipeline_trace_data(result)

    st.subheader("Pipeline trace")
    st.caption("A compact audit trail from retrieval to the final outcome.")

    cards: list[str] = []
    for index, step in enumerate(steps, 1):
        value = str(step["value"])
        if step["label"] == "Outcome":
            value = value.replace("_", " ").title()
        cards.append(
            '<div class="trace-step">'
            f'<div class="trace-index">STEP {index:02d}</div>'
            f'<div class="trace-label">{html.escape(str(step["label"]))}</div>'
            f'<div class="trace-value">{html.escape(value)}</div>'
            f'<div class="trace-detail">{html.escape(str(step.get("detail") or ""))}</div>'
            '</div>'
        )
    st.markdown(
        '<div class="trace-grid">' + "".join(cards) + '</div>',
        unsafe_allow_html=True,
    )

    # Architectural invariant callout
    n_retrieved = len(result.retrieval_results)
    n_applicable = len(result.applicable_results)
    if n_retrieved > 0 and n_applicable < n_retrieved:
        st.info(
            f"🔍 **Retrieved ≠ Applicable**: {n_retrieved} chunk(s) were retrieved "
            f"by semantic similarity, but only {n_applicable} passed version-specific "
            f"applicability filtering. DevTrace does not diagnose with non-applicable evidence."
        )


def render_evidence_section(result: TroubleshootingResult, st: Any) -> None:
    """
    Render the evidence section showing retrieved vs applicable evidence.

    Displays each evidence chunk with its applicability status and whether
    it was cited in the final verified answer.
    """
    items = build_evidence_display_data(result)

    if not items:
        st.info("No evidence was retrieved for this incident.")
        return

    applicable_items = [i for i in items if i["applicable"]]
    non_applicable_items = [i for i in items if not i["applicable"]]

    # --- Retrieved Evidence ---
    st.subheader("Evidence trail")
    cited_count = sum(1 for item in items if item["cited_in_final"])
    st.caption("Inspect what retrieval found, what M4 allowed, and what the final answer cited.")
    st.markdown(
        '<div class="detail-stats">'
        f'<div class="detail-stat"><span>Retrieved</span><strong>{len(items)}</strong></div>'
        f'<div class="detail-stat"><span>Applicable</span><strong>{len(applicable_items)}</strong></div>'
        f'<div class="detail-stat"><span>Cited in answer</span><strong>{cited_count}</strong></div>'
        f'<div class="detail-stat"><span>Excluded</span><strong>{len(non_applicable_items)}</strong></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # --- Applicable Evidence ---
    if applicable_items:
        st.markdown("#### Applicable evidence")
        st.caption("These chunks passed version and scope filtering. Open a card to inspect its content.")
        for item in applicable_items:
            cited_badge = " · CITED" if item["cited_in_final"] else ""
            with st.expander(
                f"{item['chunk_id']} — {item['title'] or item['doc_id']}{cited_badge}",
                expanded=False,
            ):
                st.markdown(
                    '<div class="detail-stats">'
                    f'<div class="detail-stat"><span>Score</span><strong>{item["score"]:.3f}</strong></div>'
                    f'<div class="detail-stat"><span>Version</span><strong>{html.escape(str(item["applies_to"] or "*"))}</strong></div>'
                    f'<div class="detail-stat"><span>Source</span><strong>{html.escape(str(item["source"]))}</strong></div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                if item["topic"]:
                    st.markdown(f'<div class="meta-line"><strong>Topic</strong> · {html.escape(str(item["topic"]))}</div>', unsafe_allow_html=True)
                if item["applicability_reason"]:
                    st.markdown(f'<div class="meta-line"><strong>Applicability</strong> · {html.escape(str(item["applicability_reason"]))}</div>', unsafe_allow_html=True)
                if item["content"]:
                    st.markdown(
                        f'<div class="evidence-content">{html.escape(str(item["content"]))}</div>',
                        unsafe_allow_html=True,
                    )

    # --- Non-Applicable Evidence ---
    if non_applicable_items:
        with st.expander(
            f"Excluded by applicability filtering ({len(non_applicable_items)})",
            expanded=False,
        ):
            st.caption(
                "These chunks were semantically retrieved but excluded by M4 "
                "applicability filtering (version mismatch or out-of-scope)."
            )
            for item in non_applicable_items:
                with st.expander(
                    f"{item['chunk_id']} — {item['title'] or item['doc_id']}",
                    expanded=False,
                ):
                    st.markdown(
                        '<div class="detail-stats">'
                        f'<div class="detail-stat"><span>Score</span><strong>{item["score"]:.3f}</strong></div>'
                        f'<div class="detail-stat"><span>Version</span><strong>{html.escape(str(item["applies_to"] or "*"))}</strong></div>'
                        f'<div class="detail-stat"><span>Source</span><strong>{html.escape(str(item["source"]))}</strong></div>'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                    if item["applicability_reason"]:
                        st.caption(f"Excluded: {item['applicability_reason']}")


def render_verification_section(result: TroubleshootingResult, st: Any) -> None:
    """
    Render the claim-level M5 verification section.

    Displays the verification verdict, citation validity, support, and
    contradiction status for each diagnosis claim. Uses actual M5 results —
    never performs a new verification call.
    """
    items = build_verification_display_data(result)

    if not items:
        if result.final_outcome == FinalOutcome.NEEDS_INFO:
            st.info("No diagnosis was generated (NEEDS_INFO — pipeline exited early).")
        else:
            st.info("No diagnosis claims available to display.")
        return

    st.subheader("Claim verification")
    st.caption(
        "Each claim generated by the diagnosis step is independently verified "
        "against the applicable evidence bundle."
    )

    retry_note_shown = False
    for item in items:
        verdict = item["verdict"]
        is_verified = verdict == "VERIFIED"
        in_final = item["in_final_answer"]

        icon = "✅" if is_verified else "❌"
        final_badge = " — **In Final Answer**" if in_final else ""
        retry_label = ""
        if result.retry_attempted and not retry_note_shown:
            retry_note_shown = True
            retry_label = " *(after retry)*"

        with st.expander(
            f"{icon} **{item['role']}**{retry_label}{final_badge}",
            expanded=in_final and item["role"] == "Root Cause",
        ):
            st.markdown(
                f'<div class="claim-copy">{html.escape(str(item["text"]))}</div>',
                unsafe_allow_html=True,
            )

            if item["verification_available"]:
                semantic_label = "Supported" if item["citation_correct"] else "Not supported"
                contradicted_label = "⚠️ Yes" if item["contradicted"] else "No"
            else:
                semantic_label = "UNKNOWN"
                contradicted_label = "UNKNOWN"
            st.markdown(
                '<div class="detail-stats">'
                f'<div class="detail-stat"><span>Verdict</span><strong>{html.escape(verdict.replace("_", " ").title())}</strong></div>'
                f'<div class="detail-stat"><span>Citation</span><strong>{html.escape(str(item["citation_validity"]).title())}</strong></div>'
                f'<div class="detail-stat"><span>Semantic support</span><strong>{html.escape(semantic_label)}</strong></div>'
                f'<div class="detail-stat"><span>Contradicted</span><strong>{html.escape(contradicted_label)}</strong></div>'
                '</div>',
                unsafe_allow_html=True,
            )

            if item["reason"]:
                st.markdown(f'<div class="meta-line"><strong>Verifier reasoning</strong> · {html.escape(str(item["reason"]))}</div>', unsafe_allow_html=True)

            if item["evidence_ids"]:
                st.markdown(f'<div class="meta-line"><strong>Cited evidence</strong> · {html.escape(", ".join(item["evidence_ids"]))}</div>', unsafe_allow_html=True)

            if item["supporting_evidence_ids"]:
                st.markdown(f'<div class="meta-line"><strong>Supporting evidence</strong> · {html.escape(", ".join(item["supporting_evidence_ids"]))}</div>', unsafe_allow_html=True)

            if item["contradicting_evidence_ids"]:
                st.warning(
                    f"Contradicting evidence: {', '.join(item['contradicting_evidence_ids'])}"
                )


def render_retry_section(result: TroubleshootingResult, st: Any) -> None:
    """
    Render the M6 retry display section.

    Shows whether a retry was attempted and if the root cause survived.
    All data comes from the pipeline result — never invents retry state.
    """
    st.subheader("Retry Behaviour")

    if not result.retry_attempted:
        if result.final_outcome in (
            FinalOutcome.ANSWERED_FULL,
            FinalOutcome.ANSWERED_PARTIAL,
        ):
            st.success("✅ Retry not required — root cause was verified on first attempt.")
        elif result.final_outcome == FinalOutcome.DEGRADED:
            st.error(
                "❌ Retry was not attempted because a system error interrupted "
                "the workflow before retry evaluation could complete."
            )
        elif result.final_outcome == FinalOutcome.NEEDS_INFO:
            st.info("ℹ️ Retry was not attempted because more incident information is required.")
        else:
            st.info("ℹ️ No retry was attempted; no root cause was verified.")
        return

    st.warning(f"⚡ One targeted retry was triggered.")
    if result.retry_reason:
        st.caption(f"Reason: {result.retry_reason}")

    if result.retry_verification:
        retry_verified = result.retry_verification.verified_claim_ids
        if retry_verified:
            st.success(
                f"✅ Retry succeeded — {len(retry_verified)} claim(s) verified "
                "after retry."
            )
        else:
            st.error("❌ Retry did not recover a verified root cause.")
    else:
        st.error("❌ Retry verification failed (system error during retry).")

    st.caption(
        "DevTrace enforces a maximum of one retry. "
        "This invariant is enforced by the M6 orchestrator, not the UI."
    )


# ---------------------------------------------------------------------------
# Safety helper
# ---------------------------------------------------------------------------

def _sanitize_error(error_text: str) -> str:
    """
    Strip potential secret patterns from error messages before UI display.

    This is a defensive measure only — secrets should never reach error messages
    in the first place if M1–M6 are implemented correctly.
    """
    import re  # noqa: PLC0415
    # Mask anything that looks like an API key (long alphanumeric strings)
    sanitized = re.sub(r"[A-Za-z0-9_\-]{32,}", "[REDACTED]", error_text)
    return sanitized
