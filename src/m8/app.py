"""
DevTrace — Module 8: Streamlit Main Application Interface.

Provides an interactive web demonstration for DevTrace:
  1. Live Incident Diagnostics — Run incidents through M6 orchestration pipeline
  2. Evaluation & Benchmark Dashboard — Compare DevTrace against Baselines A & B
  3. System Architecture & Invariants — Visual breakdown of the 8-module pipeline
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import streamlit as st

from src.config import settings
from src.m7.models import EvaluationReport
from src.m8.pipeline import build_llm_client, build_retriever, run_pipeline
from src.m8.render import (
    render_degraded_section,
    render_evidence_section,
    render_final_answer,
    render_needs_info_section,
    render_pipeline_trace,
    render_retry_section,
    render_verification_section,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DevTrace — Evidence-Grounded Developer Troubleshooting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Presentation-only design system. Pipeline behavior remains in M1-M6.
st.markdown(
    """
    <style>
    :root {
        --dt-primary: #5ee7a8;
        --dt-blue: #65a7ff;
        --dt-warning: #f4c66a;
        --dt-danger: #ff7b86;
        --dt-radius: 14px;
    }
    .block-container {
        max-width: 1240px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid color-mix(in srgb, var(--text-color) 12%, transparent);
    }
    .main-title {
        font-size: clamp(2rem, 4vw, 3.15rem);
        font-weight: 760;
        letter-spacing: -0.045em;
        line-height: 1.05;
        margin: 0;
    }
    .sub-title {
        font-size: 1.05rem;
        color: color-mix(in srgb, var(--text-color) 68%, transparent);
        line-height: 1.65;
        max-width: 780px;
        margin: .75rem 0 1.4rem;
    }
    .invariant-banner {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: .55rem;
        background: color-mix(in srgb, var(--dt-primary) 8%, var(--secondary-background-color));
        border: 1px solid color-mix(in srgb, var(--dt-primary) 28%, transparent);
        padding: .8rem 1rem;
        border-radius: 12px;
        font-size: .82rem;
        font-weight: 650;
        margin-bottom: 1.4rem;
    }
    .invariant-banner .flow { color: var(--dt-primary); letter-spacing: .04em; }
    .sidebar-brand { display:flex; align-items:center; gap:.75rem; margin:.4rem 0 1.25rem; }
    .sidebar-mark {
        display:grid; place-items:center; width:42px; height:42px; border-radius:12px;
        background:linear-gradient(145deg,var(--dt-primary),#54a8ff); color:#07120d;
        font-size:1.25rem; font-weight:900; box-shadow:0 10px 28px rgba(37,199,122,.18);
    }
    .sidebar-brand strong { display:block; font-size:1.12rem; }
    .sidebar-brand small { color:color-mix(in srgb,var(--text-color) 58%,transparent); font-size:.75rem; }
    .connection-pill {
        display:flex; align-items:center; gap:.5rem; padding:.7rem .8rem; border-radius:10px;
        background:color-mix(in srgb,var(--dt-primary) 10%,var(--secondary-background-color));
        border:1px solid color-mix(in srgb,var(--dt-primary) 24%,transparent); font-size:.82rem;
    }
    .connection-dot { width:8px; height:8px; border-radius:50%; background:var(--dt-primary); }
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap:.35rem; border-bottom:1px solid color-mix(in srgb,var(--text-color) 12%,transparent);
    }
    [data-testid="stTabs"] button[role="tab"] { padding:.8rem 1rem; border-radius:10px 10px 0 0; }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color:var(--dt-primary); }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color:var(--dt-primary); }
    button[kind="primary"] {
        background:linear-gradient(135deg,#28ce80,#58dda2) !important;
        border-color:#28ce80 !important; color:#07120d !important; font-weight:750 !important;
    }
    button[kind="primary"]:hover { filter:brightness(1.06); }
    [data-testid="stForm"], [data-testid="stExpander"], [data-testid="stStatusWidget"] {
        border-radius:var(--dt-radius); border-color:color-mix(in srgb,var(--text-color) 14%,transparent);
    }
    [data-testid="stForm"] {
        padding:1.15rem 1.25rem 1.3rem;
        background:color-mix(in srgb,var(--secondary-background-color) 70%,transparent);
        border:1px solid color-mix(in srgb,var(--text-color) 14%,transparent) !important;
    }
    .section-kicker {
        color:var(--dt-primary); font-weight:750; font-size:.74rem;
        letter-spacing:.12em; text-transform:uppercase; margin-bottom:.25rem;
    }
    .section-intro {
        color:color-mix(in srgb,var(--text-color) 62%,transparent);
        margin:-.35rem 0 1.1rem; max-width:760px;
    }
    .architecture-flow { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.75rem; margin:1rem 0 1.5rem; }
    .architecture-node, .pillar-card, .system-card {
        background:color-mix(in srgb,var(--secondary-background-color) 76%,transparent);
        border:1px solid color-mix(in srgb,var(--text-color) 13%,transparent);
        border-radius:13px; padding:1rem;
    }
    .architecture-node strong, .pillar-card strong, .system-card strong { display:block; margin-bottom:.3rem; }
    .architecture-node span, .pillar-card span, .system-card span {
        color:color-mix(in srgb,var(--text-color) 62%,transparent); font-size:.84rem; line-height:1.5;
    }
    .architecture-node em { display:inline-block; color:var(--dt-primary); font-style:normal; font-size:.72rem; font-weight:750; margin-bottom:.45rem; }
    .pillar-grid, .system-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:.8rem; }
    .system-grid { grid-template-columns:repeat(3,minmax(0,1fr)); margin:1rem 0; }
    .trace-grid {
        display:grid; grid-template-columns:repeat(auto-fit,minmax(145px,1fr));
        gap:.7rem; margin:1rem 0 .7rem;
    }
    .trace-step {
        min-width:0; min-height:150px; padding:1rem;
        background:color-mix(in srgb,var(--secondary-background-color) 74%,transparent);
        border:1px solid color-mix(in srgb,var(--text-color) 13%,transparent);
        border-radius:13px;
    }
    .trace-index { color:var(--dt-primary); font-size:.68rem; font-weight:800; letter-spacing:.12em; }
    .trace-label { margin-top:.45rem; color:color-mix(in srgb,var(--text-color) 66%,transparent); font-size:.76rem; font-weight:700; text-transform:uppercase; }
    .trace-value {
        margin:.3rem 0 .45rem; font-size:clamp(1.2rem,2vw,1.65rem); font-weight:720;
        line-height:1.15; letter-spacing:-.025em; overflow-wrap:anywhere;
    }
    .trace-detail { color:color-mix(in srgb,var(--text-color) 58%,transparent); font-size:.77rem; line-height:1.45; }
    .detail-stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(125px,1fr)); gap:.55rem; margin:.85rem 0; }
    .detail-stat {
        padding:.75rem .8rem; border-radius:10px;
        background:color-mix(in srgb,var(--secondary-background-color) 70%,transparent);
        border:1px solid color-mix(in srgb,var(--text-color) 11%,transparent);
    }
    .detail-stat span { display:block; color:color-mix(in srgb,var(--text-color) 57%,transparent); font-size:.69rem; text-transform:uppercase; letter-spacing:.06em; }
    .detail-stat strong { display:block; margin-top:.24rem; font-size:.95rem; overflow-wrap:anywhere; }
    .claim-copy { font-size:1rem; line-height:1.65; margin:.3rem 0 1rem; }
    .meta-line { color:color-mix(in srgb,var(--text-color) 61%,transparent); font-size:.82rem; line-height:1.55; margin:.3rem 0; }
    .meta-line strong { color:color-mix(in srgb,var(--text-color) 82%,transparent); }
    .evidence-content {
        margin-top:.8rem; max-height:230px; overflow:auto; white-space:pre-wrap;
        padding:.9rem 1rem; border-radius:10px; font: .8rem/1.55 ui-monospace,SFMono-Regular,Consolas,monospace;
        background:color-mix(in srgb,var(--background-color) 72%,#000 8%);
        border:1px solid color-mix(in srgb,var(--text-color) 10%,transparent);
        color:color-mix(in srgb,var(--text-color) 78%,transparent);
    }
    .result-copy { font-size:1.02rem; line-height:1.72; }
    hr { border-color:color-mix(in srgb,var(--text-color) 10%,transparent) !important; }
    @media (max-width:900px) { .architecture-flow,.system-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }
    @media (max-width:640px) {
        .block-container { padding:1rem .85rem 3rem; }
        .architecture-flow,.pillar-grid,.system-grid { grid-template-columns:1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached Data Loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading DevTrace Corpus & Hybrid Retriever...")
def get_cached_retriever() -> Any:
    """Load and cache the HybridRetriever instance."""
    return build_retriever()


@st.cache_data(show_spinner=False)
def load_preset_dataset() -> list[dict[str, Any]]:
    """Load benchmark preset evaluation cases from disk."""
    eval_path = Path("data/eval/eval_dataset.json")
    if eval_path.exists():
        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Failed to load eval_dataset.json: %s", exc)
    return [
        {
            "case_id": "PRESET-01",
            "description": "SDK 3.1 AUTH_401 (Breaking Change Mismatch)",
            "incident": {
                "description": "AUTH_401 on all API requests after upgrading from SDK 2.8 to SDK 3.1. We were sending 'Authorization: ApiKey' but now every request fails.",
                "current_version": "3.1",
                "previous_version": "2.8",
                "error_codes": ["AUTH_401"],
                "product": "DevCore SDK",
            },
        },
        {
            "case_id": "PRESET-02",
            "description": "Missing Version Info (Triggers NEEDS_INFO)",
            "incident": {
                "description": "Authentication fails with HTTP 401 when initializing the SDK client.",
                "current_version": "",
                "previous_version": "",
                "error_codes": ["AUTH_401"],
                "product": "DevCore SDK",
            },
        },
        {
            "case_id": "PRESET-03",
            "description": "Rate Limiting & Backoff (Straightforward)",
            "incident": {
                "description": "Our integration keeps getting RATE_429 errors during batch processing jobs. How do we handle rate limiting?",
                "current_version": "3.1",
                "previous_version": "",
                "error_codes": ["RATE_429"],
                "product": "DevCore API",
            },
        },
    ]


def load_evaluation_report(
    report_path: Path = Path("data/eval_results/evaluation_results.json"),
) -> tuple[EvaluationReport | None, str | None]:
    """Load the latest M7 artifact without recalculating evaluation metrics."""
    if not report_path.exists():
        return None, f"Evaluation report not found: {report_path}"
    try:
        return EvaluationReport.model_validate_json(
            report_path.read_text(encoding="utf-8")
        ), None
    except Exception as exc:
        logger.warning("Failed to load evaluation report: %s", exc)
        return None, "Evaluation report is malformed or incompatible. Run M7 again."


# ---------------------------------------------------------------------------
# Main Streamlit App Layout
# ---------------------------------------------------------------------------

def main() -> None:
    # Sidebar
    st.sidebar.markdown(
        '<div class="sidebar-brand"><div class="sidebar-mark">D</div>'
        '<div><strong>DevTrace</strong><small>Evidence-grounded troubleshooting</small></div></div>',
        unsafe_allow_html=True,
    )

    llm_mode = st.sidebar.radio(
        "LLM Mode",
        options=["auto", "fake", "live"],
        index=0,
        format_func=lambda x: {
            "auto": "Auto (Gemini if key set)",
            "fake": "Fake LLM (Safety-Path Demo)",
            "live": "Live Gemini LLM",
        }[x],
        help="Select LLM provider mode for diagnosis and verification.",
    )

    if settings.gemini_api_key:
        st.sidebar.markdown(
            f'<div class="connection-pill"><span class="connection-dot"></span>'
            f'Gemini connected · {settings.gemini_model}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.info("Running without a Gemini key. Safety-path mode is available.")

    st.sidebar.divider()
    st.sidebar.markdown(
        "**Trust boundary**\n\n`RETRIEVED ≠ APPLICABLE`  \n"
        "`APPLICABLE ≠ SUPPORTED`  \n`SUPPORTED ≠ SURVIVED`"
    )

    # Header
    st.markdown('<div class="section-kicker">Developer reliability system</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Troubleshoot with evidence,<br>not guesswork.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">DevTrace retrieves technical evidence, filters it by version, verifies every generated claim, and withholds answers that do not survive those checks.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="invariant-banner"><span>Trust pipeline</span><span class="flow">RETRIEVED → APPLICABLE → SUPPORTED → SURVIVED</span></div>',
        unsafe_allow_html=True,
    )

    # Main Tabs
    tab_diag, tab_bench, tab_arch = st.tabs([
        "Diagnose",
        "Evaluation",
        "Architecture",
    ])

    # -----------------------------------------------------------------------
    # TAB 1: Interactive Incident Diagnostics
    # -----------------------------------------------------------------------
    with tab_diag:
        render_diagnostics_tab(llm_mode)

    # -----------------------------------------------------------------------
    # TAB 2: Evaluation & Benchmark Dashboard
    # -----------------------------------------------------------------------
    with tab_bench:
        render_benchmark_tab()

    # -----------------------------------------------------------------------
    # TAB 3: System Architecture & Invariants
    # -----------------------------------------------------------------------
    with tab_arch:
        render_architecture_tab()


# ---------------------------------------------------------------------------
# Tab Handlers
# ---------------------------------------------------------------------------

def render_diagnostics_tab(llm_mode: str) -> None:
    st.markdown("### Describe the incident")
    st.markdown(
        '<div class="section-intro">Add the symptoms you observed. Versions and error codes help DevTrace reject documentation that does not apply.</div>',
        unsafe_allow_html=True,
    )

    presets = load_preset_dataset()
    preset_options = ["Custom Input"] + [f"{p.get('case_id', 'PRESET')} — {p.get('description', '')}" for p in presets]
    
    selected_preset_str = st.selectbox(
        "Start from an example",
        options=preset_options,
        index=0,
    )

    # Default form values
    default_desc = ""
    default_curr = ""
    default_prev = ""
    default_errs = ""
    default_prod = ""

    if selected_preset_str != "Custom Input":
        idx = preset_options.index(selected_preset_str) - 1
        preset = presets[idx]
        inc = preset.get("incident", {})
        default_desc = inc.get("description", "")
        default_curr = inc.get("current_version", "") or ""
        default_prev = inc.get("previous_version", "") or ""
        default_errs = ", ".join(inc.get("error_codes", []))
        default_prod = inc.get("product", "") or ""

    with st.form("incident_form"):
        desc = st.text_area(
            "Incident Description*",
            value=default_desc,
            placeholder="e.g., Getting AUTH_401 errors after upgrading SDK from version 2.8 to 3.1...",
            height=120,
        )
        col1, col2 = st.columns(2)
        curr_ver = col1.text_input("Current Version", value=default_curr, placeholder="e.g., 3.1")
        prev_ver = col2.text_input("Previous Version", value=default_prev, placeholder="e.g., 2.8")
        col3, col4 = st.columns(2)
        err_codes_str = col3.text_input("Error Codes", value=default_errs, placeholder="e.g., AUTH_401, HTTP_401")
        prod = col4.text_input("Product / Component", value=default_prod, placeholder="e.g., DevCore SDK")

        submitted = st.form_submit_button("Run evidence-grounded diagnosis", type="primary", use_container_width=True)

    if submitted:
        if not desc.strip():
            st.warning("Please provide an incident description.")
            return

        err_codes = [e.strip() for e in err_codes_str.split(",") if e.strip()]

        retriever = get_cached_retriever()
        if retriever is None:
            st.error("Could not load knowledge corpus or retriever. Please ensure data directory is populated.")
            return

        try:
            llm_client = build_llm_client(mode=llm_mode)
        except Exception as exc:
            st.error(f"Failed to build LLM client: {exc}")
            return

        with st.spinner("Retrieving evidence, filtering applicability, and verifying claims..."):
            result = run_pipeline(
                description=desc,
                current_version=curr_ver or None,
                previous_version=prev_ver or None,
                error_codes=err_codes,
                product=prod or None,
                retriever=retriever,
                llm_client=llm_client,
            )

        st.divider()
        st.markdown('<div class="section-kicker">Verified response</div>', unsafe_allow_html=True)
        st.markdown("## Troubleshooting result")

        # Render sections using M8 render helpers
        render_final_answer(result, st)
        st.divider()
        render_pipeline_trace(result, st)
        st.divider()
        render_verification_section(result, st)
        st.divider()
        render_retry_section(result, st)
        st.divider()
        render_evidence_section(result, st)


def render_benchmark_tab() -> None:
    st.markdown("### Evaluation snapshot")
    st.markdown(
        '<div class="section-intro">A controlled comparison of naive retrieval, threshold abstention, and the complete DevTrace pipeline.</div>',
        unsafe_allow_html=True,
    )
    report, error = load_evaluation_report()

    if report is None:
        st.warning(error or "No trustworthy evaluation report is available.")
        st.caption("Run `python -m src.m7.runner` to generate a report.")
        return

    generated = report.generated_at_utc.isoformat() if report.generated_at_utc else "not recorded"
    st.caption(
        f"Latest {report.evaluation_mode} evaluation: {report.total_cases} controlled "
        f"developer-troubleshooting incidents · generated {generated}."
    )

    if report.evaluation_mode == "deterministic":
        st.info(
            "These are controlled FakeLLM fixture results used to validate pipeline "
            "contracts. They are not independent measurements of live-model quality."
        )

    st.markdown(
        '<div class="system-grid">'
        '<div class="system-card"><strong>Baseline A</strong><span>Naive hybrid retrieval with no version filtering or claim verification.</span></div>'
        '<div class="system-card"><strong>Baseline B</strong><span>Threshold-based abstention without applicability or grounding checks.</span></div>'
        '<div class="system-card"><strong>DevTrace</strong><span>Version filtering, per-claim verification, and one targeted retry.</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    def pct(value: float | None) -> str:
        return "N/A" if value is None else f"{value * 100:.1f}%"

    dt = report.aggregate_devtrace
    od = report.outcome_distribution
    answered = (od.answered_full + od.answered_partial) if od else 0
    st.markdown(
        '<div class="detail-stats">'
        f'<div class="detail-stat"><span>Cases</span><strong>{report.total_cases}</strong></div>'
        f'<div class="detail-stat"><span>Verified answers</span><strong>{answered}</strong></div>'
        f'<div class="detail-stat"><span>False answer rate</span><strong>{pct(dt.false_answer_rate if dt else None)}</strong></div>'
        f'<div class="detail-stat"><span>False abstention rate</span><strong>{pct(dt.false_abstention_rate if dt else None)}</strong></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    if od and od.degraded:
        st.error(
            f"{od.degraded} case(s) ended DEGRADED. Treat this run as having execution "
            "failures, not as evidence of successful diagnosis."
        )

    st.divider()
    st.subheader("Comparative metrics")

    def metric_row(name: str, metrics: Any) -> dict[str, str]:
        return {
            "System": name,
            "Retrieval hit": pct(metrics.retrieval_hit_rate if metrics else None),
            "Forbidden retrieval": pct(metrics.forbidden_retrieval_rate if metrics else None),
            "Citation validity": pct(metrics.citation_validity_rate if metrics else None),
            "Citation correctness": pct(metrics.citation_correctness_rate if metrics else None),
            "False answer": pct(metrics.false_answer_rate if metrics else None),
            "False abstention": pct(metrics.false_abstention_rate if metrics else None),
        }

    bench_data = [
        metric_row("Baseline A (Naive RAG)", report.aggregate_a),
        metric_row("Baseline B (Threshold RAG)", report.aggregate_b),
        metric_row("DevTrace (Full Pipeline)", report.aggregate_devtrace),
    ]

    st.dataframe(bench_data, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Outcome distribution")
    if od:
        st.dataframe([{
            "ANSWERED_FULL": od.answered_full,
            "ANSWERED_PARTIAL": od.answered_partial,
            "INSUFFICIENT_EVIDENCE": od.insufficient_evidence,
            "NEEDS_INFO": od.needs_info,
            "DEGRADED": od.degraded,
        }], hide_index=True, use_container_width=True)

    st.subheader("Dataset categories")
    st.dataframe([
        {"Category": item.category, "Cases": item.case_count}
        for item in report.category_metrics
    ], hide_index=True, use_container_width=True)


def render_architecture_tab() -> None:
    st.markdown("### How DevTrace earns an answer")
    st.markdown(
        '<div class="section-intro">Each stage narrows what is allowed to reach the user. Retrieval is only the beginning of the trust pipeline.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="architecture-flow">'
        '<div class="architecture-node"><em>M2</em><strong>Normalize</strong><span>Extract versions, products, and error signals.</span></div>'
        '<div class="architecture-node"><em>M3</em><strong>Retrieve</strong><span>Combine dense similarity with BM25 keyword search.</span></div>'
        '<div class="architecture-node"><em>M4</em><strong>Filter & diagnose</strong><span>Request missing facts and remove incompatible evidence.</span></div>'
        '<div class="architecture-node"><em>M5</em><strong>Verify</strong><span>Validate citations and independently ground every claim.</span></div>'
        '<div class="architecture-node"><em>M6</em><strong>Survive</strong><span>Require a verified root cause and allow at most one retry.</span></div>'
        '<div class="architecture-node"><em>M7</em><strong>Evaluate</strong><span>Compare controlled outcomes against two RAG baselines.</span></div>'
        '<div class="architecture-node"><em>M8</em><strong>Present</strong><span>Expose the answer, evidence, and audit trail without new reasoning.</span></div>'
        '<div class="architecture-node"><em>OUTCOME</em><strong>Answer or abstain</strong><span>Return only supported content—or explain why no answer is safe.</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Reliability invariants")
    st.markdown(
        '<div class="pillar-grid">'
        '<div class="pillar-card"><strong>Retrieved ≠ Applicable</strong><span>Similarity does not prove version compatibility. M4 removes evidence that does not apply.</span></div>'
        '<div class="pillar-card"><strong>Applicable ≠ Supported</strong><span>Relevant context does not make generated prose true. M5 checks each claim against the text.</span></div>'
        '<div class="pillar-card"><strong>Supported ≠ Survived</strong><span>The root cause itself must verify before a diagnosis can enter the final answer.</span></div>'
        '<div class="pillar-card"><strong>Missing facts stop the pipeline</strong><span>When versions change the correct answer, NEEDS_INFO asks for clarification before generation.</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
