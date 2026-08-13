"""
app.py — Capsule Cover Benchmarking prototype (hackathon Brief B).
Run with: streamlit run app.py
"""

import os
import streamlit as st

from data_loader import load_dataset, CANONICAL_COVERS
from benchmark import Prospect, find_comparators, benchmark_all_covers
from recommendation import build_all_recommendations
from ai_service import phrase_recommendation
from report import build_report_pdf

st.set_page_config(page_title="Capsule Cover Benchmarking", page_icon="🛡️", layout="wide")

DATA_ROOT = os.environ.get("CAPSULE_DATA_ROOT", "dataset")

# Only FinTech & Venture is wired up end-to-end for this hackathon build.
SUPPORTED_VERTICALS = ["FinTech & Venture"]
VERTICAL_LABEL_TO_DATASET = {"FinTech & Venture": "FinTech"}

FUNDING_STAGES = ["Bootstrapped", "Pre-Seed", "Seed", "Series A", "Series B", "Series C", "Series D", "Other"]


@st.cache_resource(show_spinner="Loading Capsule dataset...")
def get_dataset():
    return load_dataset(DATA_ROOT)


def _fmt_gbp(x):
    if x is None:
        return "—"
    if x >= 1_000_000:
        return f"£{x/1_000_000:.2f}m".replace(".00m", "m")
    if x >= 1_000:
        return f"£{x/1_000:.0f}k"
    return f"£{x:,.0f}"


def init_state():
    defaults = {
        "page": 1,
        "vertical": "FinTech & Venture",
        "employee_count": None,
        "turnover_gbp": None,
        "funding_raised_gbp": None,
        "funding_series": None,
        "current_covers": {c: None for c in CANONICAL_COVERS},
        "results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

try:
    ds = get_dataset()
    dataset_error = None
except Exception as e:
    ds = None
    dataset_error = str(e)

st.sidebar.title("🛡️ Capsule Benchmark")
st.sidebar.caption("Cover Benchmarking — hackathon prototype")

if dataset_error:
    st.sidebar.error(f"Dataset not found at '{DATA_ROOT}'.")
    st.error(
        f"Could not load the dataset from `{DATA_ROOT}/`.\n\n"
        f"Error: {dataset_error}\n\n"
        "Unzip the Capsule dataset so this folder contains `dataset/README.md`, "
        "`dataset/2_book_of_business/`, etc., or set the CAPSULE_DATA_ROOT environment variable "
        "to point at it."
    )
    st.stop()

pages = ["1. Business profile", "2. Current cover", "3. Benchmark results", "4. Download report"]
choice = st.sidebar.radio("Steps", pages, index=st.session_state["page"] - 1)
st.session_state["page"] = pages.index(choice) + 1

st.sidebar.divider()
st.sidebar.caption(
    "Guidance only — not regulated insurance advice. "
    "This tool benchmarks against Capsule's anonymised client dataset."
)

# Optional: preload a synthetic test brief
with st.sidebar.expander("Load a test prospect (synthetic, fictional)"):
    briefs = ds.synthetic_briefs
    fin_briefs = briefs[briefs["vertical"] == "FinTech"] if not briefs.empty else briefs
    if not fin_briefs.empty:
        options = ["—"] + fin_briefs["brief_id"].tolist()
        pick = st.selectbox("FinTech synthetic briefs", options)
        if pick != "—":
            row = fin_briefs[fin_briefs["brief_id"] == pick].iloc[0]
            st.caption(row["business_description"])
            if st.button("Use this brief"):
                st.session_state["employee_count"] = float(row["employee_count"])
                st.session_state["turnover_gbp"] = float(row["turnover_gbp"])
                st.session_state["funding_raised_gbp"] = float(row["total_funding_raised_gbp"])
                st.session_state["funding_series"] = row["funding_series"]
                st.session_state["page"] = 1
                st.success("Loaded. Current cover was NOT auto-filled — enter it on step 2 (deliberately fictional/no join to real cover data).")
                st.rerun()

# ---------------------------------------------------------------------------
# Page 1: Business profile
# ---------------------------------------------------------------------------
if st.session_state["page"] == 1:
    st.header("1. Business profile")
    st.write("Tell us about your business. This is used to find comparable companies.")

    vertical_label = st.selectbox("Vertical", SUPPORTED_VERTICALS, index=0)
    st.session_state["vertical"] = vertical_label
    if vertical_label != "FinTech & Venture":
        st.info("Only FinTech & Venture is wired up end-to-end in this prototype build.")

    col1, col2 = st.columns(2)
    with col1:
        st.session_state["employee_count"] = st.number_input(
            "Number of employees", min_value=0, value=int(st.session_state["employee_count"] or 0), step=1,
        )
        st.session_state["turnover_gbp"] = st.number_input(
            "Turnover, past financial year (£)", min_value=0,
            value=int(st.session_state["turnover_gbp"] or 0), step=10_000,
        )
    with col2:
        st.session_state["funding_raised_gbp"] = st.number_input(
            "Total funding raised (£, if applicable)", min_value=0,
            value=int(st.session_state["funding_raised_gbp"] or 0), step=10_000,
        )
        current_stage = st.session_state["funding_series"] or FUNDING_STAGES[0]
        idx = FUNDING_STAGES.index(current_stage) if current_stage in FUNDING_STAGES else 0
        st.session_state["funding_series"] = st.selectbox("Funding stage", FUNDING_STAGES, index=idx)

    st.button("Next: Current cover →", on_click=lambda: st.session_state.update(page=2))

# ---------------------------------------------------------------------------
# Page 2: Current cover
# ---------------------------------------------------------------------------
elif st.session_state["page"] == 2:
    st.header("2. Current insurance cover")
    st.write("Enter your current limit for any covers you hold. Leave blank if you don't hold that cover.")

    cols = st.columns(2)
    for i, cover in enumerate(CANONICAL_COVERS):
        with cols[i % 2]:
            existing = st.session_state["current_covers"].get(cover)
            val = st.number_input(
                f"{cover} — current limit (£)", min_value=0,
                value=int(existing or 0), step=10_000, key=f"cover_{cover}",
            )
            st.session_state["current_covers"][cover] = float(val) if val > 0 else None

    c1, c2 = st.columns(2)
    c1.button("← Back", on_click=lambda: st.session_state.update(page=1))
    c2.button("Next: Benchmark results →", on_click=lambda: st.session_state.update(page=3))

# ---------------------------------------------------------------------------
# Page 3: Benchmark results
# ---------------------------------------------------------------------------
elif st.session_state["page"] == 3:
    st.header("3. Benchmark results")

    if not st.session_state["employee_count"] and not st.session_state["turnover_gbp"]:
        st.warning("Add at least employee count or turnover on step 1 for a meaningful comparator match.")

    with st.spinner("Finding comparators and benchmarking cover..."):
        prospect = Prospect(
            vertical=VERTICAL_LABEL_TO_DATASET[st.session_state["vertical"]],
            employee_count=st.session_state["employee_count"] or None,
            turnover_gbp=st.session_state["turnover_gbp"] or None,
            funding_raised_gbp=st.session_state["funding_raised_gbp"] or None,
            funding_series=st.session_state["funding_series"],
            current_covers=st.session_state["current_covers"],
        )
        comparator_result = find_comparators(ds, prospect, max_n=50)
        comparator_ids = [c.client_id for c in comparator_result.comparators]
        benchmarks = benchmark_all_covers(ds, comparator_ids, st.session_state["current_covers"])
        recommendations = build_all_recommendations(benchmarks, prospect.vertical, ds.deck_recommendations)

        ai_texts = {}
        for rec in recommendations:
            ai_texts[rec.cover] = phrase_recommendation(rec)

    st.session_state["results"] = {
        "prospect": prospect,
        "comparator_result": comparator_result,
        "recommendations": recommendations,
        "ai_texts": ai_texts,
    }

    conf_colour = {"High": "green", "Medium": "orange", "Low": "red"}.get(comparator_result.confidence, "grey")
    st.markdown(
        f"**Comparator group:** {len(comparator_result.comparators)} companies matched "
        f"(pool of {comparator_result.pool_size} in {prospect.vertical}) &nbsp;&nbsp;"
        f"**Confidence:** :{conf_colour}[{comparator_result.confidence}]"
    )
    st.caption(comparator_result.confidence_reason)

    st.divider()

    priority_icon = {"Immediate action": "🔴", "For consideration": "🟠", "Review at renewal": "🟢"}

    for rec in recommendations:
        with st.container(border=True):
            top = st.columns([3, 1])
            top[0].subheader(rec.cover)
            top[1].markdown(f"### {priority_icon.get(rec.priority,'')} {rec.priority}")

            m = st.columns(4)
            m[0].metric("Current cover", _fmt_gbp(rec.current_limit_gbp))
            m[1].metric("Peer median", _fmt_gbp(rec.median_gbp))
            m[2].metric("Comparator group", f"{rec.n_peers} companies")
            m[3].metric("Confidence", rec.confidence)

            st.caption(f"Peer 25th–75th percentile: {_fmt_gbp(rec.p25_gbp)} – {_fmt_gbp(rec.p75_gbp)}")

            st.write(ai_texts[rec.cover])
            if rec.evidence_snippets:
                with st.expander("Historical Capsule guidance referenced"):
                    for s in rec.evidence_snippets:
                        st.write(f"> {s}")
            st.caption(f"Source: {rec.source_note}")

    st.divider()
    c1, c2 = st.columns(2)
    c1.button("← Back", on_click=lambda: st.session_state.update(page=2))
    c2.button("Next: Download report →", on_click=lambda: st.session_state.update(page=4))

# ---------------------------------------------------------------------------
# Page 4: Download report
# ---------------------------------------------------------------------------
elif st.session_state["page"] == 4:
    st.header("4. Download report")

    results = st.session_state.get("results")
    if not results:
        st.warning("Run the benchmark on step 3 first.")
    else:
        business_profile = {
            "vertical": results["prospect"].vertical,
            "employee_count": results["prospect"].employee_count,
            "turnover_gbp": results["prospect"].turnover_gbp,
            "funding_raised_gbp": results["prospect"].funding_raised_gbp,
            "funding_series": results["prospect"].funding_series,
        }
        pdf_bytes = build_report_pdf(
            business_profile,
            results["comparator_result"],
            results["recommendations"],
            results["ai_texts"],
        )
        st.success("Report ready.")
        st.download_button(
            "⬇️ Download PDF report",
            data=pdf_bytes,
            file_name="capsule_cover_benchmark_report.pdf",
            mime="application/pdf",
        )
        st.caption(
            "Contains: business profile, peer comparator summary, per-cover benchmark & "
            "recommendation with priority/confidence/evidence, and the guidance disclaimer."
        )

    st.button("← Back", on_click=lambda: st.session_state.update(page=3))
