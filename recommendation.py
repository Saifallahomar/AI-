"""
recommendation.py
Turns a CoverBenchmark (all-deterministic numbers) into a full recommendation record:
priority, a plain-English deterministic explanation (used if the AI layer is unavailable),
and up to 2 short pieces of grounding evidence pulled verbatim-adjacent from
deck_recommendations.csv (Capsule's own historical broker guidance for the same vertical
and cover). The AI layer (ai_service.py) only rephrases these facts - it does not add any.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd

from benchmark import CoverBenchmark
from data_loader import canonicalise_cover


def _fmt_gbp(x: float | None) -> str:
    if x is None:
        return "not provided"
    if x >= 1_000_000:
        return f"£{x/1_000_000:.2f}m".replace(".00m", "m")
    if x >= 1_000:
        return f"£{x/1_000:.0f}k"
    return f"£{x:,.0f}"


_STATUS_TEXT = {
    "no_data": "you did not record a current limit for this cover",
    "significantly_below": "your current limit is significantly below the peer benchmark",
    "slightly_below": "your current limit is below the peer median",
    "around": "your current limit is broadly in line with peers",
    "above": "your current limit is above the peer median",
}


@dataclass
class Recommendation:
    cover: str
    current_limit_gbp: float | None
    median_gbp: float | None
    p25_gbp: float | None
    p75_gbp: float | None
    n_peers: int
    confidence: str
    confidence_reason: str
    priority: str
    status: str
    deterministic_explanation: str
    evidence_snippets: list  # list[str], drawn from deck_recommendations.csv
    source_note: str

    def as_dict(self):
        return asdict(self)


def _retrieve_evidence(deck_recs: pd.DataFrame, vertical: str, cover: str, max_items: int = 2) -> list[str]:
    """Keyword-matches deck_recommendations.csv `item`/`action` text against the canonical
    cover for the same vertical, to ground the AI's prose in real historical broker language.
    This is retrieval, not generation - text is truncated but not altered."""

    sub = deck_recs[deck_recs["vertical"] == vertical].copy()
    if sub.empty:
        return []

    sub["_canon"] = sub["item"].apply(canonicalise_cover)
    matches = sub[sub["_canon"] == cover]
    if matches.empty:
        # fall back to matching on the action text too
        sub["_canon_action"] = sub["action"].apply(canonicalise_cover)
        matches = sub[sub["_canon_action"] == cover]

    snippets = []
    for _, row in matches.head(max_items).iterrows():
        text = row.get("action") if isinstance(
            row.get("action"), str) else row.get("item")
        if isinstance(text, str) and text.strip():
            snippet = text.strip().replace("\n", " ")
            if len(snippet) > 280:
                snippet = snippet[:277] + "..."
            snippets.append(snippet)
    return snippets


def build_recommendation(bm: CoverBenchmark, vertical: str, deck_recs: pd.DataFrame) -> Recommendation:
    status_text = _STATUS_TEXT[bm.status]

    if bm.status == "no_data":
        explanation = (
            f"No current limit was provided for {bm.cover}, so the prospect's current cover "
            f"cannot be compared against the peer benchmark. Among {bm.n_peers} comparable {vertical} companies with recorded "
            f"cover, the median limit is {_fmt_gbp(bm.median_gbp)} (peer range "
            f"{_fmt_gbp(bm.p25_gbp)}\u2013{_fmt_gbp(bm.p75_gbp)}). We recommend confirming your "
            f"current limit for this line with a Capsule broker."
        )
    else:
        pct_bit = ""
        if bm.pct_peers_above_prospect is not None:
            pct_bit = f" {bm.pct_peers_above_prospect:.0f}% of comparable peers carry a higher limit than you."
        explanation = (
            f"Your current {bm.cover} limit is {_fmt_gbp(bm.current_limit_gbp)}. Among "
            f"{bm.n_peers} comparable {vertical} companies, the median limit is "
            f"{_fmt_gbp(bm.median_gbp)} (peer range {_fmt_gbp(bm.p25_gbp)}\u2013{_fmt_gbp(bm.p75_gbp)}), "
            f"so {status_text}.{pct_bit}"
        )

    evidence = _retrieve_evidence(deck_recs, vertical, bm.cover)

    return Recommendation(
        cover=bm.cover,
        current_limit_gbp=bm.current_limit_gbp,
        median_gbp=bm.median_gbp,
        p25_gbp=bm.p25_gbp,
        p75_gbp=bm.p75_gbp,
        n_peers=bm.n_peers,
        confidence=bm.confidence,
        confidence_reason=bm.confidence_reason,
        priority=bm.priority,
        status=bm.status,
        deterministic_explanation=explanation,
        evidence_snippets=evidence,
        source_note=bm.source_note,
    )


def build_all_recommendations(benchmarks: list[CoverBenchmark], vertical: str, deck_recs: pd.DataFrame) -> list[Recommendation]:
    recs = [build_recommendation(bm, vertical, deck_recs) for bm in benchmarks]
    priority_order = {"Immediate action": 0,
                      "For consideration": 1, "Review at renewal": 2}
    recs.sort(key=lambda r: priority_order.get(r.priority, 3))
    return recs
