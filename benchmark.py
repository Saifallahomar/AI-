"""
benchmark.py
Deterministic comparator matching and cover-limit benchmarking. No LLM calls happen here -
every number produced by this module comes directly from the dataset or simple arithmetic on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data_loader import Dataset, CANONICAL_COVERS, _funding_stage_rank


@dataclass
class Prospect:
    vertical: str
    employee_count: float | None
    turnover_gbp: float | None
    funding_raised_gbp: float | None
    funding_series: str | None
    current_covers: dict  # canonical_cover -> current_limit_gbp (float or None)


@dataclass
class Comparator:
    client_id: str
    similarity: float
    employee_count: float | None
    turnover_gbp: float | None
    funding_raised_gbp: float | None
    funding_series: str | None


@dataclass
class ComparatorResult:
    comparators: list  # list[Comparator], ranked best-first
    pool_size: int      # size of the whole same-vertical pool considered
    confidence: str      # High / Medium / Low
    confidence_reason: str


@dataclass
class CoverBenchmark:
    cover: str
    current_limit_gbp: float | None
    n_peers: int
    median_gbp: float | None
    p25_gbp: float | None
    p75_gbp: float | None
    pct_peers_above_prospect: float | None
    confidence: str
    confidence_reason: str
    status: str          # significantly_below / slightly_below / around / above / no_data
    priority: str         # Immediate action / For consideration / Review at renewal / (none)
    source_note: str


# ---------------------------------------------------------------------------
# Comparator matching
# ---------------------------------------------------------------------------

def _log_dist(a: float, b: float) -> float:
    """Normalised (0..1-ish) distance between two positive numbers on a log scale."""
    a = max(a, 1.0)
    b = max(b, 1.0)
    return abs(np.log10(a) - np.log10(b)) / 3.0  # /3 => 3 orders of magnitude ~= max distance


def find_comparators(ds: Dataset, prospect: Prospect, max_n: int = 50) -> ComparatorResult:
    pool = ds.company_profiles[ds.company_profiles["vertical"] == prospect.vertical].copy()
    pool_size = len(pool)

    prospect_stage_rank = _funding_stage_rank(prospect.funding_series)

    scored = []
    for _, row in pool.iterrows():
        components = []
        weights = []

        if prospect.employee_count and row["employee_count"]:
            components.append(1 - _log_dist(prospect.employee_count, row["employee_count"]))
            weights.append(1.0)
        if prospect.turnover_gbp and row["turnover_gbp"]:
            components.append(1 - _log_dist(prospect.turnover_gbp, row["turnover_gbp"]))
            weights.append(1.0)
        if prospect_stage_rank is not None and row["funding_stage_rank"] is not None:
            stage_dist = abs(prospect_stage_rank - row["funding_stage_rank"]) / 6.0
            components.append(1 - stage_dist)
            weights.append(0.75)

        if not components:
            similarity = 0.0  # same vertical only, no other signal available
        else:
            similarity = float(np.average(components, weights=weights))
        similarity = max(0.0, min(1.0, similarity))

        scored.append(Comparator(
            client_id=row["client_id"],
            similarity=round(similarity, 3),
            employee_count=row["employee_count"],
            turnover_gbp=row["turnover_gbp"],
            funding_raised_gbp=row["funding_raised_gbp"],
            funding_series=row["funding_series"],
        ))

    scored.sort(key=lambda c: c.similarity, reverse=True)
    top = scored[:max_n]

    if pool_size >= 25:
        conf, reason = "High", f"{pool_size} {prospect.vertical} companies available in the comparator pool."
    elif pool_size >= 10:
        conf, reason = "Medium", f"Only {pool_size} {prospect.vertical} companies available in the comparator pool."
    else:
        conf, reason = "Low", f"Just {pool_size} {prospect.vertical} companies available in the comparator pool."

    return ComparatorResult(comparators=top, pool_size=pool_size, confidence=conf, confidence_reason=reason)


# ---------------------------------------------------------------------------
# Per-cover benchmarking
# ---------------------------------------------------------------------------

def _confidence_for_n(n: int) -> tuple[str, str]:
    if n >= 10:
        return "High", f"{n} comparator companies had this cover on record."
    if n >= 5:
        return "Medium", f"Only {n} comparator companies had this cover on record."
    if n >= 1:
        return "Low", f"Just {n} comparator company had this cover on record." if n == 1 else f"Just {n} comparator companies had this cover on record."
    return "Low", "No comparator companies had this cover on record."


def _classify_status(current: float | None, p25: float, median: float, p75: float) -> str:
    if current is None:
        return "no_data"
    if current < p25:
        return "significantly_below"
    if current < median:
        return "slightly_below"
    if current <= p75:
        return "around"
    return "above"


_STATUS_PRIORITY = {
    "no_data": "Immediate action",
    "significantly_below": "Immediate action",
    "slightly_below": "For consideration",
    "around": "Review at renewal",
    "above": "Review at renewal",
}


def benchmark_cover(ds: Dataset, comparator_ids: list[str], cover: str, current_limit: float | None) -> CoverBenchmark | None:
    """Benchmarks one canonical cover for the prospect against the given comparator client_ids,
    pooling recorder_cover_lines.csv (primary, numeric) and deck_limit_benchmarks.csv
    (secondary, parsed) evidence."""

    cl = ds.cover_lines
    cl_sub = cl[(cl["client_id"].isin(comparator_ids)) & (cl["canonical_cover"] == cover)]
    limits_a = cl_sub["limit_amount"].dropna().tolist()
    sources_a = len(limits_a)

    db = ds.deck_benchmarks
    db_sub = db[(db["client_id"].isin(comparator_ids)) & (db["canonical_cover"] == cover)]
    limits_b = db_sub["current_limit_numeric"].dropna().tolist()
    sources_b = len(limits_b)

    all_limits = limits_a + limits_b
    n_peers = len(all_limits)

    if n_peers == 0:
        return None

    arr = np.array(all_limits, dtype=float)
    median = float(np.median(arr))
    p25 = float(np.percentile(arr, 25))
    p75 = float(np.percentile(arr, 75))

    pct_above = None
    if current_limit is not None and n_peers > 0:
        pct_above = float(np.mean(arr > current_limit) * 100)

    status = _classify_status(current_limit, p25, median, p75)
    priority = _STATUS_PRIORITY[status]
    conf, conf_reason = _confidence_for_n(n_peers)

    source_bits = []
    if sources_a:
        source_bits.append(f"{sources_a} live/expired policy record(s) from Capsule's Recorder platform")
    if sources_b:
        source_bits.append(f"{sources_b} prior health-check benchmark record(s)")
    source_note = " and ".join(source_bits) + "."

    return CoverBenchmark(
        cover=cover,
        current_limit_gbp=current_limit,
        n_peers=n_peers,
        median_gbp=median,
        p25_gbp=p25,
        p75_gbp=p75,
        pct_peers_above_prospect=pct_above,
        confidence=conf,
        confidence_reason=conf_reason,
        status=status,
        priority=priority,
        source_note=source_note,
    )


def benchmark_all_covers(ds: Dataset, comparator_ids: list[str], prospect_covers: dict) -> list[CoverBenchmark]:
    """prospect_covers: dict of canonical_cover -> current_limit_gbp (float or None).
    Only covers the prospect actually asked about (or all CANONICAL_COVERS if none specified)
    are benchmarked."""

    covers = prospect_covers if prospect_covers else {c: None for c in CANONICAL_COVERS}
    results = []
    for cover, current_limit in covers.items():
        bm = benchmark_cover(ds, comparator_ids, cover, current_limit)
        if bm is not None:
            results.append(bm)
    return results
