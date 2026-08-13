"""
data_loader.py
Loads the Capsule hackathon dataset and produces three clean, join-ready tables:

  - company_profiles  : one row per client_id, coalesced firmographics
                         (from healthcheck_questionnaire.csv + hubspot_company_context.csv)
  - cover_lines        : one row per (client_id, canonical_cover), numeric limit_amount
                         (from recorder_cover_lines.csv, canonicalised)
  - deck_benchmarks     : one row per (client_id, canonical_cover), parsed current/recommended
                         limit (from deck_limit_benchmarks.csv, canonicalised + parsed)

Also exposes deck_recommendations (raw, for retrieval grounding) and deck_bands (raw, generic
per-vertical benchmarking bands) and synthetic_prospect_briefs (for demo / testing only).

No column names are invented here — every column referenced below is taken verbatim from
DATA_DICTIONARY.md. Where a column is missing/blank the code degrades gracefully (NaN),
per the dataset's "treat missing joins as a real-world condition" guidance.
"""

from __future__ import annotations

import re
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Canonical cover taxonomy
# ---------------------------------------------------------------------------
# The raw `coverage` / `cover` text across recorder_cover_lines.csv and
# deck_limit_benchmarks.csv is inconsistent (e.g. "Directors' and Officers' Liability",
# "Directors and Officer/Corporate Liability", "Directors & Officers" all mean D&O).
# This maps raw text -> one of 8 canonical categories that match what FinTech prospects
# are actually asked about in healthcheck_questionnaire.csv's `policies_needed` field.

CANONICAL_COVERS = [
    "Directors & Officers Liability",
    "Cyber",
    "Professional Indemnity",
    "Crime",
    "Employers' Liability",
    "Public & Products Liability",
    "Business Interruption",
    "Property / Equipment & Contents",
]

_COVER_PATTERNS = [
    ("Directors & Officers Liability", r"directors|d ?& ?o\b|management liability"),
    ("Cyber", r"cyber"),
    ("Professional Indemnity", r"professional indemnity|professional liability|civil liability|errors.{0,15}omissions"),
    ("Crime", r"\bcrime\b|\btheft\b|fidelity"),
    ("Employers' Liability", r"employers.?.?.?liability|workers comp"),
    ("Public & Products Liability", r"public.*liability|products liability|general liability"),
    ("Business Interruption", r"business interruption"),
    ("Property / Equipment & Contents", r"\bcontents\b|property damage|computer.*equipment|equipment and contents"),
]


def canonicalise_cover(raw_text) -> str | None:
    """Map a raw cover/coverage string to one of CANONICAL_COVERS, or None if unmatched."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None
    t = raw_text.lower()
    for name, pattern in _COVER_PATTERNS:
        if re.search(pattern, t):
            return name
    return None


# ---------------------------------------------------------------------------
# Funding-stage ordering (for comparator similarity)
# ---------------------------------------------------------------------------
FUNDING_STAGE_RANK = {
    "bootstrapped": 0,
    "pre-seed": 1,
    "seed": 2,
    "series a": 3,
    "series b": 4,
    "series c": 5,
    "series d": 6,
    "series d+": 6,
    "growth": 6,
    "other": None,
}

# rough £ midpoints for HubSpot's coarse funding_band, used only as a fallback
FUNDING_BAND_MIDPOINT = {
    "<£1m": 500_000,
    "£1-2m": 1_500_000,
    "£2-5m": 3_500_000,
    "£5-10m": 7_500_000,
    "£10-20m": 15_000_000,
    "£20m+": 25_000_000,
}


def _to_float(value) -> float | None:
    """Robust numeric coercion. Handles the vast majority of clean numeric strings, and
    defensively handles the rare malformed JSON-blob quirk seen in this dataset
    (e.g. '{"amount":null,"currency":"GBP"}') by extracting a numeric amount if present,
    else returning None rather than raising."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return None if pd.isna(value) else float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            pass
        m = re.search(r'"amount"\s*:\s*([\d.]+)', s)
        if m:
            return float(m.group(1))
        return None
    return None


def _funding_stage_rank(series_label) -> float | None:
    if not isinstance(series_label, str):
        return None
    return FUNDING_STAGE_RANK.get(series_label.strip().lower())


def _parse_money(value) -> float | None:
    """Parse strings like '£10m', '10,000,000', '5m', '2,500,000' -> float.
    Returns None if unparseable. Deterministic, no guessing beyond unit suffixes."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)
    if not isinstance(value, str):
        return None
    s = value.strip().lower().replace("gbp", "").replace("£", "").replace(",", "").strip()
    if not s:
        return None
    m = re.match(r"^([\d.]+)\s*(m|k)?$", s)
    if not m:
        return None
    num = float(m.group(1))
    suffix = m.group(2)
    if suffix == "m":
        num *= 1_000_000
    elif suffix == "k":
        num *= 1_000
    return num


def parse_limit_range_midpoint(value) -> float | None:
    """deck_limit_benchmarks values are sometimes ranges e.g. '£10m - £15m'.
    Returns the midpoint of the range, or the single value if not a range."""
    if not isinstance(value, str):
        return _parse_money(value)
    parts = re.split(r"\s*-\s*", value.strip())
    parsed = [_parse_money(p) for p in parts]
    parsed = [p for p in parsed if p is not None]
    if not parsed:
        return None
    return float(np.mean(parsed))


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
@dataclass
class Dataset:
    company_profiles: pd.DataFrame
    cover_lines: pd.DataFrame
    deck_benchmarks: pd.DataFrame
    deck_recommendations: pd.DataFrame
    deck_bands: pd.DataFrame
    synthetic_briefs: pd.DataFrame


def load_dataset(root: str = "dataset") -> Dataset:
    """Load and normalise the dataset. `root` is the path to the folder that
    contains README.md, DATA_DICTIONARY.md, 1_recommendation_bank/, 2_book_of_business/,
    3_company_context/, 4_evaluation/ (i.e. the folder the user unzipped)."""

    def p(*parts):
        return os.path.join(root, *parts)

    questionnaire = pd.read_csv(p("3_company_context", "healthcheck_questionnaire.csv"))
    hubspot = pd.read_csv(p("3_company_context", "hubspot_company_context.csv"))
    recorder_cover = pd.read_csv(p("2_book_of_business", "recorder_cover_lines.csv"))
    recorder_policy = pd.read_csv(p("2_book_of_business", "recorder_book_of_business.csv"))
    deck_limits = pd.read_csv(p("2_book_of_business", "deck_limit_benchmarks.csv"))
    deck_recs = pd.read_csv(p("2_book_of_business", "deck_recommendations.csv"))
    deck_bands = pd.read_csv(p("2_book_of_business", "deck_benchmark_bands.csv"))
    try:
        briefs = pd.read_csv(p("4_evaluation", "synthetic_prospect_briefs.csv"))
    except FileNotFoundError:
        briefs = pd.DataFrame()

    company_profiles = _build_company_profiles(questionnaire, hubspot)
    cover_lines = _build_cover_lines(recorder_cover, recorder_policy)
    deck_benchmarks = _build_deck_benchmarks(deck_limits)

    return Dataset(
        company_profiles=company_profiles,
        cover_lines=cover_lines,
        deck_benchmarks=deck_benchmarks,
        deck_recommendations=deck_recs,
        deck_bands=deck_bands,
        synthetic_briefs=briefs,
    )


def _build_company_profiles(questionnaire: pd.DataFrame, hubspot: pd.DataFrame) -> pd.DataFrame:
    """One row per client_id with coalesced firmographics. Prefers the better-filled
    source per field (documented in DATA_DICTIONARY.md fill counts), falls back to the other."""

    q = questionnaire.set_index("client_id")
    h = hubspot.set_index("client_id")

    all_ids = sorted(set(q.index) | set(h.index))
    rows = []
    for cid in all_ids:
        qr = q.loc[cid] if cid in q.index else pd.Series(dtype=object)
        hr = h.loc[cid] if cid in h.index else pd.Series(dtype=object)

        vertical = hr.get("vertical") if isinstance(hr.get("vertical"), str) else qr.get("vertical")

        employee_count = _to_float(hr.get("employee_count"))
        if employee_count is None:
            employee_count = _to_float(qr.get("employee_count"))

        # turnover: prefer most-recent/most-complete questionnaire fields, then hubspot actual, then band
        turnover = None
        for field in ("turnover_last12m_gbp", "turnover_current_fy_est_gbp", "turnover_past_fy_gbp"):
            v = _to_float(qr.get(field))
            if v is not None:
                turnover = v
                break
        turnover_is_approx = False
        if turnover is None:
            v = _to_float(hr.get("turnover_gbp"))
            if v is not None:
                turnover = v
            else:
                v = _to_float(hr.get("annual_revenue_band"))
                if v is not None:
                    turnover = v
                    turnover_is_approx = True  # order-of-magnitude only per data dictionary

        funding_raised = _to_float(qr.get("total_funding_raised_gbp"))
        funding_is_approx = False
        if funding_raised is None:
            v = _to_float(hr.get("total_funding_gbp"))
            if v is not None:
                funding_raised = v
            else:
                band = hr.get("funding_band")
                if isinstance(band, str) and band in FUNDING_BAND_MIDPOINT:
                    funding_raised = FUNDING_BAND_MIDPOINT[band]
                    funding_is_approx = True

        funding_series = qr.get("funding_series") if isinstance(qr.get("funding_series"), str) else None
        stage_rank = _funding_stage_rank(funding_series)
        if stage_rank is None:
            stage_of_evolution = hr.get("stage_of_evolution")
            # loose mapping from Beauhurst stage_of_evolution -> same rank scale
            stage_map = {"seed": 2, "venture": 3, "growth": 5, "established": 6}
            if isinstance(stage_of_evolution, str):
                stage_rank = stage_map.get(stage_of_evolution.strip().lower())
                funding_series = funding_series or stage_of_evolution

        rows.append({
            "client_id": cid,
            "vertical": vertical,
            "employee_count": employee_count,
            "turnover_gbp": turnover,
            "turnover_is_approx": turnover_is_approx,
            "funding_raised_gbp": funding_raised,
            "funding_is_approx": funding_is_approx,
            "funding_series": funding_series,
            "funding_stage_rank": stage_rank,
            "customer_type": qr.get("customer_type"),
            "regulated_business": qr.get("regulated_business") if pd.notna(qr.get("regulated_business")) else hr.get("fca_regulated"),
            "region_city": hr.get("region_city"),
            "capsule_sector": hr.get("capsule_sector"),
            "policies_needed": qr.get("policies_needed"),
        })

    return pd.DataFrame(rows)


def _build_cover_lines(recorder_cover: pd.DataFrame, recorder_policy: pd.DataFrame) -> pd.DataFrame:
    """One row per canonicalised coverage line with a numeric limit_amount.
    Keeps policy status (Active/Expired/Cancelled) so callers can filter if desired;
    per README, expired policies remain valid benchmarking evidence for limits bought."""

    df = recorder_cover.copy()
    df["canonical_cover"] = df["coverage"].apply(canonicalise_cover)
    df = df[df["canonical_cover"].notna() & df["limit_amount"].notna()].copy()

    status_by_policy = recorder_policy.set_index("policy_id")["status"]
    df["policy_status"] = df["policy_id"].map(status_by_policy)

    return df[[
        "client_id", "vertical", "policy_id", "product", "canonical_cover",
        "limit_amount", "limit_basis", "excess_amount", "policy_status",
    ]]


def _build_deck_benchmarks(deck_limits: pd.DataFrame) -> pd.DataFrame:
    """Parses deck_limit_benchmarks.csv current_limit / capsule_recommended_limit text
    (e.g. '£10m', '£10m - £15m') into numeric midpoints, and canonicalises `cover`."""

    df = deck_limits.copy()
    df["canonical_cover"] = df["cover"].apply(canonicalise_cover)
    df["current_limit_numeric"] = df["current_limit"].apply(parse_limit_range_midpoint)
    df["capsule_recommended_numeric"] = df["capsule_recommended_limit"].apply(parse_limit_range_midpoint)
    df = df[df["canonical_cover"].notna()].copy()
    return df[[
        "client_id", "vertical", "canonical_cover", "cover",
        "current_limit_numeric", "capsule_recommended_numeric", "commentary",
    ]]
