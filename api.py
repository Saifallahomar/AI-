"""
api.py
FastAPI wrapper around the deterministic benchmarking engine (benchmark.py /
recommendation.py / report.py) and the AI phrasing layer (ai_service.py).

This is the "API-centric engine" the brief calls for. Streamlit is just one
client of it now — Capsule's marketing site, a broker-facing tool, or anything
else could call the same two endpoints. No benchmarking/recommendation logic
lives here; this file only translates HTTP <-> the existing dataclasses.

Run with:
    uvicorn api:app --reload --port 8000

Docs auto-served at http://localhost:8000/docs
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from data_loader import load_dataset, CANONICAL_COVERS
from benchmark import Prospect, find_comparators, benchmark_all_covers
from recommendation import build_all_recommendations
from ai_service import phrase_recommendation
from report import build_report_pdf

DATA_ROOT = os.environ.get("CAPSULE_DATA_ROOT", "dataset")

app = FastAPI(
    title="Capsule Cover Benchmarking API",
    description="Profile in, benchmark + recommendations out. Guidance only, not regulated advice.",
    version="0.1.0",
)

_ds = None  # loaded once, lazily, shared across requests


def get_ds():
    global _ds
    if _ds is None:
        try:
            _ds = load_dataset(DATA_ROOT)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Dataset failed to load from '{DATA_ROOT}': {e}",
            )
    return _ds


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ProfileIn(BaseModel):
    vertical: str = "FinTech"
    employee_count: Optional[float] = None
    turnover_gbp: Optional[float] = None
    funding_raised_gbp: Optional[float] = None
    funding_series: Optional[str] = None
    # canonical cover name -> current limit in GBP (omit or null = not held)
    current_covers: dict[str, Optional[float]] = {}


class ComparatorSummary(BaseModel):
    pool_size: int
    matched: int
    confidence: str
    confidence_reason: str


class BenchmarkOut(BaseModel):
    comparators: ComparatorSummary
    recommendations: list[dict]


# ---------------------------------------------------------------------------
# Shared core (used by both endpoints so /report can never drift from /benchmark)
# ---------------------------------------------------------------------------

def _run(profile: ProfileIn):
    ds = get_ds()
    covers = profile.current_covers or {c: None for c in CANONICAL_COVERS}

    prospect = Prospect(
        vertical=profile.vertical,
        employee_count=profile.employee_count,
        turnover_gbp=profile.turnover_gbp,
        funding_raised_gbp=profile.funding_raised_gbp,
        funding_series=profile.funding_series,
        current_covers=covers,
    )

    comparator_result = find_comparators(ds, prospect, max_n=50)
    comparator_ids = [c.client_id for c in comparator_result.comparators]
    benchmarks = benchmark_all_covers(ds, comparator_ids, covers)
    recommendations = build_all_recommendations(
        benchmarks, prospect.vertical, ds.deck_recommendations
    )

    ai_texts = {rec.cover: phrase_recommendation(
        rec) for rec in recommendations}
    return prospect, comparator_result, recommendations, ai_texts


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/covers")
def covers():
    """Canonical cover taxonomy prospects/UI should offer."""
    return {"canonical_covers": CANONICAL_COVERS}


@app.post("/benchmark", response_model=BenchmarkOut)
def benchmark(profile: ProfileIn):
    _prospect, comparator_result, recommendations, ai_texts = _run(profile)

    recs_out = []
    for rec in recommendations:
        d = rec.as_dict()
        d["explanation"] = ai_texts[rec.cover]
        recs_out.append(d)

    return BenchmarkOut(
        comparators=ComparatorSummary(
            pool_size=comparator_result.pool_size,
            matched=len(comparator_result.comparators),
            confidence=comparator_result.confidence,
            confidence_reason=comparator_result.confidence_reason,
        ),
        recommendations=recs_out,
    )


@app.post("/report")
def report(profile: ProfileIn):
    """Recomputes the same benchmark server-side and returns a PDF.
    Deliberately does not accept a client-supplied recommendation set, so the
    downloaded report can never disagree with what /benchmark returned."""
    prospect, comparator_result, recommendations, ai_texts = _run(profile)

    business_profile = {
        "vertical": prospect.vertical,
        "employee_count": prospect.employee_count,
        "turnover_gbp": prospect.turnover_gbp,
        "funding_raised_gbp": prospect.funding_raised_gbp,
        "funding_series": prospect.funding_series,
    }

    pdf_bytes = build_report_pdf(
        business_profile, comparator_result, recommendations, ai_texts
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": "attachment; filename=capsule_cover_benchmark_report.pdf"
        },
    )
