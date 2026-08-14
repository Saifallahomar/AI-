"""Regression tests for trustworthy comparator and cover benchmarking.

The tests deliberately use tiny in-memory data frames so they never read or copy the
hackathon dataset.  They describe the behaviour the benchmark engine must guarantee:

* missing profile values must not create perfect matches;
* funding raised must influence otherwise-equal comparator ranking;
* one company contributes at most one observation per cover;
* Recorder is the primary source and deck data is only a fallback;
* peer counts describe unique companies, not policy/cover-line records.

Run from the project directory with:

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Support both `python -m unittest discover -s tests -v` and direct execution of
# `python tests/test_benchmark.py` from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark import Prospect, _classify_status, find_comparators, benchmark_cover
from data_loader import Dataset
from recommendation import build_recommendation


COVER = "Cyber"


def _dataset(
    profiles: list[dict],
    cover_lines: list[dict] | None = None,
    deck_benchmarks: list[dict] | None = None,
) -> Dataset:
    """Build the smallest Dataset needed by benchmark.py."""

    profile_columns = [
        "client_id",
        "vertical",
        "employee_count",
        "turnover_gbp",
        "funding_raised_gbp",
        "funding_series",
        "funding_stage_rank",
    ]
    cover_columns = [
        "client_id",
        "canonical_cover",
        "limit_amount",
        "policy_status",
        "effective_date",
        "end_date",
    ]
    deck_columns = ["client_id", "canonical_cover", "current_limit_numeric"]

    return Dataset(
        company_profiles=pd.DataFrame(profiles, columns=profile_columns),
        cover_lines=pd.DataFrame(cover_lines or [], columns=cover_columns),
        deck_benchmarks=pd.DataFrame(deck_benchmarks or [], columns=deck_columns),
        deck_recommendations=pd.DataFrame(),
        deck_bands=pd.DataFrame(),
        synthetic_briefs=pd.DataFrame(),
    )


def _profile(
    client_id: str,
    *,
    employees: float = 50,
    turnover: float = 5_000_000,
    funding: float = 5_000_000,
    stage: str = "Series A",
    stage_rank: float = 3,
) -> dict:
    return {
        "client_id": client_id,
        "vertical": "FinTech",
        "employee_count": employees,
        "turnover_gbp": turnover,
        "funding_raised_gbp": funding,
        "funding_series": stage,
        "funding_stage_rank": stage_rank,
    }


def _prospect(**overrides) -> Prospect:
    values = {
        "vertical": "FinTech",
        "employee_count": 50,
        "turnover_gbp": 5_000_000,
        "funding_raised_gbp": 5_000_000,
        "funding_series": "Series A",
        "current_covers": {COVER: 1_000_000},
    }
    values.update(overrides)
    return Prospect(**values)


class ComparatorMatchingTests(unittest.TestCase):
    def test_exact_profile_ranks_before_distant_profile(self):
        ds = _dataset([
            _profile("exact"),
            _profile("distant", employees=5, turnover=100_000, stage="Seed", stage_rank=2),
        ])

        result = find_comparators(ds, _prospect(), max_n=2)

        self.assertEqual([c.client_id for c in result.comparators], ["exact", "distant"])
        self.assertGreater(result.comparators[0].similarity, result.comparators[1].similarity)

    def test_nan_profile_values_do_not_become_a_perfect_match(self):
        ds = _dataset([
            _profile(
                "missing",
                employees=np.nan,
                turnover=np.nan,
                funding=np.nan,
                stage=np.nan,
                stage_rank=np.nan,
            ),
            _profile("valid", employees=55, turnover=5_500_000),
        ])

        result = find_comparators(ds, _prospect(), max_n=2)
        scores = {c.client_id: c.similarity for c in result.comparators}

        self.assertGreater(scores["valid"], scores["missing"])
        self.assertLess(scores["missing"], 1.0)

    def test_funding_amount_breaks_an_otherwise_equal_tie(self):
        ds = _dataset([
            _profile("funding-match", funding=5_000_000),
            _profile("funding-distant", funding=500_000_000),
        ])

        result = find_comparators(ds, _prospect(), max_n=2)

        self.assertEqual(result.comparators[0].client_id, "funding-match")
        self.assertGreater(result.comparators[0].similarity, result.comparators[1].similarity)

    def test_max_n_limits_selected_comparators(self):
        ds = _dataset([_profile(f"c{i}", employees=50 + i) for i in range(40)])

        result = find_comparators(ds, _prospect(), max_n=30)

        self.assertEqual(result.pool_size, 40)
        self.assertEqual(len(result.comparators), 30)


class CoverBenchmarkTests(unittest.TestCase):
    def test_active_policy_is_preferred_over_newer_expired_policy(self):
        ds = _dataset(
            [_profile("a")],
            cover_lines=[
                {
                    "client_id": "a",
                    "canonical_cover": COVER,
                    "limit_amount": 1_000_000,
                    "policy_status": "Active",
                    "effective_date": "2024-01-01",
                    "end_date": "2025-01-01",
                },
                {
                    "client_id": "a",
                    "canonical_cover": COVER,
                    "limit_amount": 9_000_000,
                    "policy_status": "Expired",
                    "effective_date": "2025-01-01",
                    "end_date": "2026-01-01",
                },
            ],
        )

        result = benchmark_cover(ds, ["a"], COVER, 500_000)

        self.assertIsNotNone(result)
        self.assertEqual(result.n_peers, 1)
        self.assertEqual(result.median_gbp, 1_000_000)

    def test_latest_effective_year_is_used_when_status_is_equal(self):
        ds = _dataset(
            [_profile("a")],
            cover_lines=[
                {
                    "client_id": "a",
                    "canonical_cover": COVER,
                    "limit_amount": 1_000_000,
                    "policy_status": "Expired",
                    "effective_date": "2022-01-01",
                    "end_date": "2023-01-01",
                },
                {
                    "client_id": "a",
                    "canonical_cover": COVER,
                    "limit_amount": 3_000_000,
                    "policy_status": "Expired",
                    "effective_date": "2024-01-01",
                    "end_date": "2025-01-01",
                },
            ],
        )

        result = benchmark_cover(ds, ["a"], COVER, 500_000)

        self.assertIsNotNone(result)
        self.assertEqual(result.n_peers, 1)
        self.assertEqual(result.median_gbp, 3_000_000)

    def test_highest_limit_breaks_tie_within_same_policy_period(self):
        ds = _dataset(
            [_profile("a")],
            cover_lines=[
                {
                    "client_id": "a",
                    "canonical_cover": COVER,
                    "limit_amount": 1_000_000,
                    "policy_status": "Active",
                    "effective_date": "2025-01-01",
                    "end_date": "2026-01-01",
                },
                {
                    "client_id": "a",
                    "canonical_cover": COVER,
                    "limit_amount": 2_000_000,
                    "policy_status": "Active",
                    "effective_date": "2025-01-01",
                    "end_date": "2026-01-01",
                },
            ],
        )

        result = benchmark_cover(ds, ["a"], COVER, 500_000)

        self.assertIsNotNone(result)
        self.assertEqual(result.n_peers, 1)
        self.assertEqual(result.median_gbp, 2_000_000)

    def test_percentiles_use_one_observation_per_company(self):
        ds = _dataset(
            [_profile("a"), _profile("b")],
            cover_lines=[
                {"client_id": "a", "canonical_cover": COVER, "limit_amount": 1_000_000, "policy_status": "Active"},
                {"client_id": "a", "canonical_cover": COVER, "limit_amount": 2_000_000, "policy_status": "Active"},
                {"client_id": "b", "canonical_cover": COVER, "limit_amount": 4_000_000, "policy_status": "Active"},
            ],
        )

        result = benchmark_cover(ds, ["a", "b"], COVER, 1_000_000)

        self.assertIsNotNone(result)
        self.assertEqual(result.n_peers, 2)
        # The documented representative-value rule is max valid Recorder limit per company.
        self.assertEqual(result.median_gbp, 3_000_000)

    def test_recorder_is_primary_and_deck_is_fallback_per_company(self):
        ds = _dataset(
            [_profile("a"), _profile("b")],
            cover_lines=[
                {"client_id": "a", "canonical_cover": COVER, "limit_amount": 2_000_000, "policy_status": "Active"},
            ],
            deck_benchmarks=[
                # Must not be counted because client a already has primary Recorder data.
                {"client_id": "a", "canonical_cover": COVER, "current_limit_numeric": 9_000_000},
                # Must be used because client b has no Recorder data.
                {"client_id": "b", "canonical_cover": COVER, "current_limit_numeric": 4_000_000},
            ],
        )

        result = benchmark_cover(ds, ["a", "b"], COVER, 1_000_000)

        self.assertIsNotNone(result)
        self.assertEqual(result.n_peers, 2)
        self.assertEqual(result.median_gbp, 3_000_000)

    def test_percentage_above_uses_unique_companies(self):
        ds = _dataset(
            [_profile("a"), _profile("b")],
            cover_lines=[
                {"client_id": "a", "canonical_cover": COVER, "limit_amount": 1_000_000, "policy_status": "Active"},
                {"client_id": "a", "canonical_cover": COVER, "limit_amount": 2_000_000, "policy_status": "Active"},
                {"client_id": "b", "canonical_cover": COVER, "limit_amount": 500_000, "policy_status": "Active"},
            ],
        )

        result = benchmark_cover(ds, ["a", "b"], COVER, 1_000_000)

        self.assertIsNotNone(result)
        self.assertEqual(result.pct_peers_above_prospect, 50.0)

    def test_no_peer_data_is_reported_transparently(self):
        ds = _dataset([_profile("a")])

        result = benchmark_cover(ds, ["a"], COVER, 1_000_000)

        self.assertIsNotNone(result)
        self.assertEqual(result.n_peers, 0)
        self.assertIsNone(result.median_gbp)
        self.assertEqual(result.confidence, "Low")
        self.assertEqual(result.status, "insufficient_peer_data")
        self.assertEqual(result.priority, "For consideration")
        self.assertIn("No comparator companies", result.source_note)

        recommendation = build_recommendation(
            result,
            "FinTech",
            pd.DataFrame(columns=["vertical", "item", "action"]),
        )
        self.assertIn("no comparable FinTech companies", recommendation.deterministic_explanation)
        self.assertIn("data-availability limitation", recommendation.deterministic_explanation)

    def test_status_boundaries_are_deterministic(self):
        self.assertEqual(_classify_status(None, 1, 2, 3), "no_data")
        self.assertEqual(_classify_status(0.5, 1, 2, 3), "significantly_below")
        self.assertEqual(_classify_status(1, 1, 2, 3), "slightly_below")
        self.assertEqual(_classify_status(2, 1, 2, 3), "around")
        self.assertEqual(_classify_status(3, 1, 2, 3), "around")
        self.assertEqual(_classify_status(4, 1, 2, 3), "above")


if __name__ == "__main__":
    unittest.main()
