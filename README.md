# Capsule Cover Benchmarking — Hackathon Prototype (Brief B)

A Streamlit tool that takes a FinTech & Venture prospect's business profile and current
insurance cover, finds comparable companies in Capsule's anonymised dataset, benchmarks their
cover against those peers, and produces evidence-based, priority-ranked recommendations —
framed as guidance, not regulated insurance advice.

**Scope for this build:** one vertical, end-to-end — **FinTech & Venture** — per the brief.

## 1. Setup

```bash
cd capsule-benchmark
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

Unzip the Capsule hackathon dataset so this project folder contains a `dataset/` directory
with `README.md`, `DATA_DICTIONARY.md`, `1_recommendation_bank/`, `2_book_of_business/`,
`3_company_context/`, `4_evaluation/` inside it — i.e.:

```
capsule-benchmark/
  dataset/
    README.md
    DATA_DICTIONARY.md
    2_book_of_business/...
    3_company_context/...
    4_evaluation/...
  app.py
  ...
```

`dataset/` is git-ignored — it will never be committed or uploaded anywhere by this project.

If you'd rather keep the dataset elsewhere, set an env var instead of copying it in:

```bash
export CAPSULE_DATA_ROOT=/path/to/dataset
```

## 2. (Optional) enable AI-phrased explanations

Without an API key the app still works end-to-end — every number is real, and explanations
fall back to a deterministic, template-written paragraph. To get AI-polished prose instead:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# optional: export CAPSULE_AI_MODEL=claude-sonnet-5   (defaults to claude-sonnet-5)
```

To swap to Gemini later (e.g. if that's the approved hackathon model), no app code changes
are needed:

```bash
pip install google-generativeai
export CAPSULE_AI_PROVIDER=gemini
export GEMINI_API_KEY=...
export CAPSULE_AI_MODEL=gemini-1.5-pro
```

## 3. Run

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501). Use the sidebar to optionally
load one of the 6 fictional FinTech synthetic test briefs (`4_evaluation/synthetic_prospect_briefs.csv`)
to try the tool without typing in a real profile.

## How it works

### Comparator matching (`benchmark.py: find_comparators`)
1. Filter Capsule's combined company-profile table (`healthcheck_questionnaire.csv` +
   `hubspot_company_context.csv`, coalesced in `data_loader.py`) to the prospect's vertical.
2. Score every same-vertical company on log-scale closeness of headcount and turnover
   (weight 1.0 each) and total funding raised (weight 0.75), plus funding-stage distance
   (weight 0.75; Pre-Seed → Seed → Series A → B → C…). Missing, non-finite, zero or
   negative numeric values are excluded from the calculation rather than treated as valid
   matches. If no usable comparison factor is available for a company, its similarity is 0.
3. Rank and return the top matches (up to 50). If the whole vertical pool is smaller than
   that, everything in it is returned and comparator confidence is marked accordingly
   (High ≥25 companies, Medium 10–24, Low <10).

### Cover benchmarking (`benchmark.py: benchmark_cover`)
For each cover the prospect specified, peer limits are taken from two dataset sources,
restricted to the comparator set found above:
- `recorder_cover_lines.csv` — clean numeric `limit_amount` per coverage line (primary source)
- `deck_limit_benchmarks.csv` — parsed from text like `£10m` / `£10m - £15m` (secondary source)

Each comparator company contributes at most one representative value per cover, even when it
has several policy years or cover-line records. Recorder records are selected in this order:
active policy first, then the latest effective date, then the latest end date, with the highest
valid limit used only as a deterministic tie-breaker within the same policy period. Expired or
cancelled policies therefore remain usable when no higher-priority record is available.

Recorder is the primary source. A company's parsed deck limit is used only when that company
has no valid Recorder value for the same cover; the two sources are not counted as separate
peers. `n_peers` consequently means the number of **unique comparator companies**, not the
number of policy or cover-line records.

Median / 25th / 75th percentile and the percentage of peers above the prospect are computed
with `numpy` from those unique-company values — **no LLM ever touches these numbers**.
Per-cover confidence is High/Medium/Low based on the number of unique companies contributing
a value (≥10 / 5–9 / <5).

### Cover-name normalisation
Raw cover text is very inconsistent across the dataset (e.g. "Directors' and Officers'
Liability", "Directors and Officer/Corporate Liability", "Directors & Officers" all mean the
same thing). `data_loader.canonicalise_cover()` maps everything to 8 canonical categories that
match what prospects are actually asked about in `healthcheck_questionnaire.csv`'s
`policies_needed` field: Directors & Officers, Cyber, Professional Indemnity, Crime,
Employers' Liability, Public & Products Liability, Business Interruption, Property/Equipment.

**Deliberately not used:** `deck_current_cover.csv`. Inspection showed its `cover`/`limit`
fields are frequently multiple line items concatenated into a single string (e.g. nine
different covers and their limits glued together with no separator) — parsing it reliably
wasn't a good use of hackathon time and risked silently wrong numbers. `deck_recommendations.csv`
is still used, for retrieval-grounding text only, not benchmark numbers.

### Recommendation & priority logic (`recommendation.py`)
Status is classified against the peer distribution: below the 25th percentile →
**significantly below** → *Immediate action*; between p25 and median → **slightly below** →
*For consideration*; between median and p75, or above → **around/above benchmark** →
*Review at renewal*. No current limit recorded → also *Immediate action* (can't verify
adequate cover). Each recommendation also carries up to 2 short evidence snippets pulled from
Capsule's own historical `deck_recommendations.csv` for the same vertical and cover, to ground
the explanation in real past broker guidance.

If no comparator company has a usable value for a cover, the cover remains visible rather than
being silently omitted. It is returned as `insufficient_peer_data`, with Low confidence and
*For consideration* priority. The explanation makes clear that this is a data-availability
limitation, not evidence that the prospect's current limit is too high or too low.

### AI layer (`ai_service.py`)
The AI is only ever given a small JSON object of *already-computed* facts (limit, peer
median/percentiles, n peers, status, priority, confidence, 0–2 evidence snippets) and a system
prompt that explicitly forbids inventing or altering any number. It only rewrites this into
plain English. If no API key is configured, or the call fails for any reason, the app falls
back to a deterministic, template-written explanation — the tool never breaks or blocks on the
AI layer. Swapping providers (e.g. to Gemini) only requires setting env vars — see above.

### Report (`report.py`)
A downloadable PDF (via `reportlab`) reusing the exact same numbers shown in the UI: business
profile, peer/comparator summary, and per-cover benchmark + recommendation + priority +
confidence + evidence/source, closing with the required disclaimer:

> This report provides benchmarking guidance only and does not constitute individual insurance
> advice. Please speak with a Capsule broker for personalised advice.

## Data security

- `dataset/` is git-ignored — never committed, never uploaded.
- Only small, already-aggregated numbers (limits, medians, percentiles, counts, 1–2 short
  historical-guidance snippets already anonymised in source) are sent to the AI provider — never
  raw client rows, never a full comparator list, never prospect-identifying information.
- No dataset content is written to logs by this app.

## Known limitations (honest, for judges)

- Comparator *company* counts comfortably reach 30–50 for FinTech & Venture (43 in the pool).
  After selecting one representative value per company and using deck data only as a fallback,
  comparator counts *per specific cover type* are smaller (currently 4–13 unique companies for
  FinTech). This is a real limitation of the underlying dataset, not the algorithm, and is
  exactly why per-cover confidence is reported separately from comparator-pool confidence.
- Only FinTech & Venture is wired up end-to-end; the other 3 verticals (Consumer, MedTech,
  Tech) use the same dataset shape and the same code, so extending is a matter of adding them
  to `SUPPORTED_VERTICALS` in `app.py`, not new logic.
- `deck_current_cover.csv` is not used as a benchmark source (see above) — a good next task if
  more time is available, with a purpose-built parser for its concatenated fields.

## Project structure

```
capsule-benchmark/
  app.py              Streamlit UI (4 pages)
  data_loader.py       Loads + normalises the dataset, canonicalises cover names
  benchmark.py          Comparator matching + deterministic per-cover statistics
  recommendation.py     Priority classification + evidence retrieval + deterministic prose
  ai_service.py          Modular AI phrasing layer (Claude default, Gemini swap-in)
  report.py            PDF report builder (reportlab)
  requirements.txt
  .gitignore            Excludes dataset/ and secrets
  README.md            This file
```
