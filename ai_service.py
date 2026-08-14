"""
ai_service.py
The only module that talks to an external AI model. Its job is narrow and strict:
turn already-computed, deterministic facts into clear English. It must never invent
benchmark numbers, comparator counts, limits, historical recommendations or sources -
those are all passed in as structured context and the system prompt forbids adding to them.

Modular by design: swap providers by setting CAPSULE_AI_PROVIDER=anthropic|gemini|none.
`none` (or a missing API key) falls back to the deterministic explanation already produced
by recommendation.py, so the app still works end-to-end without any AI key configured.

Data-security note: only the minimum derived numbers/strings needed for phrasing are sent
(cover name, limit figures, peer stats, priority, a couple of short evidence snippets that are
already anonymised in the source dataset). No raw client rows, no full comparator lists, and no
prospect-identifying information are sent to the model.
"""

from __future__ import annotations

import os
from typing import Optional

SYSTEM_PROMPT = """You are a writing assistant for an insurance benchmarking report.
You will be given a JSON object of ALREADY-CALCULATED facts (limits, peer statistics,
priority, confidence, and optional short evidence quotes from historical broker notes).

Rules you must follow exactly:
- Do NOT invent, adjust, round differently, or restate any number that is not already in the
  JSON you were given. Use the numbers exactly as provided (formatting/currency wording is fine).
- Do NOT invent comparator counts, company names, sources, or historical recommendations.
- Do NOT give individual insurance advice or tell the reader what to buy - explain the
  benchmarking picture and why the priority label was assigned, and suggest discussing it
  with a Capsule broker if relevant.
- Keep it to 2-4 concise sentences, professional but plain English, no bullet points.
- If evidence_snippets are provided, you may reference that "Capsule's historical guidance for
  similar businesses" raised a similar point, but do not fabricate a quote - paraphrase only
  the snippet(s) given.
"""


def _build_user_prompt(context: dict) -> str:
    import json
    return (
        "Write the explanation paragraph for this benchmarked cover, using ONLY the facts below:\n\n"
        + json.dumps(context, indent=2, default=str)
    )


def _call_anthropic(context: dict, model: str) -> Optional[str]:
    try:
        import anthropic
    except ImportError:
        return None
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(context)}],
    )
    parts = [b.text for b in resp.content if getattr(
        b, "type", None) == "text"]
    return "\n".join(parts).strip() or None


def _call_gemini(context: dict, model: str) -> Optional[str]:
    """Native Gemini SDK path - only relevant if a raw Google API key is ever issued
    instead of the event's OpenAI-compatible proxy key. Not what this event uses."""
    try:
        import google.generativeai as genai
    except ImportError:
        return None
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    gmodel = genai.GenerativeModel(model, system_instruction=SYSTEM_PROMPT)
    resp = gmodel.generate_content(_build_user_prompt(context))
    return (resp.text or "").strip() or None


def _call_litellm(context: dict, model: str) -> Optional[str]:
    """The event's actual issued key: an OpenAI-compatible endpoint (litellm proxy)
    in front of Gemini models. Base URL is fixed for the event; only the API key is
    secret and comes from the environment, never hardcoded here."""
    try:
        from openai import OpenAI
    except ImportError:
        return None
    api_key = os.environ.get("LITELLM_API_KEY")
    if not api_key:
        return None
    base_url = os.environ.get(
        "LITELLM_BASE_URL", "https://litellm.perceptura.com")
    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(context)},
        ],
        max_tokens=300,
    )
    text = resp.choices[0].message.content
    return (text or "").strip() or None


_PROVIDERS = {
    "anthropic": (_call_anthropic, os.environ.get("CAPSULE_AI_MODEL", "claude-sonnet-5")),
    "gemini": (_call_gemini, os.environ.get("CAPSULE_AI_MODEL", "gemini-1.5-pro")),
    "litellm": (_call_litellm, os.environ.get("CAPSULE_AI_MODEL", "gemini-3.6-flash")),
}


def phrase_recommendation(recommendation) -> str:
    """recommendation: a recommendation.Recommendation instance.
    Returns AI-phrased prose, or the deterministic fallback explanation if no provider/key
    is configured or the call fails for any reason."""

    provider_name = os.environ.get("CAPSULE_AI_PROVIDER", "anthropic").lower()
    fallback = recommendation.deterministic_explanation

    if provider_name not in _PROVIDERS:
        return fallback

    context = {
        "cover": recommendation.cover,
        "current_limit_gbp": recommendation.current_limit_gbp,
        "peer_median_gbp": recommendation.median_gbp,
        "peer_25th_percentile_gbp": recommendation.p25_gbp,
        "peer_75th_percentile_gbp": recommendation.p75_gbp,
        "n_comparable_peers_with_this_cover": recommendation.n_peers,
        "status": recommendation.status,
        "priority": recommendation.priority,
        "confidence": recommendation.confidence,
        "confidence_reason": recommendation.confidence_reason,
        "evidence_snippets": recommendation.evidence_snippets,
    }

    fn, model = _PROVIDERS[provider_name]
    try:
        result = fn(context, model)
    except Exception:
        result = None
    return result or fallback
