"""
report.py
Builds a downloadable PDF benchmarking report from the same structured data shown in the
Streamlit UI. All figures are re-used from the deterministic benchmark/recommendation objects -
nothing is recalculated or re-invented here.
"""

from __future__ import annotations

import io
from datetime import date

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
)

DISCLAIMER = (
    "This report provides benchmarking guidance only and does not constitute individual "
    "insurance advice. Please speak with a Capsule broker for personalised advice."
)

_PRIORITY_COLOURS = {
    "Immediate action": "#C0392B",
    "For consideration": "#B7791F",
    "Review at renewal": "#2E7D32",
}


def _fmt_gbp(x):
    if x is None:
        return "Not provided"
    if x >= 1_000_000:
        s = f"£{x/1_000_000:.2f}m".replace(".00m", "m")
        return s
    if x >= 1_000:
        return f"£{x/1_000:.0f}k"
    return f"£{x:,.0f}"


def build_report_pdf(business_profile: dict, comparator_result, recommendations, ai_texts: dict) -> bytes:
    """
    business_profile: dict with vertical/employee_count/turnover_gbp/funding_raised_gbp/funding_series
    comparator_result: benchmark.ComparatorResult
    recommendations: list[recommendation.Recommendation]
    ai_texts: dict cover -> phrased explanation string
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1Capsule", fontSize=20, leading=24, spaceAfter=6, textColor=colors.HexColor("#1A2B4C")))
    styles.add(ParagraphStyle(name="H2Capsule", fontSize=13, leading=16, spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#1A2B4C")))
    styles.add(ParagraphStyle(name="Small", fontSize=8.5, leading=11, textColor=colors.HexColor("#555555")))
    styles.add(ParagraphStyle(name="BodySmall", fontSize=9.5, leading=13))

    story = []

    story.append(Paragraph("Capsule Cover Benchmarking Report", styles["H1Capsule"]))
    story.append(Paragraph(f"Vertical: {business_profile.get('vertical', '—')} &nbsp;|&nbsp; Generated: {date.today().isoformat()}", styles["Small"]))
    story.append(Spacer(1, 10))

    # --- Business profile ---
    story.append(Paragraph("Business Profile", styles["H2Capsule"]))
    profile_rows = [
        ["Vertical", business_profile.get("vertical") or "—"],
        ["Employees", str(business_profile.get("employee_count") or "—")],
        ["Turnover", _fmt_gbp(business_profile.get("turnover_gbp"))],
        ["Funding raised", _fmt_gbp(business_profile.get("funding_raised_gbp"))],
        ["Funding stage", business_profile.get("funding_series") or "—"],
    ]
    t = Table(profile_rows, colWidths=[45 * mm, 110 * mm])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1A2B4C")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#DDDDDD")),
    ]))
    story.append(t)

    # --- Comparator summary ---
    story.append(Paragraph("Peer Comparator Summary", styles["H2Capsule"]))
    story.append(Paragraph(
        f"{len(comparator_result.comparators)} comparator companies matched out of a pool of "
        f"{comparator_result.pool_size} in the same vertical. "
        f"Comparator confidence: <b>{comparator_result.confidence}</b> — {comparator_result.confidence_reason}",
        styles["BodySmall"],
    ))

    # --- Each cover ---
    story.append(Paragraph("Cover-by-Cover Benchmark & Recommendations", styles["H2Capsule"]))

    for rec in recommendations:
        colour_hex = _PRIORITY_COLOURS.get(rec.priority, "#333333")
        header = f'<font color="{colour_hex}"><b>{rec.cover}</b></font>'
        story.append(Paragraph(header, styles["BodySmall"]))

        stat_rows = [
            ["Current cover", _fmt_gbp(rec.current_limit_gbp)],
            ["Peer median", _fmt_gbp(rec.median_gbp)],
            ["Peer 25th–75th percentile", f"{_fmt_gbp(rec.p25_gbp)} – {_fmt_gbp(rec.p75_gbp)}"],
            ["Comparator group", f"{rec.n_peers} companies"],
            ["Confidence", rec.confidence],
            ["Priority", rec.priority],
        ]
        st = Table(stat_rows, colWidths=[45 * mm, 110 * mm])
        st.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(st)

        explanation = ai_texts.get(rec.cover, rec.deterministic_explanation)
        story.append(Paragraph(f"<b>Explanation:</b> {explanation}", styles["BodySmall"]))
        story.append(Paragraph(f"<i>Evidence/source: {rec.source_note}</i>", styles["Small"]))
        story.append(Spacer(1, 8))

    story.append(PageBreak())
    story.append(Paragraph("Disclaimer", styles["H2Capsule"]))
    story.append(Paragraph(DISCLAIMER, styles["BodySmall"]))

    doc.build(story)
    return buf.getvalue()
