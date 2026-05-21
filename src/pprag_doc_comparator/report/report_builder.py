"""
DocComparator: Report Builder

Assembles all comparison results into a formatted Markdown report.

Report format per section:
  - Doc 1 section: Title + ~100 word excerpt
  - Doc 2 sections: List of relevant titles
  - For each Doc 2 match:
      - Title + ~100 word excerpt
      - Discrepancy traffic light
      - Analysis (only if not GREEN)
  - Section summary at the end

Usage:
    from pprag_doc_comparator.report.report_builder import build_report
"""
import datetime


RATING_ICONS = {
    "RED": "🔴",
    "YELLOW": "🟡",
    "GREEN": "🟢",
    "UNRELATED": "⚪",
    "UNKNOWN": "⚠️",
}

RATING_LABELS = {
    "RED": "SIGNIFICANT DISCREPANCY",
    "YELLOW": "MODERATE DIFFERENCE",
    "GREEN": "ALIGNED",
    "UNRELATED": "FUNCTIONALLY DISTINCT / NO DIRECT ANALOG FOUND",
    "UNKNOWN": "UNABLE TO DETERMINE",
}


def build_executive_summary(criteria, doc1_name, doc2_name, doc_type,
                             total_sections, total_comparisons, ratings):
    """
    Build the executive summary section with aggregate stats.
    """
    red = ratings.get("RED", 0)
    yellow = ratings.get("YELLOW", 0)
    green = ratings.get("GREEN", 0)
    unrelated = ratings.get("UNRELATED", 0)
    unknown = ratings.get("UNKNOWN", 0)

    total = red + yellow + green + unrelated + unknown
    alignment = (green / total * 100) if total > 0 else 0

    return f"""# {doc_type.capitalize()} comparison report
#### Powered by Proxy-Pointer

**Generated:** {datetime.date.today().isoformat()}<br>
**Criteria:** {criteria}<br>
**Document 1:** {doc1_name}<br>
**Document 2:** {doc2_name}<br>
**Document Type:** {doc_type}

---

## Executive Summary

<table style="max-width: 600px; border-collapse: collapse; border: 1px solid #334155;">
  <thead>
    <tr style="background-color: #f8fafc;">
      <th style="border: 1px solid #cbd5e1; padding: 12px; text-align: left;">Metric</th>
      <th style="border: 1px solid #cbd5e1; padding: 12px; text-align: left;">Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border: 1px solid #cbd5e1; padding: 12px;">{doc1_name} Sections Analyzed</td>
      <td style="border: 1px solid #cbd5e1; padding: 12px;">{total_sections}</td>
    </tr>
    <tr>
      <td style="border: 1px solid #cbd5e1; padding: 12px;">Comparisons Performed</td>
      <td style="border: 1px solid #cbd5e1; padding: 12px;">{total_comparisons}</td>
    </tr>
    <tr>
      <td style="border: 1px solid #cbd5e1; padding: 12px;">🔴 Significant Discrepancies</td>
      <td style="border: 1px solid #cbd5e1; padding: 12px;">{red}</td>
    </tr>
    <tr>
      <td style="border: 1px solid #cbd5e1; padding: 12px;">🟡 Moderate Differences</td>
      <td style="border: 1px solid #cbd5e1; padding: 12px;">{yellow}</td>
    </tr>
    <tr>
      <td style="border: 1px solid #cbd5e1; padding: 12px;">🟢 Aligned Sections</td>
      <td style="border: 1px solid #cbd5e1; padding: 12px;">{green}</td>
    </tr>
  </tbody>
</table>

---

## Detailed Comparison

"""


def build_section_block(section_num, doc1_section, doc2_matches, comparison_results, criteria, doc1_name="Doc 1", doc2_name="Doc 2", doc_type="document"):
    """
    Build one complete section comparison block.

    Args:
        section_num: int — sequential section number.
        doc1_section: dict with {title, breadcrumb, full_text}.
        doc2_matches: list of doc2 section dicts.
        comparison_results: list of comparison result dicts.
        criteria: str — comparison criteria for context.
        doc1_name: str — name of document 1
        doc2_name: str — name of document 2
        doc_type: str — type of document being compared

    Returns:
        str: Formatted markdown block for this section.
    """
    title = doc1_section.get("title", "Unknown")
    doc1_text = doc1_section.get("full_text", "")
    is_academic = "paper" in doc_type.lower() or "academic" in doc_type.lower()

    # Doc 1 excerpt (~50 words)
    doc1_excerpt = _excerpt(doc1_text, 50)

    # Section header with Doc 1 info (Standard Markdown H3 heading to prevent overlapping numbers)
    md = f"### Comparison #{section_num} | {title}\n\n"
    md += f"> **{doc1_name} excerpt:** {doc1_excerpt}\n\n"

    # List of matching Doc 2 section titles
    doc2_titles = [r.get("doc2_title", "?") for r in comparison_results]
    md += f"**Matching {doc2_name} sections:** {', '.join(doc2_titles)}\n\n"
    md += f"**Criteria:** *{criteria}*\n\n"

    # Each Doc 2 comparison
    for i, result in enumerate(comparison_results):
        rating = result.get("rating", "UNKNOWN")
        icon = RATING_ICONS.get(rating, "⚠️")
        label = RATING_LABELS.get(rating, "UNKNOWN")
        doc2_title = result.get("doc2_title", "Unknown")
        # Extract exactly 50 words directly from the source text
        raw_text = doc2_matches[i].get("full_text", "") if i < len(doc2_matches) else ""
        doc2_excerpt = _excerpt(raw_text, 50)
        analysis = result.get("analysis", "")

        # Doc 2 header (Standard Markdown H4 heading with a distinct sub-level arrow)
        md += f"#### ↳ {doc2_title}\n\n"
        md += f"> **{doc2_name} excerpt:** {doc2_excerpt}\n\n" if doc2_excerpt else ""
        md += f"**Discrepancy:** {icon} {label}\n\n"

        role = result.get("role", "")
        if role:
            md += f"**Role:** {role}\n\n"

        shared = result.get("shared", "")
        if shared and shared.upper() != "NONE" and is_academic:
            md += f"**Shared Concepts:** 🤝 *{shared}*\n\n"

        risk = result.get("risk", "")
        if risk and risk.upper() != "N/A" and rating not in ("GREEN", "UNRELATED"):
            if is_academic:
                md += f"**Key Difference:** 📐 *{risk}*\n\n"
            else:
                md += f"**Risk Direction:** ⚖️ *{risk}*\n\n"

        if rating != "GREEN" and analysis:
            md += f"**Analysis:** {analysis}\n\n"

    # Thin divider between Doc 1 sections
    md += "---\n\n"
    return md


def assemble_report(executive_summary, section_blocks):
    """
    Assemble the final report from all component parts.
    """
    report = executive_summary
    report += "\n".join(section_blocks)
    return report


def _excerpt(text, word_count=100):
    """Extract roughly word_count words from text, stripping ATX heading prefix, HTML tags, and newlines."""
    import re
    # Strip leading ATX heading (## SECTION 1.01. Title)
    clean = re.sub(r'^#{1,6}\s+', '', text.strip())
    # Strip HTML tags (like <u> or </u>)
    clean = re.sub(r'<[^>]+>', '', clean)
    # Replace newlines with spaces so it doesn't break markdown blockquotes
    clean = re.sub(r'\s+', ' ', clean)

    words = clean.split()
    if len(words) <= word_count:
        return clean.strip()
    return " ".join(words[:word_count]) + "..."
