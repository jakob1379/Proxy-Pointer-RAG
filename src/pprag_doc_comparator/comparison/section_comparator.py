"""
DocComparator: Section Comparator — LLM Compare + Traffic Light

Compares a Doc 1 section against a Doc 2 section using the enriched
comparison prompt (Prompt C). Produces structured comparison results
with traffic light rating and analysis.

Usage:
    from pprag_doc_comparator.comparison.section_comparator import compare_sections, generate_section_summary
"""
import os
import sys
import re
import logging
import time

from pprag_doc_comparator.config import LLM_MODEL

import google.generativeai as genai


def compare_sections(comparison_prompt, doc1_section, doc2_section, doc1_name="Doc 1", doc2_name="Doc 2", doc_type="document"):
    """
    Passes the matched sections to the Gemini model for strict classification and analysis.

    Args:
        comparison_prompt: The global comparison instructions.
        doc1_section: Dict with {breadcrumb, title, full_text}.
        doc2_section: Dict with {breadcrumb, title, full_text}.
        doc1_name: String name of document 1
        doc2_name: String name of document 2
        doc_type: String type of document (used to adapt vocabulary)

    Returns:
        dict: {rating, analysis, doc2_title, doc2_excerpt}
    """
    doc1_text = doc1_section.get("full_text", "")
    doc2_text = doc2_section.get("full_text", "")
    doc2_title = doc2_section.get("title", "Unknown")

    is_academic = "paper" in doc_type.lower() or "academic" in doc_type.lower()

    if is_academic:
        prompt = f"""{comparison_prompt}

── {doc1_name} SECTION ──
Section: {doc1_section.get('breadcrumb', 'Unknown')}
Content:
{doc1_text}

── {doc2_name} SECTION ──
Section: {doc2_section.get('breadcrumb', 'Unknown')}
Content:
{doc2_text}

Produce your comparison in EXACTLY this format (keep the labels exactly as shown):

SHARED: [List the core concepts, objectives, or methods that both sections share. e.g. "Both use X to optimize Y". Write "None" if completely unrelated.]
RATING: [GREEN or YELLOW or RED or UNRELATED]
ROLE: [Short description of the functional role of the sections being compared, e.g., Experimental Setup, Optimization Loss]
RISK: [Classify the architectural difference, e.g., deterministic vs stochastic, optimization-heavy vs initialization-heavy. Write "N/A" if GREEN or UNRELATED]
ANALYSIS: [2-3 sentence explanation focused on differences and tradeoff implications. Use varied rhetorical framing (causal, comparative, objective, tradeoff). You MUST use the actual document names ("{doc1_name}" and "{doc2_name}"). Write "Sections are aligned." if GREEN]
"""
    else:
        prompt = f"""{comparison_prompt}

── {doc1_name} SECTION ──
Section: {doc1_section.get('breadcrumb', 'Unknown')}
Content:
{doc1_text}

── {doc2_name} SECTION ──
Section: {doc2_section.get('breadcrumb', 'Unknown')}
Content:
{doc2_text}

Produce your comparison in EXACTLY this format (keep the labels exactly as shown):

RATING: [GREEN or YELLOW or RED or UNRELATED]
ROLE: [Short description of the functional role of the sections being compared, e.g., Trigger Definition, Enforcement Mechanism]
RISK: [Who benefits from the difference? e.g., Lender-favorable, Borrower-favorable, Neutral, Operationally stricter for borrower. Write "N/A" if GREEN or UNRELATED]
ANALYSIS: [2-3 sentence explanation focused on PRACTICAL IMPLICATIONS: missing protections, enforcement impact, liability exposure, covenant strictness. You MUST use the actual document names ("{doc1_name}" and "{doc2_name}"). Do NOT use phrases like "scope difference" or "structural misalignment". Write "Sections are aligned." if GREEN]
"""

    try:
        model = genai.GenerativeModel(LLM_MODEL)

        max_retries = 5
        base_delay = 2.0
        response = None
        for attempt in range(max_retries):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.0,
                        max_output_tokens=5000,
                    )
                )
                break
            except Exception as e:
                if "429" in str(e) or "Resource exhausted" in str(e):
                    if attempt == max_retries - 1:
                        raise e
                    delay = base_delay * (2 ** attempt)
                    logging.warning(f"Rate limit hit during generation. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise e

        raw = response.text.strip()
        logging.info(f"  [COMPARE] Raw LLM response ({len(raw)} chars): {raw[:300]}")
        result = _parse_tagged_response(raw)
        logging.info(f"  [COMPARE] Parsed: rating={result['rating']}, analysis={result['analysis'][:80]}...")
        result["doc2_title"] = doc2_title
        return result

    except Exception as e:
        logging.error(f"Comparison failed: {e}")
        return {
            "rating": "UNKNOWN",
            "analysis": f"Comparison failed: {e}",
            "shared": "",
            "role": "",
            "risk": "",
            "doc2_excerpt": "",
            "doc2_title": doc2_title,
        }


def _parse_tagged_response(raw):
    """Parse RATING/EXCERPT/ANALYSIS from tagged text output.
    Handles variations like **RATING:**, bold markers, markdown formatting."""

    # Strip markdown bold from labels
    clean = re.sub(r'\*\*', '', raw)

    # Extract RATING — try labeled first, then scan for keywords
    rating_m = re.search(r'RATING\s*:\s*(RED|YELLOW|GREEN|UNRELATED)', clean, re.IGNORECASE)
    if not rating_m:
        # Fallback: look for traffic light emoji + label
        rating_m = re.search(r'(🔴|RED|🟡|YELLOW|🟢|GREEN|⚪|UNRELATED)', clean)
    if rating_m:
        val = rating_m.group(1).upper()
        if '🔴' in val or 'RED' in val:
            rating = 'RED'
        elif '🟡' in val or 'YELLOW' in val:
            rating = 'YELLOW'
        elif '🟢' in val or 'GREEN' in val:
            rating = 'GREEN'
        elif '⚪' in val or 'UNRELATED' in val:
            rating = 'UNRELATED'
        else:
            rating = 'UNKNOWN'
    else:
        rating = 'UNKNOWN'

    # Extract SHARED (everything between SHARED: and RATING:)
    shared_m = re.search(r'SHARED\s*:\s*(.*?)(?=RATING\s*:|ROLE\s*:)', clean, re.DOTALL | re.IGNORECASE)
    shared = shared_m.group(1).strip() if shared_m else ""

    # Extract ROLE (everything between ROLE: and RISK:)
    role_m = re.search(r'ROLE\s*:\s*(.*?)(?=RISK\s*:|ANALYSIS\s*:)', clean, re.DOTALL | re.IGNORECASE)
    role = role_m.group(1).strip() if role_m else ""

    # Extract RISK (everything between RISK: and ANALYSIS:)
    risk_m = re.search(r'RISK\s*:\s*(.*?)(?=ANALYSIS\s*:)', clean, re.DOTALL | re.IGNORECASE)
    risk = risk_m.group(1).strip() if risk_m else ""

    # Extract ANALYSIS (everything after ANALYSIS:)
    analysis_m = re.search(r'ANALYSIS\s*:\s*(.*)', clean, re.DOTALL | re.IGNORECASE)
    analysis = analysis_m.group(1).strip() if analysis_m else ""

    # Suppress analysis for aligned sections
    if rating == "GREEN":
        analysis = ""
        risk = ""

    return {
        "rating": rating,
        "shared": shared,
        "role": role,
        "risk": risk,
        "analysis": analysis,
        "doc2_excerpt": "", # Handled by report builder directly
    }





def extract_rating(result):
    """
    Extract the rating from a comparison result.
    Returns one of: 'RED', 'YELLOW', 'GREEN', 'UNRELATED', or 'UNKNOWN'.
    """
    if isinstance(result, dict):
        return result.get("rating", "UNKNOWN")
    # Fallback for raw text
    text_upper = str(result).upper()
    if "RED" in text_upper:
        return "RED"
    elif "YELLOW" in text_upper:
        return "YELLOW"
    elif "GREEN" in text_upper:
        return "GREEN"
    elif "UNRELATED" in text_upper:
        return "UNRELATED"
    return "UNKNOWN"
