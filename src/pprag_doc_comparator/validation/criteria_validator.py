"""
DocComparator: Criteria Validator + Prompt Construction

Iterative LLM-based feasibility check for comparison criteria.
Builds three stage-specific prompts once criteria is accepted.

Usage:
    from pprag_doc_comparator.validation.criteria_validator import validate_criteria, build_prompts
"""
import os
import sys
import json
import time
import logging
import typing

class ValidationResponse(typing.TypedDict):
    feasible: bool
    reason: str
    suggested_criteria: list[str]
    document_type_detected: str

from pprag_doc_comparator.config import LLM_MODEL

import google.generativeai as genai


def _get_structure_summary(tree_path):
    """
    Extract a flattened list of titles from the tree to give the LLM
    a full bird's-eye view of the document's scope.
    """
    with open(tree_path, "r", encoding="utf-8") as f:
        tree_data = json.load(f)

    titles = []

    def walk_tree(node_list):
        for node in node_list:
            titles.append(node.get("title", "Untitled"))
            if "nodes" in node and node["nodes"]:
                walk_tree(node["nodes"])

    walk_tree(tree_data.get("structure", []))
    return "\n".join(f"  - {t}" for t in titles)


def validate_criteria(criteria, doc1_tree_path, doc1_md_path,
                      doc2_tree_path, doc2_md_path):
    """
    LLM-based feasibility check for comparison criteria.

    Returns:
        {
          "feasible": bool,
          "reason": str,
          "suggested_criteria": [str, ...],   # only if not feasible
          "document_type_detected": str
        }
    """
    doc1_text = _get_structure_summary(doc1_tree_path)
    doc2_text = _get_structure_summary(doc2_tree_path)

    doc1_name = os.path.splitext(os.path.basename(doc1_md_path))[0]
    doc2_name = os.path.splitext(os.path.basename(doc2_md_path))[0]

    prompt = f"""You are a document analysis specialist.

I have two documents. Below are the full section titles (Table of Contents) for each:
  Doc 1: "{doc1_name}" — section titles:
{doc1_text}

  Doc 2: "{doc2_name}" — section titles:
{doc2_text}

The user wants to compare them using this criteria: "{criteria}"

TASKS:
1. Is this criteria feasible and meaningful for these documents?
   - Answer YES or NO.
2. If NO: Explain briefly why, and suggest 1-2 alternative criteria that WOULD be meaningful.
3. If YES: Return the criteria unchanged.
4. Determine the document type. Be precise. If section titles contain words like "Credit Agreement", "Guarantees", "Lender", or "Borrower", classify as "legal contract". Choose one label: "research paper", "legal contract", "financial report", etc.

RESPONSE FORMAT (JSON only, no markdown fencing):
{{
  "feasible": true or false,
  "reason": "brief explanation",
  "suggested_criteria": ["alt 1", "alt 2"],
  "document_type_detected": "e.g. research paper, legal contract, financial report, technical manual"
}}
"""

    model = genai.GenerativeModel(LLM_MODEL)

    max_retries = 5
    base_delay = 2.0
    response = None
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=2048,
                    response_mime_type="application/json",
                    response_schema=ValidationResponse
                )
            )
            break
        except Exception as e:
            if "429" in str(e) or "Resource exhausted" in str(e):
                if attempt == max_retries - 1:
                    logging.error("Criteria validation failed after retries: %s", e)
                    response = None
                    break
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
            else:
                logging.error("Criteria validation request failed: %s", e)
                response = None
                break

    if response is None:
        return {
            "feasible": False,
            "reason": "Criteria validation is temporarily unavailable. Please retry validation before proceeding.",
            "suggested_criteria": [],
            "document_type_detected": "document",
        }

    text = response.text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]).strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        logging.error(f"Failed to parse criteria validation response: {text}")

        result = {
            "feasible": True,
            "reason": "Could not validate — proceeding with user criteria.",
            "suggested_criteria": [],
            "document_type_detected": "document"
        }

    # If we still landed on "document", enforce an explicit halt and warn the user
    detected = result.get("document_type_detected", "document").lower()
    if detected in ["document", "other", ""] or "unknown" in detected:
        result["feasible"] = False
        result["reason"] = "The system could not confidently detect the document type. Please ensure you are uploading valid technical papers or legal agreements, and re-submit your request."
        result["suggested_criteria"] = []

    logging.info(f"  [VALIDATOR] Detected type: '{result.get('document_type_detected', 'unknown')}' (Feasible: {result.get('feasible')})")
    return result


# ── Prompt Construction ─────────────────────────────────────────────────

def build_section_selection_query(criteria, doc_type):
    """
    Prompt A: Query for PP retrieval Stage 1 (vector search on Doc 1).
    Kept concise — embeddings respond to semantic content, not instructions.
    """
    return f"{doc_type}: {criteria}"


def build_selector_prompt(criteria, doc_type, candidates_text, k_final):
    """Prompt for the LLM re-ranker in PP retrieval Stage 2 (Doc 1 selection)."""
    is_academic = "paper" in doc_type.lower() or "academic" in doc_type.lower()

    base_prompt = f"""You are a senior {doc_type} analyst. Rank sections by relevance to: {criteria}

Select UP TO {k_final} candidates from the list below that are most relevant
to the comparison criteria. Only select sections that are clearly relevant.
If fewer than {k_final} are relevant, only list those.

CANDIDATE LIST (INDEX | Path):
{candidates_text}

RULES:
"""
    if is_academic:
        base_prompt += """1. Broadly select sections that provide context, motivation, methodology, evaluation, or theoretical background related to the criteria.
2. Err on the side of inclusion—if a section touches upon the concepts, include it to provide a rich, comprehensive comparison.
"""
    else:
        base_prompt += """1. Select sections with definitions, terms, obligations, risks, liabilities
2. Include payment terms, financial schedules, default clauses if relevant
"""

    base_prompt += f"""3. Each INDEX must appear ONLY ONCE
4. CRITICAL: Output ONLY a comma-separated list of numeric INDEX numbers. No text, no explanations, no section names.

Output format example: 3, 7, 12, 0, 25
"""
    return base_prompt

def build_cross_reranker_prompt(criteria, doc_type, doc1_breadcrumb, candidates_text, k_final):
    """Prompt for the LLM re-ranker in cross-retrieval Stage 2 (Doc 2 matching)."""
    is_academic = "paper" in doc_type.lower() or "academic" in doc_type.lower()

    base_prompt = f"""You are a structural re-ranker for cross-document comparison.

You are finding the Doc 2 sections that best match a specific Doc 1 section
for comparison purposes.

Doc 1 section being matched: "{doc1_breadcrumb}"
Comparison criteria: "{criteria}"

CANDIDATE Doc 2 SECTIONS (INDEX | Full Path):
{candidates_text}

RANKING RULES:
"""
    if is_academic:
        base_prompt += """1. THEMATIC RELEVANCE: Select Doc 2 sections that discuss similar concepts, methodologies, or related context to the Doc 1 section.
   - It is acceptable to match "Methodology" with "Related Work" or "Introduction" if they discuss the same theoretical concepts.
2. Err on the side of inclusion to ensure a rich, comprehensive comparison. If a section provides useful context, include it.
"""
    else:
        base_prompt += """1. STRICT FUNCTIONAL EQUIVALENCE: Only select Doc 2 sections that serve the SAME legal/commercial function as the Doc 1 section. For example:
   - "Negative Pledge" should match "Liens" or "Negative Pledge" — NOT "Leverage Ratio" or "Pricing Schedule"
   - "Company Guarantee" should match "Guaranty" — NOT "Release Mechanics" or "Conditions Precedent"
   - "Events of Default" should match "Events of Default" — NOT "Applicable Rate"
2. Do NOT match operative covenants with disclosure schedules, pricing grids, or compliance worksheets
"""

    base_prompt += f"""3. If no Doc 2 section is a true functional equivalent, return FEWER than {k_final} results or an empty list
4. Each INDEX must appear ONLY ONCE
5. Output ONLY a comma-separated list of the Top {k_final} unique numeric indices (or fewer)

Output Example: 3, 7, 12
"""
    return base_prompt


def build_cross_retrieval_query(doc1_breadcrumb, doc1_section_text, criteria):
    """
    Prompt B: Query for vector search on Doc 2's chunks.
    Combines Doc 1 section context + criteria for targeted retrieval.
    """
    return f"[{criteria}] [{doc1_breadcrumb}]\n{doc1_section_text}"


def build_comparison_prompt(criteria, doc_type, doc1_name, doc2_name):
    """
    Prompt C: Master prompt for section comparison (§3.7).
    Prepended to each (Doc1 section, Doc2 section) LLM call.
    """
    is_academic = "paper" in doc_type.lower() or "academic" in doc_type.lower()

    if is_academic:
        return f"""You are a senior {doc_type} analyst specializing in {criteria}.
You write for experienced researchers who need actionable technical and methodological analysis, not structural descriptions.

You are comparing sections from two documents:
  - Document 1: "{doc1_name}"
  - Document 2: "{doc2_name}"

── ANALYSIS PHILOSOPHY ──
1. SIMILARITY BEFORE DISCREPANCY:
   - Before grading a difference, you MUST explicitly identify what these sections share (e.g., "Both frameworks use differentiable rasterization and gradient-based refinement...").
   - Derive severity based on the RATIO of overlap vs. divergence, not merely the existence of divergence.

2. RHETORICAL DIVERSITY:
   - Do NOT use identical sentence structures repeatedly.
   - Use varied rhetorical framing: causal framing ("Because Document 1 uses X, it requires Y..."), comparative framing ("While Document 1 focuses on X, Document 2 shifts to Y..."), objective framing ("Document 1 optimizes X..."), tradeoff framing ("Document 1 trades flexibility for fidelity by...").

── SEVERITY CLASSIFICATION ──

🟢 GREEN (ALIGNED):
  Use when the methodologies, algorithms, or definitions are identical or functionally equivalent. (same -> green)

🟡 YELLOW (PARTIAL ALIGNMENT / MODERATE DIFFERENCE):
  This is the most common category for academic comparisons.

  *** INTERMEDIATE REASONING LAYER ***
  Before assigning RED, ask yourself:
  - Are they solving the same core problem?
  - Do they share an optimization framework?
  - Do they share primitives or rendering pipelines?
  - Is the difference mainly: initialization strategy, loss formulation, conditioning signal, or optimization heuristic?

  If YES to any of the above, you MUST assign YELLOW.

  *** SHARED-FOUNDATION DAMPENER ***
  If your "SHARED:" section above identifies that both papers use the same core stack (e.g. "Both use DiffVG", "Both optimize Bézier primitives", "Both use gradient-based refinement"), you MUST heavily bias toward YELLOW.
  Differences in implementation (same core + different implementation) or priors (same objective + different priors) are YELLOW.

🔴 RED (SIGNIFICANT DISCREPANCY):
  Use ONLY for fundamentally different paradigms and structural opposition.
  Reserve RED strictly for:
  - contradictory assumptions,
  - incompatible objectives,
  - fundamentally different pipelines,
  - opposing architectural philosophy,
  - different problem framing altogether.

⚪ UNRELATED:
  Use ONLY when the sections are completely off-topic relative to "{criteria}".
"""
    else:
        return f"""You are a senior {doc_type} analyst specializing in {criteria}.
You write for experienced legal professionals who need actionable risk analysis, not structural descriptions.

You are comparing sections from two documents:
  - Document 1: "{doc1_name}"
  - Document 2: "{doc2_name}"

── ANALYSIS PHILOSOPHY ──
Focus on PRACTICAL IMPLICATIONS, not structural labels:
  - Do NOT use filler phrases like "scope difference", "structural misalignment", or "different document artifacts"
  - Instead, identify: missing protections, covenant strictness, borrower vs. lender friendliness, enforcement implications, liability exposure
  - For every difference, state the RISK DIRECTION: who benefits (lender-favorable, borrower-favorable, operationally stricter, etc.)
  - Example of BAD analysis: "TRoadhouse includes a good faith projection safe harbor."
  - Example of GOOD analysis: "TRoadhouse shields the borrower from liability for inaccurate forward-looking disclosures absent bad faith, reducing lender remedies tied to projection errors. This is borrower-favorable."

── SEVERITY CLASSIFICATION ──

🟢 GREEN (ALIGNED):
  Use when the clauses serve substantially the same legal/commercial function and the borrower/lender risk allocation is materially similar. Minor variations in notice periods, wording, formatting, definitions, or administrative mechanics should remain GREEN if the practical effect is substantially equivalent.

🟡 YELLOW (MODERATE DIFFERENCE):
  Use when the clauses address the same general legal topic but differ in scope, thresholds, carve-outs, mechanics, timing, or operational burden — AND the difference could reasonably affect administration, compliance, monitoring, negotiations, or litigation posture — BUT does not fundamentally change the core economic bargain or enforcement framework.
  This category represents: "same functional category, meaningfully different implementation."
  Typical triggers: different materiality thresholds, different reporting deadlines, narrower/broader carve-outs, different cure periods, additional notice obligations, differing operational covenants, different calculation methodologies that do not radically alter economics.

🔴 RED (SIGNIFICANT DISCREPANCY):
  Use ONLY when the clauses create a materially different legal, economic, collateral, enforcement, or credit-risk outcome. A discrepancy is significant if one agreement: grants a major protection the other lacks, materially shifts lender/borrower risk, materially changes remedies or guarantee/collateral coverage, materially changes covenant enforceability, materially alters default exposure, or creates/removes major structural protections.
  Do NOT classify as RED merely because the wording is broader, one clause is more detailed, or the sections discuss related but distinct concepts.

⚪ UNRELATED:
  Use ONLY when the sections are completely off-topic relative to "{criteria}" (e.g., "Service of Process" matched with "Collateral Structure") or are purely administrative boilerplate (e.g., counterparts, governing law, jury waiver) with zero commercial relevance to the criteria. If the sections are even tangentially related to the criteria topic, compare them as GREEN/YELLOW/RED instead.

── GUARDRAILS ──
  - When uncertain between YELLOW and RED, default to YELLOW.
  - Do NOT infer that a protection is absent from the full agreement solely because it is absent from the retrieved sections. However, you should speak definitively about the text provided.
  - Before assigning RED, ask: Does this difference materially change legal/economic risk? Would counsel negotiate this point? Could it affect recoveries, enforcement rights, pricing, or collateral coverage?
"""
