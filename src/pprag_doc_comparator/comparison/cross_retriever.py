"""
DocComparator: Cross-Document Retriever

For each Doc 1 section, retrieves matching sections from Doc 2
using the full Proxy-Pointer 2-stage retrieval pipeline:
  Stage 1: Broad vector recall with doc_id filtering
  Stage 2: LLM re-ranker selects top-k by structural relevance

Algorithm:
  1. Build query: [criteria] [breadcrumb] + section_text[:500]
  2. Vector search shared index, native filter to doc2_id
  3. Deduplicate by node_id (keep first = highest similarity)
  4. LLM re-ranker picks top k_final from deduplicated candidates
  5. Load full section text from .md file

Usage:
    from pprag_doc_comparator.comparison.cross_retriever import retrieve_matching_sections
"""
import os
import sys
import re
import logging
import time

from pprag_doc_comparator.config import DOCUMENTS_DIR, LLM_MODEL
from pprag_doc_comparator.validation.criteria_validator import build_cross_reranker_prompt
from pprag_doc_comparator.comparison.section_selector import resolve_md_path_for_doc_id

import google.generativeai as genai
import typing
import json

class RankingResponse(typing.TypedDict):
    ranked_indices: list[int]


def _find_md_path_for_doc_id(doc_id, data_dir=None):
    """Find the .md file path that corresponds to a doc_id."""
    return resolve_md_path_for_doc_id(doc_id, data_dir)


def retrieve_matching_sections(vector_db, doc2_id, cross_query,
                                doc1_breadcrumb, criteria, doc_type,
                                k_final=3, data_dir=None):
    """
    For a given Doc 1 section, find the most relevant Doc 2 sections
    using the full PP 2-stage retrieval pipeline.

    Args:
        vector_db: Loaded FAISS index.
        doc2_id: The doc_id of Document 2 to filter results to.
        cross_query: Prompt B — [criteria] [breadcrumb] + section_text.
        doc1_breadcrumb: Breadcrumb of the Doc 1 section being matched.
        criteria: The user's comparison criteria (for re-ranker context).
        k_final: Max unique sections to return after re-ranking.
        data_dir: Path to documents directory.

    Returns:
        List of dicts: [{node_id, breadcrumb, title, start_line, end_line, full_text}, ...]
    """
    if data_dir is None:
        data_dir = str(DOCUMENTS_DIR)

    # ── Stage 1: Broad vector recall with native metadata filtering ──
    # fetch_k must be > k because FAISS post-filters by metadata.
    doc2_results = vector_db.similarity_search(
        cross_query,
        k=50,
        filter={"doc_id": doc2_id},
        fetch_k=200
    )

    if not doc2_results:
        return []

    # Deduplicate by node_id, keep first occurrence (highest similarity)
    candidates = []
    seen_nodes = set()

    for doc in doc2_results:
        node_id = doc.metadata.get("node_id")
        if node_id in seen_nodes:
            continue
        seen_nodes.add(node_id)
        candidates.append({
            "node_id": node_id,
            "breadcrumb": doc.metadata.get("breadcrumb", "Unknown Path"),
            "title": doc.metadata.get("title", ""),
            "start_line": int(doc.metadata.get("start_line", 0)),
            "end_line": int(doc.metadata.get("end_line", 0)),
            "content": doc.page_content,
        })

    if not candidates:
        return []

    # If we have k_final or fewer candidates, skip re-ranking
    if len(candidates) <= k_final:
        selected = candidates
    else:
        # ── Stage 2: LLM Re-Ranker ──
        selected = _rerank_candidates(candidates, doc1_breadcrumb, criteria, doc_type, k_final)

    # Load full section text for selected candidates
    doc2_md_path = _find_md_path_for_doc_id(doc2_id, data_dir)
    doc2_lines = None
    if doc2_md_path:
        with open(doc2_md_path, "r", encoding="utf-8") as f:
            doc2_lines = f.readlines()

    results = []
    for section in selected:
        full_text = ""
        if doc2_lines:
            full_text = "".join(
                doc2_lines[section["start_line"]:section["end_line"]]
            ).strip()

        if not full_text:
            full_text = section["content"]

        results.append({
            "node_id": section["node_id"],
            "breadcrumb": section["breadcrumb"],
            "title": section["title"],
            "start_line": section["start_line"],
            "end_line": section["end_line"],
            "full_text": full_text,
        })

    return results


def _rerank_candidates(candidates, doc1_breadcrumb, criteria, doc_type, k_final):
    """
    LLM re-ranker: given deduplicated Doc 2 candidates, select the top-k
    that are most relevant to the Doc 1 section and comparison criteria.
    Falls back to raw similarity order on failure.
    """
    index_map = {str(i): c for i, c in enumerate(candidates[:50])}
    candidates_text = ""
    for i, c in enumerate(candidates[:50]):
        candidates_text += f"{i}. [{c['breadcrumb']}] (node: {c['node_id']})\n"

    prompt = build_cross_reranker_prompt(criteria, doc_type, doc1_breadcrumb, candidates_text, k_final)

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
                        temperature=0.1,
                        max_output_tokens=2048,
                        response_mime_type="application/json",
                        response_schema=RankingResponse
                    )
                )
                break
            except Exception as e:
                if "429" in str(e) or "Resource exhausted" in str(e):
                    if attempt == max_retries - 1:
                        raise e
                    delay = base_delay * (2 ** attempt)
                    logging.warning(f"Rate limit hit during cross-retrieval. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise e

        response_text = response.text.strip()
        logging.info(f"  [CROSS-RERANKER] Raw response: {response_text[:200]}")

        ranked_ids = []
        try:
            ranking_data = json.loads(response_text)
            ranked_ids = [str(idx) for idx in ranking_data.get("ranked_indices", [])]
        except (json.JSONDecodeError, TypeError):
            # Powerful regex fallback for unformatted verbose responses
            potential_nums = re.findall(r'\d+', response_text)
            ranked_ids = [num for num in potential_nums if num in index_map]

        final_sections = []
        seen = set()
        for rid in ranked_ids:
            if rid in index_map and rid not in seen:
                final_sections.append(index_map[rid])
                seen.add(rid)
            if len(final_sections) >= k_final:
                break

        # Authoritative Return: Respect the reranker's choice, even if it is empty ([])
        return final_sections

    except Exception as e:
        logging.warning(f"Cross-retrieval re-ranker failed ({e}). Falling back to top-{k_final}.")

    # Fallback: top-k by raw similarity
    return candidates[:k_final]
