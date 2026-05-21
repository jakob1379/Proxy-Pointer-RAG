"""
DocComparator: Section Selector — PP 2-Stage Retrieval for Doc 1

Identifies which Doc 1 sections are relevant to the comparison criteria
using the Proxy-Pointer retrieval pipeline:
  Stage 1: Broad vector recall (k=200), filtered to doc1_id
  Stage 2: LLM re-ranker selects top N sections by hierarchical path

Usage:
    from pprag_doc_comparator.comparison.section_selector import select_relevant_sections
"""
import os
import re
import sys
import json
import logging
import time
import hashlib
from functools import lru_cache
from pathlib import Path

from pprag_doc_comparator.config import LLM_MODEL, DOCUMENTS_DIR
from pprag_doc_comparator.validation.criteria_validator import build_selector_prompt

import google.generativeai as genai
import typing

class RankingResponse(typing.TypedDict):
    ranked_indices: list[int]


def select_relevant_sections(vector_db, doc_id, criteria_query,
                              criteria, doc_type, k_search=200, k_final=10):
    """
    PP 2-stage retrieval to find Doc 1 sections relevant to the criteria.

    Args:
        vector_db: Loaded FAISS index.
        doc_id: The doc_id to filter results to.
        criteria_query: Prompt A — semantic query for vector search.
        criteria: User comparison criteria.
        doc_type: Detected document type for dynamic persona/rules.
        k_search: Number of candidates for broad recall.
        k_final: Number of final sections to return.

    Returns:
        List of dicts: [{node_id, breadcrumb, title, doc_id, start_line, end_line, content}, ...]
    """
    # Stage 1: Broad vector recall with native metadata filtering
    # fetch_k must be >> k because FAISS fetches first, then filters by metadata.
    doc_results = vector_db.similarity_search(
        criteria_query,
        k=k_search,
        filter={"doc_id": doc_id},
        fetch_k=1000
    )

    if not doc_results:
        logging.warning(f"No results found for doc_id={doc_id}")
        return []

    # Deduplicate by node_id (keep first occurrence = highest similarity)
    candidates = []
    seen_nodes = set()
    for doc in doc_results:
        node_id = doc.metadata.get("node_id")
        if node_id not in seen_nodes:
            seen_nodes.add(node_id)
            candidates.append({
                "node_id": node_id,
                "breadcrumb": doc.metadata.get("breadcrumb", "Unknown Path"),
                "title": doc.metadata.get("title", ""),
                "doc_id": doc_id,
                "start_line": int(doc.metadata.get("start_line", 0)),
                "end_line": int(doc.metadata.get("end_line", 0)),
                "content": doc.page_content,
            })

    logging.info(f"  [SELECTOR] {len(doc_results)} chunks → {len(candidates)} unique sections (requesting top {k_final})")

    if len(candidates) <= k_final:
        return candidates

    # Stage 2: LLM Re-Ranker
    index_map = {str(i): h for i, h in enumerate(candidates[:50])}
    candidates_text = ""
    for i, h in enumerate(candidates[:50]):
        candidates_text += f"{i}. [{h['breadcrumb']}] (node: {h['node_id']})\n"

    prompt = build_selector_prompt(criteria, doc_type, candidates_text, k_final)

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
                    logging.warning(f"Rate limit hit during section selection. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise e

        response_text = response.text.strip()

        # Log the server's finish reason for telemetry
        try:
            finish_reason = response.candidates[0].finish_reason
            logging.info(f"  [RE-RANKER] Finish Reason: {finish_reason}")
        except Exception:
            pass

        logging.info(f"  [RE-RANKER] Raw response: {response_text[:200]}")

        ranked_ids = []
        try:
            ranking_data = json.loads(response_text)
            ranked_ids = [str(idx) for idx in ranking_data.get("ranked_indices", [])]
        except (json.JSONDecodeError, TypeError):
            # Powerful regex fallback: find any freestanding integer strings in the text
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

        # If re-ranker returned zero valid indices, fall back to raw similarity
        if not final_sections:
            logging.warning(f"  [RE-RANKER] No valid indices parsed. Falling back to top-{k_final} by similarity.")
            return candidates[:k_final]

        return final_sections

    except Exception as e:
        logging.warning(f"LLM Re-ranker failed ({e}). Falling back to top-{k_final}.")

    # Fallback: top-k by raw similarity
    return candidates[:k_final]


@lru_cache(maxsize=256)
def resolve_md_path_for_doc_id(doc_id, data_dir=None):
    """Resolve a generated doc_id to its Markdown path, caching file hashes."""
    if data_dir is None:
        data_dir = str(DOCUMENTS_DIR)

    try:
        filenames = os.listdir(data_dir)
    except OSError as exc:
        logging.warning("Unable to list document directory %s: %s", data_dir, exc)
        return None

    for filename in filenames:
        if not filename.endswith(".md"):
            continue
        md_path = os.path.join(data_dir, filename)
        try:
            with open(md_path, "rb") as fh:
                content_hash = hashlib.sha256(fh.read()).hexdigest()[:12]
        except OSError as exc:
            logging.warning("Unable to hash markdown file %s: %s", md_path, exc)
            continue
        file_doc_id = f"{Path(md_path).stem}_{content_hash}"
        if file_doc_id == doc_id:
            return md_path
    return None


def load_full_section_text(doc_id, start_line, end_line, data_dir=None, md_path=None):
    """Load the full section text from the .md file using line ranges."""
    if data_dir is None:
        data_dir = str(DOCUMENTS_DIR)
    if md_path is None:
        md_path = resolve_md_path_for_doc_id(doc_id, data_dir)
    if md_path is None:
        return None

    try:
        with open(md_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError as exc:
        logging.warning("Unable to read markdown file %s: %s", md_path, exc)
        return None

    try:
        start = int(start_line)
        end = int(end_line)
    except (TypeError, ValueError):
        logging.warning("Invalid section line range for %s: %r-%r", doc_id, start_line, end_line)
        return None
    safe_start = max(0, min(len(lines), start))
    safe_end = max(safe_start, min(len(lines), end))
    return "".join(lines[safe_start:safe_end]).strip()
