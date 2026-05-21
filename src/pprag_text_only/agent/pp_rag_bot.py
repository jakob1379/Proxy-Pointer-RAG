"""
Proxy-Pointer: Structural RAG Bot

Interactive RAG bot that:
  1. Vector search (k=200) for broad recall
  2. Deduplicates by (doc_id, node_id) — unique per section per document
  3. LLM re-ranker selects top 5 by hierarchical path relevance
  4. Loads full document sections from source .md files
  5. LLM synthesizer generates grounded answers

Usage:
    python -m pprag_text_only.agent.pp_rag_bot
"""
import os
import re
import json
import logging
from collections.abc import Sequence as SequenceABC

from pprag_text_only.config import DATA_DIR, INDEX_DIR, EMBEDDING_MODEL, EMBEDDING_DIMS, SYNTH_MODEL
from pprag.faiss_security import require_trusted_faiss_deserialization

logger = logging.getLogger(__name__)

import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


def _normalize_embeddings(result, expected_count):
    """Return a list of embedding vectors from Gemini's single or batch shapes."""
    if not isinstance(result, dict):
        raise ValueError(f"Unexpected embedding response type: {type(result).__name__}")

    if "embeddings" in result:
        embeddings = result["embeddings"]
    elif "embedding" in result:
        embeddings = result["embedding"]
    else:
        raise ValueError(f"Embedding response missing embedding data: {result!r}")

    if expected_count == 1 and embeddings and isinstance(embeddings[0], (int, float)):
        return [embeddings]
    if not isinstance(embeddings, SequenceABC):
        raise ValueError(f"Embedding response is not a sequence: {result!r}")
    vectors = list(embeddings)
    if len(vectors) != expected_count:
        raise ValueError(f"Expected {expected_count} embedding(s), received {len(vectors)}")
    return vectors


def _parse_non_negative_int(value, default=0):
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _load_trusted_faiss_index(index_path, embeddings):
    """Load a locally generated FAISS index with explicit trust documentation.

    LangChain's FAISS.load_local requires pickle deserialization for metadata.
    Only pass paths that this application created locally or that the user has
    explicitly chosen to trust; never load indexes from untrusted downloads.
    """
    require_trusted_faiss_deserialization(index_path, "PP_TRUST_FAISS_INDEX")
    return FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )


# ── Custom Embedding Wrapper ────────────────────────────────────────────
class GeminiEmbeddings(Embeddings):
    """LangChain-compatible wrapper for Gemini embeddings at configurable dims."""

    def __init__(self, model=EMBEDDING_MODEL, dimensionality=EMBEDDING_DIMS):
        self.model = model
        self.dimensionality = dimensionality

    def embed_documents(self, texts):
        result = genai.embed_content(
            model=self.model,
            content=texts,
            output_dimensionality=self.dimensionality
        )
        return _normalize_embeddings(result, len(texts))

    def embed_query(self, text):
        result = genai.embed_content(
            model=self.model,
            content=text,
            output_dimensionality=self.dimensionality
        )
        return _normalize_embeddings(result, 1)[0]


class ProxyPointerRAG:
    def __init__(self, index_path=None, data_dir=None):
        self.data_dir = str(data_dir or DATA_DIR)
        index_path = str(index_path or INDEX_DIR)

        # 1. Load Gemini Embeddings
        logger.info("Loading %s @ %s dims...", EMBEDDING_MODEL, EMBEDDING_DIMS)
        self.embeddings = GeminiEmbeddings()

        # 2. Load FAISS Index
        logger.info("Loading index from %s...", index_path)
        self.vector_db = _load_trusted_faiss_index(index_path, self.embeddings)

        # 3. Initialize synthesis model
        self.model = genai.GenerativeModel(SYNTH_MODEL)
        self.llm_timeout = float(os.getenv("PPRAG_LLM_TIMEOUT", "120"))

    def _generate_content(self, *args, **kwargs):
        try:
            return self.model.generate_content(
                *args, request_options={"timeout": self.llm_timeout}, **kwargs
            )
        except TypeError:
            # Test doubles and older SDKs may not accept request_options.
            return self.model.generate_content(*args, **kwargs)

    def retrieve_unique_nodes(self, query, k_search=200, k_final=5):
        """Stage 1: Broad vector recall → Stage 2: LLM re-ranking."""

        # Stage 1: Broad Recall
        docs = self.vector_db.similarity_search(query, k=k_search)

        candidates = []
        seen_nodes = set()   # (doc_id, node_id) — dedup within AND across docs
        for doc in docs:
            node_id = doc.metadata.get("node_id")
            doc_id = doc.metadata.get("doc_id", "UNK")
            dedup_key = (doc_id, node_id)
            if dedup_key not in seen_nodes:
                seen_nodes.add(dedup_key)
                internal_crumb = doc.metadata.get("breadcrumb", "Unknown Path")
                global_crumb = f"{doc_id} > {internal_crumb}"

                start_line = _parse_non_negative_int(doc.metadata.get("start_line", 0))
                end_line = _parse_non_negative_int(doc.metadata.get("end_line", start_line))
                if end_line < start_line:
                    logging.warning("Invalid line range for %s/%s; using chunk content", doc_id, node_id)
                    start_line = end_line = 0

                info = {
                    "node_id": node_id,
                    "global_breadcrumb": global_crumb,
                    "doc_id": doc_id,
                    "start_line": start_line,
                    "end_line": end_line,
                    "content": doc.page_content,
                }
                candidates.append(info)

        # Stage 2: LLM Re-Ranker
        # Build an index-keyed map so re-ranker IDs are always unique,
        # regardless of node_id collisions across documents.
        rerank_limit = max(k_final, int(os.getenv("PPRAG_RERANK_LIMIT", "50")))
        candidate_subset = candidates[:min(len(candidates), rerank_limit)]
        index_map = {str(i): h for i, h in enumerate(candidate_subset)}
        candidates_text = ""
        for i, h in enumerate(candidate_subset):
            candidates_text += (
                f"{i}. [{h['global_breadcrumb']}] (node: {h['node_id']})\n"
            )

        prompt = f"""You are a structural re-ranker.
Your goal is to find the Top {k_final} most relevant candidates based on their HIERARCHICAL PATH relative to the user's query.

User Query: "{query}"

CANDIDATE HIERARCHIES (INDEX | Full Path):
{candidates_text}

RANKING RULES:
1. Highly Specific Matches (e.g. if query is about 'Chapter 2 questions', a path like 'Chapter 2 > Intro > Questions' is Rank 1).
2. If specific matches are not found, include similar, partial matches.
3. If the query is not pointing to any specific chapter or section, look for the most relevant Contextual Matches (e.g. if query is about 'India growth', a path like 'Chapter 1 > Outlook > Country outlooks' is very strong).
4. Structural Priority: Prioritize exact structural anchors (Box 1.1, Figure B1.1) if the query mentions them.
5. Each INDEX must appear ONLY ONCE. Do not repeat any index.
6. Output ONLY a comma-separated list of the Top {k_final} unique numeric indices. No text, no explanation.

Output Example: 3, 7, 12, 0, 25
"""
        try:
            response = self._generate_content(prompt).text.strip()
            clean_text = re.sub(r"[^0-9, ]", "", response)
            ranked_ids = [
                rid.strip() for rid in clean_text.split(",") if rid.strip()
            ]

            final_pointers = []
            seen = set()
            for rid in ranked_ids:
                if rid in index_map and rid not in seen:
                    final_pointers.append(index_map[rid])
                    seen.add(rid)
                if len(final_pointers) >= k_final:
                    break

            if final_pointers:
                return final_pointers
        except Exception as e:
            logger.warning("LLM ranker failed; falling back to top %s: %s", k_final, e)

        # Fallback to top-k unique by similarity
        return candidate_subset[:k_final]

    def chat(self, query):
        """Orchestrate Retrieval and Synthesis."""
        pointers = self.retrieve_unique_nodes(query)

        print("\n" + "=" * 100)
        print(f"Final Context Selection (Top {len(pointers)} Unique Nodes):")
        for p in pointers:
            print(f"  -> Node {p['node_id']:<6} | {p['global_breadcrumb']}")
        print("=" * 100 + "\n")

        context = []
        for p in pointers:
            md_path = os.path.join(self.data_dir, f"{p['doc_id']}.md")
            if os.path.exists(md_path):
                with open(md_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    safe_start = max(0, min(len(lines), p["start_line"]))
                    safe_end = max(safe_start, min(len(lines), p["end_line"]))
                    text = "".join(lines[safe_start:safe_end]).strip() or p["content"]
                    context.append(
                        f"### REFERENCE: {p['global_breadcrumb']}\n{text}"
                    )
            else:
                # Fallback to vector DB chunk content if .md file is missing
                context.append(
                    f"### REFERENCE: {p['global_breadcrumb']}\n{p['content']}"
                )

        synth_prompt = (
            f"Query: {query}\n\nContext:\n"
            + "\n\n".join(context)
            + "\n\n"
            "INSTRUCTIONS:\n"
            "1. Answer the query concisely using ONLY the context above.\n"
            "2. Do NOT reference any IDs (e.g. 'ID: 0114' or 'node: 0085') anywhere in your answer.\n"
            "3. At the END of your answer, add a 'Sources:' section listing the breadcrumb paths you used, e.g.:\n"
            "   Sources:\n"
            "   - AMD > Results of Operations > Data Center\n"
            "   - AMD > Consolidated Statements of Operations\n"
        )
        try:
            response = self._generate_content(synth_prompt)
            return response.text
        except Exception as exc:
            logger.exception("Synthesis LLM call failed")
            return f"Error generating response: {exc}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    bot = ProxyPointerRAG()

    print("\nProxy-Pointer RAG Bot ready. Type 'exit' to quit.\n")
    try:
        while True:
            user_in = input("User >> ")
            if user_in.lower() in ["exit", "quit"]:
                break
            print(bot.chat(user_in))
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
