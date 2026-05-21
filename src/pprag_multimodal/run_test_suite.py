import os
import sys
import json
import time
from pathlib import Path

from pprag_multimodal.agent.mm_rag_bot import MultimodalProxyPointerRAG
from pprag_multimodal.config import INDEX_DIR, TREES_DIR, DATASET_DIR, RESULTS_DIR

# Ensure results dir exists
results_path = Path(RESULTS_DIR)
results_path.mkdir(parents=True, exist_ok=True)

def run_suite():
    # 1. Load Queries from results dir
    query_file = results_path / "test_queries.json"
    if not query_file.exists():
        print(f"Error: {query_file} not found.")
        return

    with open(query_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        queries = data.get("test_queries", [])
    if not isinstance(queries, list):
        print(f"Error: 'test_queries' must be a list, got {type(queries).__name__}")
        return

    results_log = []
    log_path = results_path / "test_log.json"

    def save_results(completed):
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({"run_summary": {"total": len(queries), "completed": completed}, "results": results_log}, f, indent=2)

    # 2. Init Bot
    print(f"Initializing Multimodal RAG Bot (Index: {INDEX_DIR})...")
    try:
        bot = MultimodalProxyPointerRAG(INDEX_DIR, TREES_DIR, DATASET_DIR)
    except Exception as exc:
        print(f"Error initializing Multimodal RAG Bot: {exc}")
        sys.exit(1)

    print(f"\nStarting Test Suite ({len(queries)} queries)...\n")

    for i, q in enumerate(queries):
        qid = q.get("id")
        text_query = q.get("query")
        category = q.get("category")

        if not isinstance(text_query, str) or not text_query.strip():
            results_log.append({
                "id": qid,
                "category": category,
                "query": text_query,
                "response": "",
                "sources": [],
                "images_found": [],
                "time_seconds": 0.0,
                "error": "Invalid query: expected non-empty string",
            })
            save_results(i + 1)
            print(f"[{i+1}/{len(queries)}] -> ERROR: invalid query payload")
            continue

        print(f"[{i+1}/{len(queries)}] Running Query: {text_query[:60]}...")

        start_time = time.perf_counter()
        try:
            response = bot.chat(text_query)
            elapsed = time.perf_counter() - start_time

            # Format results for logging
            if not isinstance(response, dict):
                response = {}
            images = response.get("images", [])
            if not isinstance(images, list):
                images = []
            result_entry = {
                "id": qid,
                "category": category,
                "query": text_query,
                "response": response.get("text", ""),
                "sources": response.get("paths", []),
                "images_found": [
                    {
                        "label": img.get("label", ""),
                        "path": img.get("full_path", ""),
                        "exists": img.get("exists", False)
                    } for img in images if isinstance(img, dict)
                ],
                "time_seconds": round(elapsed, 2)
            }
            results_log.append(result_entry)

            # Save incrementally after each query
            save_results(i + 1)

            print(f"  -> SUCCESS (Took {elapsed:.1f}s, found {len(images)} images)")

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            print(f"  -> ERROR: {e}")
            results_log.append({
                "id": qid,
                "category": category,
                "query": text_query,
                "response": "",
                "sources": [],
                "images_found": [],
                "time_seconds": round(elapsed, 2),
                "error": str(e),
            })
            save_results(i + 1)

    print(f"\nTest Suite Complete! Report saved to: {log_path}")

if __name__ == "__main__":
    run_suite()
