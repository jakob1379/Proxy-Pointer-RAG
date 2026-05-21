"""
DocComparator: Centralized Configuration

All paths are relative to the project root. Override via .env or environment variables.
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

import google.generativeai as genai

def _default_project_root(subproject: str) -> Path:
    explicit_root = os.getenv("DC_PROJECT_ROOT") or os.getenv("PPRAG_PROJECT_ROOT")
    if explicit_root:
        return Path(explicit_root).expanduser()

    cwd = Path.cwd()
    checkout_subproject = cwd / subproject
    if checkout_subproject.is_dir():
        return checkout_subproject
    return cwd


# ── Project Root ────────────────────────────────────────────────────────
PROJECT_ROOT = _default_project_root("DocComparator")

# ── Environment ─────────────────────────────────────────────────────────
load_dotenv(PROJECT_ROOT / ".env")
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key:
    genai.configure(api_key=google_api_key)
else:
    logging.warning("GOOGLE_API_KEY is not set; Gemini API calls will fail until it is configured.")

# ── Paths ───────────────────────────────────────────────────────────────
UPLOADS_DIR   = Path(os.getenv("DC_UPLOADS_DIR",   PROJECT_ROOT / "data" / "uploads"))
DOCUMENTS_DIR = Path(os.getenv("DC_DOCUMENTS_DIR", PROJECT_ROOT / "data" / "documents"))
TREES_DIR     = Path(os.getenv("DC_TREES_DIR",     PROJECT_ROOT / "data" / "trees"))
INDEX_DIR     = Path(os.getenv("DC_INDEX_DIR",      PROJECT_ROOT / "data" / "index"))

# ── LlamaParse ──────────────────────────────────────────────────────────
LLAMA_PARSE_TIER = os.getenv("LLAMA_PARSE_TIER", "cost_effective")

# ── Models ──────────────────────────────────────────────────────────────
EMBEDDING_MODEL    = "models/gemini-embedding-001"
EMBEDDING_DIMS     = 1536
LLM_MODEL          = os.getenv("DC_LLM_MODEL", "gemini-3.1-flash-lite-preview")

# ── Chunking ────────────────────────────────────────────────────────────
CHUNK_SIZE    = 2000
CHUNK_OVERLAP = 200

# ── Comparison Limits ──────────────────────────────────────────────────
MAX_DOC1_SECTIONS = 10
MAX_DOC2_MATCHES  = 3
