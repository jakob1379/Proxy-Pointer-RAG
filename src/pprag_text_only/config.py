"""
Proxy-Pointer: Centralized Configuration

All paths are relative to the project root. Override via .env or environment variables.
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import google.generativeai as genai

def _default_project_root(subproject: str) -> Path:
    explicit_root = os.getenv("PP_PROJECT_ROOT") or os.getenv("PPRAG_PROJECT_ROOT")
    if explicit_root:
        return Path(explicit_root).expanduser()

    cwd = Path.cwd()
    checkout_subproject = cwd / subproject
    if checkout_subproject.is_dir():
        return checkout_subproject
    return cwd


# ── Project Root ────────────────────────────────────────────────────────
PROJECT_ROOT = _default_project_root("Text-Only")

# ── Environment ─────────────────────────────────────────────────────────
load_dotenv(PROJECT_ROOT / ".env")
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key:
    genai.configure(api_key=google_api_key)
else:
    logging.warning("GOOGLE_API_KEY is not set; Gemini API calls will fail until it is configured.")
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ── Paths ───────────────────────────────────────────────────────────────
PDF_DIR       = Path(os.getenv("PP_PDF_DIR",       PROJECT_ROOT / "data" / "pdf"))
DATA_DIR      = Path(os.getenv("PP_DATA_DIR",      PROJECT_ROOT / "data" / "documents"))
TREES_DIR     = Path(os.getenv("PP_TREES_DIR",     PROJECT_ROOT / "data" / "trees"))
INDEX_DIR     = Path(os.getenv("PP_INDEX_DIR",      PROJECT_ROOT / "data" / "index"))
RESULTS_DIR   = Path(os.getenv("PP_RESULTS_DIR",    PROJECT_ROOT / "data" / "results"))

# ── LlamaParse ──────────────────────────────────────────────────────────
# Options: "cost_effective" (v2 default), "agentic", or "agentic_plus" (best for complex docs)
LLAMA_PARSE_TIER = os.getenv("LLAMA_PARSE_TIER", "cost_effective")

# ── Models ──────────────────────────────────────────────────────────────
EMBEDDING_MODEL    = "models/gemini-embedding-001"
EMBEDDING_DIMS     = 1536
NOISE_FILTER_MODEL = "gemini-3.1-flash-lite-preview"
SYNTH_MODEL        = "gemini-3.1-flash-lite-preview"
