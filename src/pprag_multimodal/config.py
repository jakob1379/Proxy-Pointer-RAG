import logging
import os
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

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
BASE_DIR = str(_default_project_root("MultiModal"))

load_dotenv(Path(BASE_DIR) / ".env")
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    logging.warning("GOOGLE_API_KEY is not set; Gemini API calls will fail until it is configured.")

# ── Paths (Overrideable via ENV) ──────────────────────────────────────────
# We use the 'PP_' prefix to align with the Text-Only version
PDF_DIR     = os.getenv("PP_PDF_DIR", os.path.join(BASE_DIR, "data", "pdf"))
DATASET_DIR = os.getenv("PP_DATA_DIR", os.path.join(BASE_DIR, "data", "extracted_papers"))
TREES_DIR   = os.getenv("PP_TREES_DIR", os.path.join(BASE_DIR, "data", "trees"))
INDEX_DIR   = os.getenv("PP_INDEX_DIR", os.path.join(BASE_DIR, "data", "index"))
RESULTS_DIR = os.getenv("PP_RESULTS_DIR", os.path.join(BASE_DIR, "results"))

# ── Model Config ────────────────────────────────────────────────────────
EMBEDDING_MODEL    = "models/gemini-embedding-001"
EMBEDDING_DIMS     = 1536
NOISE_FILTER_MODEL = "gemini-3.1-flash-lite-preview"
SYNTH_MODEL        = "gemini-3.1-flash-lite-preview"
VISION_FILTER      = False # Set to True for high-fidelity image verification (adds ~30s latency)
EMBEDDING_BATCH_SIZE = int(os.getenv("PP_EMBEDDING_BATCH_SIZE", "20"))
EMBEDDING_BATCH_DELAY = float(os.getenv("PP_EMBEDDING_BATCH_DELAY", "5"))
MIN_SECTION_LENGTH = int(os.getenv("PP_MIN_SECTION_LENGTH", "100"))
