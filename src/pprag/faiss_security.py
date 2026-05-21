"""Safety guard for LangChain FAISS index loading.

LangChain stores FAISS metadata in a pickle-backed docstore. Loading an index
therefore executes pickle deserialization and must only happen for indexes the
user explicitly trusts.
"""
from __future__ import annotations

import os

_TRUTHY = {"1", "true", "yes", "on"}
_DEFAULT_ENV_VARS = ("PPRAG_TRUST_LOCAL_FAISS", "PPRAG_TRUST_FAISS_INDEX")


def is_faiss_deserialization_trusted(*env_vars: str) -> bool:
    """Return True only when an explicit trust opt-in env var is truthy."""
    for name in (*env_vars, *_DEFAULT_ENV_VARS):
        if os.getenv(name, "").strip().lower() in _TRUTHY:
            return True
    return False


def require_trusted_faiss_deserialization(index_path: str, *env_vars: str) -> None:
    """Raise unless FAISS pickle deserialization has been explicitly trusted."""
    if is_faiss_deserialization_trusted(*env_vars):
        return
    accepted = ", ".join((*env_vars, *_DEFAULT_ENV_VARS))
    raise RuntimeError(
        "Refusing to load FAISS index metadata without explicit trust. "
        f"Verify the index at {index_path!r} was generated locally, then set one "
        f"of these environment variables to 1: {accepted}."
    )
