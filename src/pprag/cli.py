from __future__ import annotations

import argparse
import importlib
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]


class MissingExtraError(RuntimeError):
    """Raised when a modality command is used without its optional extra."""


def require_extra(extra: str, import_name: str) -> None:
    try:
        importlib.import_module(import_name)
    except ImportError as exc:
        raise MissingExtraError(
            f"The {extra} runner requires optional dependencies.\n\n"
            f"Install them with:\n\n"
            f"  pip install \"pprag[{extra}]\"\n\n"
            f"Or install everything with:\n\n"
            f"  pip install \"pprag[full]\""
        ) from exc


def _run_module(project_dir: str, module: str, args: Sequence[str]) -> int:
    sys.path.insert(0, str(ROOT / project_dir))
    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    try:
        sys.argv = [module, *args]
        # Preserve each migrated implementation's existing relative data/config paths.
        import os
        os.chdir(ROOT / project_dir)
        runpy.run_module(module, run_name="__main__")
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
        try:
            sys.path.remove(str(ROOT / project_dir))
        except ValueError:
            pass
    return 0


def _run_streamlit(project_dir: str, extra: str, args: Sequence[str]) -> int:
    require_extra(extra, "streamlit")
    return subprocess.call([sys.executable, "-m", "streamlit", "run", "app.py", *args], cwd=ROOT / project_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pprag",
        description="Proxy-Pointer RAG: text, multimodal, and document-comparison workflows.",
    )
    subparsers = parser.add_subparsers(dest="modality", metavar="MODALITY")

    text = subparsers.add_parser("text", help="Text-only structural RAG")
    text_sub = text.add_subparsers(dest="command", metavar="COMMAND")
    text_index = text_sub.add_parser("index", help="Build the text-only FAISS index")
    text_index.add_argument("args", nargs=argparse.REMAINDER)
    text_ask = text_sub.add_parser("ask", help="Start the interactive text-only RAG bot")
    text_ask.add_argument("args", nargs=argparse.REMAINDER)
    text_extract = text_sub.add_parser("extract", help="Extract PDFs to Markdown with LlamaParse")
    text_extract.add_argument("args", nargs=argparse.REMAINDER)
    text_benchmark = text_sub.add_parser("benchmark", help="Run the text-only benchmark")
    text_benchmark.add_argument("args", nargs=argparse.REMAINDER)

    multimodal = subparsers.add_parser("multimodal", help="Multimodal RAG with visual citations")
    mm_sub = multimodal.add_subparsers(dest="command", metavar="COMMAND")
    mm_index = mm_sub.add_parser("index", help="Build the multimodal FAISS index")
    mm_index.add_argument("args", nargs=argparse.REMAINDER)
    mm_extract = mm_sub.add_parser("extract", help="Extract PDFs with Adobe PDF Services")
    mm_extract.add_argument("args", nargs=argparse.REMAINDER)
    mm_ui = mm_sub.add_parser("ui", help="Start the multimodal Streamlit UI")
    mm_ui.add_argument("args", nargs=argparse.REMAINDER)
    mm_benchmark = mm_sub.add_parser("benchmark", help="Run the multimodal test suite")
    mm_benchmark.add_argument("args", nargs=argparse.REMAINDER)

    compare = subparsers.add_parser("compare", help="Cross-document comparison")
    compare_sub = compare.add_subparsers(dest="command", metavar="COMMAND")
    compare_ui = compare_sub.add_parser("ui", help="Start the DocComparator Streamlit UI")
    compare_ui.add_argument("args", nargs=argparse.REMAINDER)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv in (["--help"], ["-h"]):
        parser.print_help()
        return 0
    args = parser.parse_args(argv)

    if args.modality is None:
        parser.print_help()
        return 0

    try:
        if args.modality == "text":
            if args.command is None:
                parser.parse_args(["text", "--help"])
                return 0
            require_extra("text", "google.generativeai")
            if args.command == "index":
                return _run_module("Text-Only", "src.indexing.build_pp_index", args.args)
            if args.command == "ask":
                return _run_module("Text-Only", "src.agent.pp_rag_bot", args.args)
            if args.command == "extract":
                return _run_module("Text-Only", "src.extraction.extract_pdf_to_md", args.args)
            if args.command == "benchmark":
                return _run_module("Text-Only", "src.agent.benchmark", args.args)

        if args.modality == "multimodal":
            if args.command is None:
                parser.parse_args(["multimodal", "--help"])
                return 0
            require_extra("multimodal", "google.generativeai")
            if args.command == "index":
                return _run_module("MultiModal", "src.indexing.build_md_index", args.args)
            if args.command == "extract":
                return _run_module("MultiModal", "src.extraction.extract_pdf", args.args)
            if args.command == "ui":
                return _run_streamlit("MultiModal", "multimodal", args.args)
            if args.command == "benchmark":
                return _run_module("MultiModal", "run_test_suite", args.args)

        if args.modality == "compare":
            if args.command is None:
                parser.parse_args(["compare", "--help"])
                return 0
            if args.command == "ui":
                return _run_streamlit("DocComparator", "compare", args.args)

    except MissingExtraError as exc:
        parser.exit(2, f"{exc}\n")

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
