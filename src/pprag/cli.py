from __future__ import annotations

import argparse
import importlib
import importlib.resources as resources
import runpy
import subprocess
import sys
from typing import Sequence


class MissingExtraError(RuntimeError):
    """Raised when a modality command is used without its optional extra."""


def require_extra(extra: str, import_name: str):
    try:
        return importlib.import_module(import_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        top_level = import_name.split(".", 1)[0]
        if missing and missing != top_level and not import_name.startswith(f"{missing}."):
            raise
        raise MissingExtraError(
            f"The {extra} runner requires optional dependencies.\n\n"
            f"Install them with:\n\n"
            f"  pip install \"pprag[{extra}]\"\n\n"
            f"Or install everything with:\n\n"
            f"  pip install \"pprag[full]\""
        ) from exc


def _run_module(extra: str, package: str, module: str, args: Sequence[str]) -> int:
    require_extra(extra, package)
    old_argv = sys.argv[:]
    try:
        module_name = f"{package}.{module}"
        sys.argv = [module_name, *args]
        runpy.run_module(module_name, run_name="__main__")
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing.startswith(package):
            raise
        raise MissingExtraError(
            f"The {extra} runner could not import {exc.name or 'a required module'}.\n\n"
            f"Install it with:\n\n"
            f"  pip install \"pprag[{extra}]\"\n\n"
            f"Or install everything with:\n\n"
            f"  pip install \"pprag[full]\""
        ) from exc
    finally:
        sys.argv = old_argv
    return 0


def _run_streamlit(extra: str, package: str, args: Sequence[str]) -> int:
    require_extra(extra, "streamlit")
    require_extra(extra, package)
    app = resources.files(package).joinpath("app.py")
    with resources.as_file(app) as app_path:
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path), *args])


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
    mm_serve = mm_sub.add_parser("serve", help="Start the multimodal Streamlit UI")
    mm_serve.add_argument("args", nargs=argparse.REMAINDER)
    mm_benchmark = mm_sub.add_parser("benchmark", help="Run the multimodal test suite")
    mm_benchmark.add_argument("args", nargs=argparse.REMAINDER)

    compare = subparsers.add_parser("compare", help="Cross-document comparison")
    compare_sub = compare.add_subparsers(dest="command", metavar="COMMAND")
    compare_ui = compare_sub.add_parser("ui", help="Start the DocComparator Streamlit UI")
    compare_ui.add_argument("args", nargs=argparse.REMAINDER)
    compare_serve = compare_sub.add_parser("serve", help="Start the DocComparator Streamlit UI")
    compare_serve.add_argument("args", nargs=argparse.REMAINDER)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv in (["--help"], ["-h"]):
        parser.print_help()
        return 0
    if len(argv) == 2 and argv[1] in ("--help", "-h"):
        help_args = [argv[0], argv[1]]
        try:
            parser.parse_args(help_args)
        except SystemExit as exc:
            return int(exc.code or 0)
    args, unknown_args = parser.parse_known_args(argv)
    if unknown_args:
        if hasattr(args, "args"):
            args.args = [*unknown_args, *args.args]
        else:
            parser.error(f"unrecognized arguments: {' '.join(unknown_args)}")

    if args.modality is None:
        parser.print_help()
        return 0

    try:
        if args.modality == "text":
            if args.command is None:
                parser.parse_args(["text", "--help"])
                return 0
            if args.command == "index":
                return _run_module("text", "pprag_text_only", "indexing.build_pp_index", args.args)
            if args.command == "ask":
                return _run_module("text", "pprag_text_only", "agent.pp_rag_bot", args.args)
            if args.command == "extract":
                return _run_module("text", "pprag_text_only", "extraction.extract_pdf_to_md", args.args)
            if args.command == "benchmark":
                return _run_module("text", "pprag_text_only", "agent.benchmark", args.args)

        if args.modality == "multimodal":
            if args.command is None:
                parser.parse_args(["multimodal", "--help"])
                return 0
            if args.command == "index":
                return _run_module("multimodal", "pprag_multimodal", "indexing.build_md_index", args.args)
            if args.command == "extract":
                return _run_module("multimodal", "pprag_multimodal", "extraction.extract_pdf", args.args)
            if args.command in ("ui", "serve"):
                return _run_streamlit("multimodal", "pprag_multimodal", args.args)
            if args.command == "benchmark":
                return _run_module("multimodal", "pprag_multimodal", "run_test_suite", args.args)

        if args.modality == "compare":
            if args.command is None:
                parser.parse_args(["compare", "--help"])
                return 0
            if args.command in ("ui", "serve"):
                return _run_streamlit("compare", "pprag_doc_comparator", args.args)

    except MissingExtraError as exc:
        parser.exit(2, f"{exc}\n")

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
