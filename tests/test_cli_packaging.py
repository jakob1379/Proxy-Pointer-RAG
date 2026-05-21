import builtins
import contextlib
import importlib
import io
import pathlib
import sys
import tomllib
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


@contextlib.contextmanager
def fake_google_genai():
    google = types.ModuleType("google")
    genai = types.ModuleType("google.generativeai")
    genai.configure = lambda **kwargs: None
    google.generativeai = genai

    old_google = sys.modules.get("google")
    old_genai = sys.modules.get("google.generativeai")
    sys.modules["google"] = google
    sys.modules["google.generativeai"] = genai
    try:
        yield
    finally:
        if old_google is None:
            sys.modules.pop("google", None)
        else:
            sys.modules["google"] = old_google
        if old_genai is None:
            sys.modules.pop("google.generativeai", None)
        else:
            sys.modules["google.generativeai"] = old_genai


def reload_module(name):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def test_package_exposes_pprag_script_and_modality_extras():
    project = load_pyproject()["project"]

    assert project["name"] == "pprag"
    assert project["scripts"] == {"pprag": "pprag.cli:main"}

    extras = project["optional-dependencies"]
    assert {"text", "multimodal", "compare", "full"}.issubset(extras)
    assert "google-generativeai" in extras["text"]
    assert "pillow" in extras["multimodal"]
    assert "pdfservices-sdk" in extras["multimodal"]
    assert "streamlit" in extras["compare"]

    full = set(extras["full"])
    for extra_name in ("text", "multimodal", "compare"):
        assert set(extras[extra_name]).issubset(full)

    modules = load_pyproject()["tool"]["uv"]["build-backend"]["module-name"]
    assert {"pprag", "pprag_text_only", "pprag_multimodal", "pprag_doc_comparator"}.issubset(modules)


def test_cli_dispatch_uses_importable_modality_packages():
    cli_source = (SRC / "pprag" / "cli.py").read_text()

    assert "parents[2]" not in cli_source
    assert '"Text-Only"' not in cli_source
    assert '"MultiModal"' not in cli_source
    assert '"DocComparator"' not in cli_source
    assert "pprag_text_only" in cli_source
    assert "pprag_multimodal" in cli_source
    assert "pprag_doc_comparator" in cli_source


def test_packaged_configs_default_to_working_tree_not_site_packages(monkeypatch, tmp_path):
    monkeypatch.chdir(ROOT)
    for key in (
        "PP_PROJECT_ROOT",
        "PPRAG_PROJECT_ROOT",
        "DC_PROJECT_ROOT",
        "PP_PDF_DIR",
        "PP_DATA_DIR",
        "PP_TREES_DIR",
        "PP_INDEX_DIR",
        "PP_RESULTS_DIR",
        "DC_UPLOADS_DIR",
        "DC_DOCUMENTS_DIR",
        "DC_TREES_DIR",
        "DC_INDEX_DIR",
    ):
        monkeypatch.delenv(key, raising=False)

    with fake_google_genai():
        text_config = reload_module("pprag_text_only.config")
        multimodal_config = reload_module("pprag_multimodal.config")
        compare_config = reload_module("pprag_doc_comparator.config")

    assert text_config.PROJECT_ROOT == ROOT / "Text-Only"
    assert text_config.DATA_DIR == ROOT / "Text-Only" / "data" / "documents"
    assert pathlib.Path(multimodal_config.BASE_DIR) == ROOT / "MultiModal"
    assert pathlib.Path(multimodal_config.PDF_DIR) == ROOT / "MultiModal" / "data" / "pdf"
    assert pathlib.Path(multimodal_config.DATASET_DIR) == ROOT / "MultiModal" / "data" / "extracted_papers"
    assert compare_config.PROJECT_ROOT == ROOT / "DocComparator"
    assert compare_config.DOCUMENTS_DIR == ROOT / "DocComparator" / "data" / "documents"

    explicit_root = tmp_path / "runtime"
    explicit_root.mkdir()
    monkeypatch.setenv("PPRAG_PROJECT_ROOT", str(explicit_root))
    with fake_google_genai():
        text_config = reload_module("pprag_text_only.config")
        multimodal_config = reload_module("pprag_multimodal.config")

    assert text_config.PROJECT_ROOT == explicit_root
    assert pathlib.Path(multimodal_config.BASE_DIR) == explicit_root


def test_multimodal_extract_uses_configured_paths():
    extract_source = (SRC / "pprag_multimodal" / "extraction" / "extract_pdf.py").read_text()

    assert "Path(__file__).parent.parent.parent" not in extract_source
    assert "from pprag_multimodal.config import DATASET_DIR, PDF_DIR" in extract_source
    assert "pdf_dir = Path(PDF_DIR)" in extract_source
    assert "output_dir = Path(DATASET_DIR)" in extract_source


def test_minimal_cli_help_does_not_import_optional_dependencies():
    blocked = {
        "google",
        "langchain_community",
        "langchain_core",
        "langchain_text_splitters",
        "faiss",
        "streamlit",
        "PIL",
        "adobe",
        "llama_cloud",
        "pandas",
    }
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".", 1)[0] in blocked:
            raise AssertionError(f"unexpected optional import during CLI help: {name}")
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import
    try:
        sys.modules.pop("pprag.cli", None)
        sys.modules.pop("pprag", None)
        import pprag.cli

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            rc = pprag.cli.main(["--help"])
    finally:
        builtins.__import__ = real_import

    assert rc == 0
    help_text = output.getvalue()
    assert "text" in help_text
    assert "multimodal" in help_text
    assert "compare" in help_text


def test_streamlit_modalities_expose_serve_alias():
    import pprag.cli

    multimodal_help = io.StringIO()
    with contextlib.redirect_stdout(multimodal_help):
        rc = pprag.cli.main(["multimodal", "--help"])
    assert rc == 0
    assert "serve" in multimodal_help.getvalue()

    compare_help = io.StringIO()
    with contextlib.redirect_stdout(compare_help):
        rc = pprag.cli.main(["compare", "--help"])
    assert rc == 0
    assert "serve" in compare_help.getvalue()


def test_serve_alias_starts_streamlit_app(monkeypatch):
    import pprag.cli

    calls = []

    def fake_run_streamlit(extra, package, args):
        calls.append((extra, package, list(args)))
        return 0

    monkeypatch.setattr(pprag.cli, "_run_streamlit", fake_run_streamlit)

    assert pprag.cli.main(["multimodal", "serve", "--server.port", "8502"]) == 0
    assert pprag.cli.main(["compare", "serve", "--server.port", "8503"]) == 0
    assert calls == [
        ("multimodal", "pprag_multimodal", ["--server.port", "8502"]),
        ("compare", "pprag_doc_comparator", ["--server.port", "8503"]),
    ]


def test_text_commands_dispatch_to_text_package(monkeypatch):
    import pprag.cli

    calls = []

    def fake_run_module(extra, package, module, args):
        calls.append((extra, package, module, list(args)))
        return 0

    monkeypatch.setattr(pprag.cli, "_run_module", fake_run_module)

    assert pprag.cli.main(["text", "index", "--fresh"]) == 0
    assert pprag.cli.main(["text", "ask"]) == 0

    assert calls == [
        ("text", "pprag_text_only", "indexing.build_pp_index", ["--fresh"]),
        ("text", "pprag_text_only", "agent.pp_rag_bot", []),
    ]


def test_missing_extra_message_names_install_target():
    from pprag.cli import MissingExtraError, require_extra

    try:
        require_extra("multimodal", "definitely_missing_dependency_for_pprag")
    except MissingExtraError as exc:
        message = str(exc)
    else:
        raise AssertionError("require_extra should fail for an unavailable dependency")

    assert 'pip install "pprag[multimodal]"' in message
    assert 'pip install "pprag[full]"' in message
