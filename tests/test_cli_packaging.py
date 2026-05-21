import builtins
import contextlib
import io
import pathlib
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_package_exposes_pprag_script_and_modality_extras():
    project = load_pyproject()["project"]

    assert project["name"] == "pprag"
    assert project["scripts"]["pprag"] == "pprag.cli:main"

    extras = project["optional-dependencies"]
    assert {"text", "multimodal", "compare", "full"}.issubset(extras)
    assert "google-generativeai" in extras["text"]
    assert "pillow" in extras["multimodal"]
    assert "pdfservices-sdk" in extras["multimodal"]
    assert "streamlit" in extras["compare"]

    full = set(extras["full"])
    for extra_name in ("text", "multimodal", "compare"):
        assert set(extras[extra_name]).issubset(full)


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
