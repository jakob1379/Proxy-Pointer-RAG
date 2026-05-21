import contextlib
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@contextlib.contextmanager
def fake_gemini():
    google = types.ModuleType("google")
    genai = types.ModuleType("google.generativeai")

    class GenerationConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Response:
        def __init__(self, text):
            self.text = text

    class GenerativeModel:
        def __init__(self, model_name):
            self.model_name = model_name

        def generate_content(self, prompt, generation_config=None):
            if "noise_nodes" in prompt:
                return Response('{"noise_nodes": []}')
            if "CANDIDATE HIERARCHIES" in prompt:
                return Response("0, 1, 2")
            if "Project Aurora" in prompt and "retrieval engine" in prompt:
                return Response(
                    "Project Aurora uses a structural retrieval engine with deterministic citations.\n\n"
                    "Sources:\n"
                    "- fixture > Project Aurora > Retrieval Architecture"
                )
            return Response("No answer")

    def configure(**kwargs):
        return None

    def embed_content(model, content, output_dimensionality):
        def embed_one(text):
            text = str(text).lower()
            vector = [0.0] * output_dimensionality
            for token, slot in (("aurora", 0), ("retrieval", 1), ("finance", 2), ("citation", 3)):
                vector[slot] = float(text.count(token))
            vector[4] = 1.0
            return vector

        if isinstance(content, list):
            return {"embedding": [embed_one(item) for item in content]}
        return {"embedding": embed_one(content)}

    genai.configure = configure
    genai.embed_content = embed_content
    genai.GenerativeModel = GenerativeModel
    genai.GenerationConfig = GenerationConfig
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


def clear_text_only_modules():
    for name in list(sys.modules):
        if name == "pprag_text_only" or name.startswith("pprag_text_only."):
            sys.modules.pop(name, None)


def test_pprag_text_cli_builds_queryable_rag_index(tmp_path, monkeypatch):
    data_dir = tmp_path / "documents"
    trees_dir = tmp_path / "trees"
    index_dir = tmp_path / "index"
    data_dir.mkdir()

    (data_dir / "fixture.md").write_text(
        "# Project Aurora\n\n"
        "Project Aurora is a test document for the Proxy-Pointer RAG integration suite.\n\n"
        "## Retrieval Architecture\n\n"
        "Project Aurora uses a structural retrieval engine. "
        "The retrieval engine indexes section breadcrumbs and returns deterministic citations. "
        "This section is intentionally long enough to be indexed as a retrievable chunk.\n\n"
        "## Finance Notes\n\n"
        "Finance notes are unrelated to the retrieval architecture question. "
        "They discuss budgets, forecasts, and approvals for the implementation team.\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("PP_DATA_DIR", str(data_dir))
    monkeypatch.setenv("PP_TREES_DIR", str(trees_dir))
    monkeypatch.setenv("PP_INDEX_DIR", str(index_dir))
    monkeypatch.setenv("PP_TRUST_FAISS_INDEX", "1")

    import pprag.cli

    clear_text_only_modules()
    with fake_gemini():
        assert pprag.cli.main(["text", "index", "--fresh"]) == 0

        assert (trees_dir / "fixture_structure.json").exists()
        assert (index_dir / "index.faiss").exists()
        assert (index_dir / "index.pkl").exists()

        try:
            from pprag_text_only.agent.pp_rag_bot import ProxyPointerRAG

            rag = ProxyPointerRAG(index_path=index_dir, data_dir=data_dir)
            pointers = rag.retrieve_unique_nodes("How does Project Aurora use retrieval citations?", k_search=5, k_final=2)
            assert pointers
            assert pointers[0]["doc_id"] == "fixture"
            assert "Retrieval Architecture" in pointers[0]["global_breadcrumb"]

            answer = rag.chat("How does Project Aurora use retrieval citations?")
            assert "structural retrieval engine" in answer
            assert "Sources:" in answer
            assert "fixture > Project Aurora > Retrieval Architecture" in answer
        finally:
            clear_text_only_modules()
