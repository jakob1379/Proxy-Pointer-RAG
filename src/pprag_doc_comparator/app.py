"""
DocComparator: Streamlit UI

Interactive document comparison tool powered by Proxy-Pointer retrieval.
Upload two documents, specify comparison criteria, and get a detailed
traffic-light report comparing them section by section.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import time
import hashlib
import logging
from pathlib import Path

import streamlit as st  # type: ignore


from pprag_doc_comparator.config import (
    UPLOADS_DIR, DOCUMENTS_DIR, TREES_DIR, INDEX_DIR,
    MAX_DOC1_SECTIONS, MAX_DOC2_MATCHES,
    EMBEDDING_MODEL, EMBEDDING_DIMS, LLM_MODEL
)
from pprag_doc_comparator.extraction.extract_pdf_to_md import extract_to_md
from pprag_doc_comparator.indexing.build_doc_index import (
    build_comparator_index, get_doc_id, GeminiEmbeddings,
    build_skeleton_trees
)
from pprag_doc_comparator.validation.criteria_validator import (
    validate_criteria,
    build_section_selection_query,
    build_cross_retrieval_query,
    build_comparison_prompt,
)
from pprag_doc_comparator.comparison.section_selector import (
    select_relevant_sections, load_full_section_text, resolve_md_path_for_doc_id
)
from pprag_doc_comparator.comparison.cross_retriever import retrieve_matching_sections
from pprag_doc_comparator.comparison.section_comparator import (
    compare_sections, extract_rating
)
from pprag_doc_comparator.report.report_builder import (
    build_executive_summary, build_section_block,
    assemble_report
)
from pprag.faiss_security import require_trusted_faiss_deserialization

from langchain_community.vectorstores import FAISS


# ── Page Config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocComparator — Proxy-Pointer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* GLOBAL */
    .main { background-color: #fcfcfc; }
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 0rem !important;
    }

    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    /* Light text for labels and captions only */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
        color: #e2e8f0 !important;
    }
    /* Dark text inside input fields */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] select,
    [data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #1e293b !important;
    }
    /* Compact file uploaders */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        margin-bottom: -10px;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] label {
        font-size: 0.8rem !important;
        margin-bottom: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        padding: 4px 8px !important;
        border: 1px dashed #475569 !important;
        border-radius: 6px !important;
        background: #1e293b !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section > div {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    /* File info text (filename, size) in uploader */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] span {
        color: #cbd5e1 !important;
    }
    /* Visible upload button */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        background: #3b82f6 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 2px 12px !important;
        font-size: 0.75rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] small,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] > div:last-child {
        display: none !important;
    }
    /* Reduce sidebar spacing */
    [data-testid="stSidebar"] .stMarkdown {
        margin-bottom: -8px;
    }
    [data-testid="stSidebar"] hr {
        margin: 6px 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 1rem !important;
    }

    /* TRAFFIC LIGHT BADGES */
    .traffic-red {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 10px 15px;
        border-radius: 4px;
        margin: 8px 0;
    }
    .traffic-yellow {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 10px 15px;
        border-radius: 4px;
        margin: 8px 0;
    }
    .traffic-green {
        background-color: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 10px 15px;
        border-radius: 4px;
        margin: 8px 0;
    }

    /* PROGRESS */
    .stProgress > div > div > div {
        background-color: #3b82f6;
    }

    /* REPORT HEADERS (Sober Blue for Doc 1, Deep Maroon for Doc 2) */
    [data-testid="stMarkdownContainer"] h3 {
        color: #1e293b !important;
        font-weight: bold !important;
        font-size: 1.3rem !important;
        margin-top: 1.5rem !important;
    }
    [data-testid="stMarkdownContainer"] h4 {
        color: #800000 !important;
        font-weight: bold !important;
        font-size: 1.05rem !important;
        margin-top: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ────────────────────────────────────────────────────

def save_uploaded_file(uploaded_file, dest_dir):
    """Save a Streamlit uploaded file to disk. Returns the path."""
    os.makedirs(dest_dir, exist_ok=True)
    safe_name = Path(uploaded_file.name).name
    dest_root = Path(dest_dir).resolve()
    dest_path = (dest_root / safe_name).resolve()
    if dest_path.parent != dest_root:
        raise ValueError("Unsafe uploaded filename")
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(dest_path)


def uploaded_file_signature(uploaded_file):
    """Stable signature for avoiding repeated extraction/indexing reruns."""
    data = uploaded_file.getvalue()
    return {
        "name": Path(uploaded_file.name).name,
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def get_tree_path(md_path):
    """Get the expected tree JSON path for a given .md file."""
    base_name = Path(md_path).stem
    return os.path.join(str(TREES_DIR), f"{base_name}_structure.json")


def load_index():
    """Load the shared FAISS index if it exists."""
    save_path = str(INDEX_DIR)
    if os.path.exists(save_path):
        embeddings = GeminiEmbeddings()
        try:
            require_trusted_faiss_deserialization(save_path, "DC_TRUST_FAISS_INDEX")
            return FAISS.load_local(
                save_path, embeddings, allow_dangerous_deserialization=True
            )
        except Exception as exc:
            logging.warning("Could not load trusted local FAISS index: %s", exc)
            return None
    return None


# ── Session State Init ──────────────────────────────────────────────────

if "stage" not in st.session_state:
    st.session_state.stage = "upload"         # upload → validate → compare → report
    st.session_state.doc1_path = None
    st.session_state.doc2_path = None
    st.session_state.doc1_md_path = None
    st.session_state.doc2_md_path = None
    st.session_state.doc1_id = None
    st.session_state.doc2_id = None
    st.session_state.criteria = ""
    st.session_state.doc_type = "document"
    st.session_state.validation_result = None
    st.session_state.criteria_validated = False
    st.session_state.depth_mode = "Standard"
    st.session_state.report_md = ""
    st.session_state.vector_db = None
    st.session_state.prepared_inputs = None
    st.session_state.is_running = False


# ── Sidebar: Upload & Settings ──────────────────────────────────────────

with st.sidebar:
    st.markdown("""
        <div style="margin-top: -30px; margin-bottom: 20px;">
            <div style="font-size: 32px; font-weight: bold; line-height: 1.2; color: #e2e8f0;">Proxy-Pointer</div>
            <div style="font-size: 18px; color: #cbd5e1; font-weight: 500;">Document Comparator</div>
        </div>
    """, unsafe_allow_html=True)

    doc1_file = st.file_uploader(
        "📎 Document 1 (.pdf, .md allowed)",
        type=["pdf", "md"],
        key="doc1_upload",
    )
    doc2_file = st.file_uploader(
        "📎 Document 2 (.pdf, .md allowed)",
        type=["pdf", "md"],
        key="doc2_upload",
    )

    criteria_input = st.text_input(
        "📋 Criteria",
        value=st.session_state.criteria if st.session_state.criteria else "",
        placeholder='e.g. "Legal and financial risk analysis"',
    )

    # No longer selecting depth mode, using defaults

    compare_clicked = st.button(
        "🔍 Compare",
        type="primary",
        disabled=(doc1_file is None or doc2_file is None or not criteria_input.strip()),
        use_container_width=True,
    )

    if compare_clicked:
        st.session_state.is_running = True
        st.session_state.criteria = criteria_input.strip()
        st.session_state.validation_result = None
        st.session_state.criteria_validated = False

# Show validation results if we have them
if st.session_state.validation_result and not st.session_state.criteria_validated:
    vr = st.session_state.validation_result

    if vr.get("feasible"):
        st.success(f"✅ Criteria validated. Document type: **{vr.get('document_type_detected', 'document')}**")
    else:
        st.warning(f"⚠️ Criteria may not be suitable: {vr.get('reason', '')}")
        suggestions = vr.get("suggested_criteria", [])
        if suggestions:
            st.info("💡 Suggested alternatives:\n" + "\n".join(f"  - {s}" for s in suggestions))

    revised_criteria = st.text_area(
        "Revise criteria (or keep as-is):",
        value=criteria_input,
        key="revised_criteria"
    )

    col_rev, col_proceed = st.columns(2)
    if col_rev.button("🔄 Re-validate", use_container_width=True):
        st.session_state.criteria = revised_criteria
        st.session_state.validation_result = None
        st.session_state.is_running = True
        st.rerun()

    if col_proceed.button("▶️ Proceed with Comparison", type="primary", use_container_width=True):
        st.session_state.criteria = revised_criteria
        st.session_state.criteria_validated = True
        st.session_state.doc_type = vr.get("document_type_detected", "document")
        st.session_state.is_running = True
        st.rerun()


if st.session_state.get("is_running", False) and doc1_file and doc2_file and st.session_state.criteria:

    # ── Progress tracking ────────────────────────────────────────────────
    progress_bar = st.progress(0)
    status_text = st.empty()
    report_container = st.empty()

    def update_progress(pct, msg):
        progress_bar.progress(min(pct, 100))
        status_text.markdown(f"**{msg}**")

    # ── Phase 1-3: Prepare reusable inputs (save, convert, tree, index) ────
    prep_key = (uploaded_file_signature(doc1_file), uploaded_file_signature(doc2_file))
    prepared = st.session_state.get("prepared_inputs")

    if prepared and prepared.get("key") == prep_key:
        update_progress(15, "♻️ Reusing prepared documents and vector index...")
        doc1_path = prepared["doc1_path"]
        doc2_path = prepared["doc2_path"]
        doc1_md = prepared["doc1_md"]
        doc2_md = prepared["doc2_md"]
        doc1_tree = prepared["doc1_tree"]
        doc2_tree = prepared["doc2_tree"]
        vector_db = prepared["vector_db"]
        doc1_id = prepared["doc1_id"]
        doc2_id = prepared["doc2_id"]
    else:
        # ── Phase 1: Save & Convert (10%) ────────────────────────────────
        update_progress(2, "📥 Saving uploaded files...")

        doc1_path = save_uploaded_file(doc1_file, str(UPLOADS_DIR))
        doc2_path = save_uploaded_file(doc2_file, str(UPLOADS_DIR))

        update_progress(5, "📝 Converting documents to Markdown...")

        os.makedirs(str(DOCUMENTS_DIR), exist_ok=True)
        doc1_md = extract_to_md(doc1_path, str(DOCUMENTS_DIR))
        doc2_md = extract_to_md(doc2_path, str(DOCUMENTS_DIR))

        if not doc1_md or not doc2_md:
            st.error("❌ Failed to convert one or both documents to Markdown.")
            st.stop()

        # ── Phase 2: Build Trees (15%) ───────────────────────────────────
        update_progress(10, "🌳 Building document structure trees...")

        os.makedirs(str(TREES_DIR), exist_ok=True)
        build_skeleton_trees(str(DOCUMENTS_DIR), str(TREES_DIR))

        doc1_tree = get_tree_path(doc1_md)
        doc2_tree = get_tree_path(doc2_md)

        if not os.path.exists(doc1_tree) or not os.path.exists(doc2_tree):
            st.error("❌ Failed to build structure trees for one or both documents.")
            st.stop()

        # ── Phase 3: Build Index (25%) ───────────────────────────────────
        update_progress(15, "📊 Building vector index...")

        vector_db, doc_ids = build_comparator_index(
            md_paths=[doc1_md, doc2_md],
            incremental=True,
            progress_callback=lambda msg: update_progress(20, msg)
        )

        if vector_db is None:
            st.error("❌ Failed to build vector index.")
            st.stop()

        doc1_id = get_doc_id(doc1_md)
        doc2_id = get_doc_id(doc2_md)
        st.session_state.prepared_inputs = {
            "key": prep_key,
            "doc1_path": doc1_path,
            "doc2_path": doc2_path,
            "doc1_md": doc1_md,
            "doc2_md": doc2_md,
            "doc1_tree": doc1_tree,
            "doc2_tree": doc2_tree,
            "vector_db": vector_db,
            "doc1_id": doc1_id,
            "doc2_id": doc2_id,
        }

    st.session_state.doc1_path = doc1_path
    st.session_state.doc2_path = doc2_path
    st.session_state.doc1_md_path = doc1_md
    st.session_state.doc2_md_path = doc2_md
    st.session_state.vector_db = vector_db
    st.session_state.doc1_id = doc1_id
    st.session_state.doc2_id = doc2_id

    # ── Phase 4: Validate Criteria (30%) ─────────────────────────────────
    update_progress(28, "🧪 Validating comparison criteria...")

    if not st.session_state.get("criteria_validated", False):
        validation = validate_criteria(
            st.session_state.criteria,
            doc1_tree, doc1_md,
            doc2_tree, doc2_md
        )
        st.session_state.validation_result = validation
        st.session_state.doc_type = validation.get("document_type_detected", "document")

        if not validation.get("feasible"):
            # Show validation UI and stop — user needs to revise or accept
            st.session_state.criteria_validated = False
            st.session_state.is_running = False
            progress_bar.empty()
            status_text.empty()
            st.rerun()

        st.session_state.criteria_validated = True

    # ── Phase 5: Select Doc 1 Sections (35%) ─────────────────────────────
    doc1_name = Path(doc1_md).stem
    doc2_name = Path(doc2_md).stem
    update_progress(32, f"🔎 Identifying relevant sections in {doc1_name}...")

    criteria = st.session_state.criteria
    doc_type = st.session_state.doc_type

    # Build stage-specific prompts
    selection_query = build_section_selection_query(criteria, doc_type)
    comparison_prompt = build_comparison_prompt(criteria, doc_type, doc1_name, doc2_name)

    doc1_sections = select_relevant_sections(
        vector_db, doc1_id, selection_query, criteria, doc_type,
        k_final=MAX_DOC1_SECTIONS
    )

    if not doc1_sections:
        st.error(f"❌ No relevant sections found in {doc1_name}.")
        st.stop()

    update_progress(35, f"Found {len(doc1_sections)} relevant sections in {doc1_name}")

    # Load full text for each Doc 1 section
    doc1_md_path = resolve_md_path_for_doc_id(doc1_id, str(DOCUMENTS_DIR))
    for sec in doc1_sections:
        full_text = load_full_section_text(
            doc1_id, sec["start_line"], sec["end_line"], md_path=doc1_md_path
        )
        sec["full_text"] = full_text or sec.get("content", "")

    # ── Phase 6: Compare Sections (35% → 90%) ───────────────────────────
    total_comparisons = 0
    all_ratings = {"RED": 0, "YELLOW": 0, "GREEN": 0, "UNKNOWN": 0}
    section_blocks = []
    compared_node_ids = set()

    num_sections = len(doc1_sections)
    comparison_base_pct = 35
    comparison_range_pct = 55  # 35% to 90%

    report_so_far = ""

    for sec_idx, doc1_sec in enumerate(doc1_sections):
        sec_num = sec_idx + 1
        sec_pct = comparison_base_pct + int(
            (sec_idx / num_sections) * comparison_range_pct
        )
        update_progress(
            sec_pct,
            f"📋 Comparing section {sec_num}/{num_sections}: {doc1_sec.get('title', '')}"
        )

        compared_node_ids.add(doc1_sec.get("node_id"))

        # Cross-retrieve Doc 2 matches
        cross_query = build_cross_retrieval_query(
            doc1_sec.get("breadcrumb", ""),
            doc1_sec.get("full_text", ""),
            criteria
        )

        doc2_matches = retrieve_matching_sections(
            vector_db, doc2_id, cross_query,
            doc1_breadcrumb=doc1_sec.get("breadcrumb", ""),
            criteria=criteria, doc_type=doc_type,
            k_final=MAX_DOC2_MATCHES
        )

        if not doc2_matches:
            logging.info(f"  [SKIP] No Doc 2 equivalents found for: {doc1_sec.get('title')}")
            continue

        # Compare each Doc 2 match
        comparison_results = []
        for match_idx, doc2_match in enumerate(doc2_matches):
            update_progress(
                sec_pct + int(
                    ((match_idx + 1) / len(doc2_matches)) *
                    (comparison_range_pct / num_sections)
                ),
                f"📋 {doc1_sec.get('title', '')} vs {doc2_match.get('title', '')}..."
            )

            result = compare_sections(comparison_prompt, doc1_sec, doc2_match, doc1_name=doc1_name, doc2_name=doc2_name, doc_type=doc_type)
            rating = extract_rating(result)

            # Keep both the result and the matched doc2 section aligned
            comparison_results.append((result, doc2_match))

            # Track ratings
            all_ratings[rating] = all_ratings.get(rating, 0) + 1
            total_comparisons += 1

        # Filter out 'UNRELATED' matches so they don't clutter the report
        filtered_pairs = [pair for pair in comparison_results if extract_rating(pair[0]) != "UNRELATED"]

        # If no matches were found at all (empty list), skip this Doc 1 section block
        if not filtered_pairs:
            continue

        # Split back out to pass to build_section_block
        filtered_results = [pair[0] for pair in filtered_pairs]
        filtered_doc2_matches = [pair[1] for pair in filtered_pairs]

        # Build the complete section block using dynamic report numbers
        section_md = build_section_block(
            len(section_blocks) + 1, doc1_sec, filtered_doc2_matches,
            filtered_results, criteria,
            doc1_name=doc1_name, doc2_name=doc2_name, doc_type=doc_type
        )

        section_blocks.append(section_md)

        # Stream update
        report_so_far += section_md
        report_container.markdown(report_so_far, unsafe_allow_html=True)

    # ── Phase 7: Assemble Report (95%) ───────────────────────────────────
    update_progress(92, "📝 Assembling final report...")

    executive_summary = build_executive_summary(
        criteria, doc1_name, doc2_name, doc_type,
        len(section_blocks), total_comparisons, all_ratings
    )

    final_report = assemble_report(executive_summary, section_blocks)

    st.session_state.report_md = final_report

    # ── Done ─────────────────────────────────────────────────────────────
    update_progress(100, "✅ Comparison complete!")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    report_container.empty()

    st.session_state.is_running = False
    st.rerun()


# ── STAGE 3: Display Report ─────────────────────────────────────────────

if st.session_state.report_md:
    st.markdown(st.session_state.report_md, unsafe_allow_html=True)

    st.markdown("---")

    # Download button
    st.download_button(
        label="📥 Download Full Report (.md)",
        data=st.session_state.report_md,
        file_name="comparison_report.md",
        mime="text/markdown",
        use_container_width=True,
    )

    # Reset button
    if st.button("🔄 New Comparison", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
