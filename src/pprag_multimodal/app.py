import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
import re
from pathlib import Path
import streamlit as st
from PIL import Image, UnidentifiedImageError

from pprag_multimodal.agent.mm_rag_bot import MultimodalProxyPointerRAG
from pprag_multimodal.config import INDEX_DIR, TREES_DIR, DATASET_DIR

# --- UI Layout ---
st.set_page_config(page_title="Proxy-Pointer MultiModal", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* GLOBAL APP STYLING */
    .main { background-color: #fcfcfc; }

    /* CHAT CONTAINER SPACING */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 0rem !important;
    }

    /* MODIFIED CHAT BUBBLES */
    .stChatMessage {
        padding: 0.5rem 1rem !important;
        margin-bottom: 0.4rem !important;
        border-radius: 12px;
    }

    /* GALLERY ALIGNMENT LOCK */
    [data-testid="column"] {
        display: flex;
        flex-direction: column;
        justify-content: flex-start !important;
        align-items: center;
        padding: 10px;
    }

    /* IMAGE CONTAINER */
    .img-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 15px;
        width: 100%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    .evidence-list {
        background-color: #f8fafc;
        padding: 15px;
        border-left: 4px solid #3b82f6;
        border-radius: 4px;
        margin: 15px 0;
    }

    /* INCREASE IMAGE CAPTION SIZE (UNDER THE IMAGE) */
    div[data-testid="stImageCaption"],
    div[data-testid="stImageCaption"] * {
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        color: #000000 !important;
        line-height: 1.5 !important;
        opacity: 1.0 !important;
    }

    /* Fallback for newer Streamlit versions */
    div[data-testid="stImage"] > div:last-child {
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        color: #000000 !important;
        opacity: 1.0 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_bot():
    return MultimodalProxyPointerRAG(str(INDEX_DIR), str(TREES_DIR), str(DATASET_DIR))

bot = load_bot()

# --- APP FLOW ---
st.title("📦 MultiModal Proxy-Pointer RAG")
st.caption("Structural paper exploration with AI-verified visual evidence")

MAX_HISTORY = 50

if "messages" not in st.session_state:
    st.session_state.messages = []


def _trim_messages():
    st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]

def clean_response_text(text):
    return re.sub(r"\[SHOW:.*?\]", "", text, flags=re.IGNORECASE).strip()

def _safe_image_name(img):
    return Path(img.get("full_path", "")).name or img.get("relative_path", "image")


def _render_one_image(img):
    try:
        with Image.open(img["full_path"]) as opened:
            st.image(opened.copy(), caption=img["label"], use_container_width=True)
    except (FileNotFoundError, PermissionError, UnidentifiedImageError, OSError) as exc:
        logging.warning("Unable to render image %s: %s", img.get("full_path"), exc)
        st.caption(f"Image unavailable: {_safe_image_name(img)}")


def render_images(images):
    if not images:
        return

    count = len(images)
    st.markdown("---")

    # 1. Text-based Evidence List (The "List" the user requested)
    st.markdown("**AI-Verified Evidence List:**")
    for img in images:
        st.markdown(f"• **{img['label']}** ({_safe_image_name(img)})")

    st.markdown(" ")

    # 2. Visual Gallery
    if count == 1:
        if images[0]["exists"]:
            _render_one_image(images[0])

    elif count == 2:
        cols = st.columns(2)
        for i in range(2):
            if images[i]["exists"]:
                with cols[i]:
                    _render_one_image(images[i])

    else:
        for row_start in range(0, count, 3):
            cols = st.columns(3)
            for i in range(3):
                idx = row_start + i
                if idx < count:
                    img = images[idx]
                    if img["exists"]:
                        with cols[i]:
                            _render_one_image(img)

def render_message(msg):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            if "paths" in msg and msg["paths"]:
                st.markdown("**Sources:**")
                for p in msg["paths"][:3]:
                    st.markdown(f"• {p}")
                if len(msg["paths"]) > 3:
                    with st.expander("Full Context", expanded=False):
                        for p in msg["paths"]:
                            st.markdown(f"• {p}")

            render_images(msg.get("images", []))

for message in st.session_state.messages:
    render_message(message)

if prompt := st.chat_input("Ask a question..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    _trim_messages()

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                result = bot.chat(prompt)
                clean_text = clean_response_text(result["text"])

                st.markdown(clean_text)

                if result.get("paths"):
                    st.markdown("**Sources:**")
                    for p in result["paths"][:3]:
                        st.markdown(f"• {p}")
                    if len(result["paths"]) > 3:
                        with st.expander("Full Context", expanded=False):
                            for p in result["paths"]:
                                st.markdown(f"• {p}")

                render_images(result.get("images", []))

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": clean_text,
                    "images": result.get("images", []),
                    "paths": result.get("paths", [])
                })
                _trim_messages()
            except Exception:
                logging.exception("Multimodal chat failed")
                error_text = "Sorry, I couldn't complete that request. Please try again."
                st.markdown(error_text)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_text,
                    "images": [],
                    "paths": [],
                })
                _trim_messages()
