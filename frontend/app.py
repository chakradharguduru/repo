"""
app.py — AXE Repository Assistant frontend.

Run with:  streamlit run app.py
(backend.py must already be running, e.g. `uvicorn backend:app --reload`)
"""
import time
from pathlib import Path

import requests
import streamlit as st

# =====================================================
# Configuration
# =====================================================
BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="AXE Repository Assistant",
    page_icon="🤖",
    layout="wide",
)

# =====================================================
# Styling
# =====================================================
st.markdown(
    """
    <style>
    .stChatMessage { border-radius: 12px; }
    .source-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
        background: #eef1f7;
        color: #333;
    }
    .metric-card {
        background: #f7f8fa;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    code { white-space: pre-wrap !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# Backend helpers
# =====================================================
@st.cache_data(ttl=10)
def check_backend_health():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


@st.cache_data(ttl=15)
def get_stats():
    try:
        r = requests.get(f"{BACKEND_URL}/stats", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def ask_backend(query: str, top_k: int):

    history = [
        {
            "role": msg["role"],
            "content": msg["content"]
        }
        for msg in st.session_state.messages[-8:]
    ]

    response = requests.post(
        f"{BACKEND_URL}/chat",
        json={
            "query": query,
            "top_k": top_k,
            "history": history
        },
        timeout=180,
    )

    response.raise_for_status()

    return response.json()

# =====================================================
# Session state
# =====================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# =====================================================
# Sidebar
# =====================================================
with st.sidebar:
    st.header("⚙️ Settings")

    healthy = check_backend_health()
    if healthy:
        st.success("Backend connected")
    else:
        st.error("Backend unreachable — is `uvicorn backend:app` running?")

    stats = get_stats() if healthy else None
    if stats:
        st.markdown("**Index stats**")
        c1, c2 = st.columns(2)
        c1.metric("Chunks", stats["total_chunks"])
        c2.metric("Files", stats["file_count"])
        if stats["layers"]:
            st.caption("Layers: " + ", ".join(stats["layers"]))
        if stats["domains"]:
            st.caption("Domains: " + ", ".join(stats["domains"][:12]) +
                       (" ..." if len(stats["domains"]) > 12 else ""))

    st.divider()

    top_k = st.slider("Chunks to retrieve (top_k)", min_value=1, max_value=10, value=5)

    st.divider()
    st.markdown("**Try an example**")
    examples = [
        "What transformations are happening in givex?",
        "How is brand derived in the ezcater transform?",
        "What pipelines write to the curated layer?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending_prompt = ex

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.messages:
        transcript = "\n\n".join(
            f"**{m['role'].capitalize()}:** {m['content']}" for m in st.session_state.messages
        )
        st.download_button(
            "⬇️ Download transcript",
            data=transcript,
            file_name="axe_chat_transcript.md",
            mime="text/markdown",
            use_container_width=True,
        )

# =====================================================
# Header
# =====================================================
st.title("🤖 AXE Repository Assistant")
st.caption(
    "Ask questions about the AXE Repository, Transformations, Pipelines, Spark Logic and Databricks."
)

# =====================================================
# Helper: render sources
# =====================================================
def display_sources(sources, elapsed=None):
    if not sources:
        return

    label = f"📄 Sources Used ({len(sources)})"
    if elapsed is not None:
        label += f" · answered in {elapsed}s"

    with st.expander(label):
        for source in sources:
            filename = Path(source["file_path"]).name
            st.markdown(f"#### 📄 {filename}")

            badges = ""
            if source.get("layer"):
                badges += f"<span class='source-badge'>layer: {source['layer']}</span>"
            if source.get("domain"):
                badges += f"<span class='source-badge'>domain: {source['domain']}</span>"
            if source.get("chunk_name"):
                badges += f"<span class='source-badge'>chunk: {source['chunk_name']}</span>"
            st.markdown(badges, unsafe_allow_html=True)

            st.code(source["file_path"], language="text")

            if source.get("preview"):
                with st.popover("Preview snippet"):
                    st.code(source["preview"], language="python")

            st.markdown("---")

# =====================================================
# Chat history
# =====================================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            display_sources(message.get("sources", []), message.get("elapsed_seconds"))

# =====================================================
# Chat input (regular box OR a sidebar example button)
# =====================================================
prompt = st.chat_input("Ask anything about the repository...")
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()

        answer = "No answer returned."
        sources = []
        elapsed = None

        with st.spinner("Searching repository..."):
            try:
                t0 = time.time()
                data = ask_backend(prompt, top_k=top_k)
                elapsed = data.get("elapsed_seconds", round(time.time() - t0, 2))
                answer = data.get("answer", answer)
                sources = data.get("sources", [])
            except requests.exceptions.Timeout:
                answer = "❌ The backend timed out. Try a smaller `top_k` or check the server logs."
            except Exception as e:
                answer = f"❌ Backend error\n\n```\n{e}\n```"

        placeholder.markdown(answer)
        display_sources(sources, elapsed)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "elapsed_seconds": elapsed,
    })