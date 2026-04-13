"""
CogInfera — Streamlit UI
Professional interface for the Agentic RAG System with live intermediate stage panels.
"""

import json
import time
import httpx
import streamlit as st

API_BASE = "http://localhost:8000"

# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="CogInfera | Agentic RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Main background */
.stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid #30363d;
}

/* Cards */
.stage-card {
    background: rgba(22, 27, 34, 0.85);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 10px 0;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}
.stage-card:hover { border-color: #58a6ff; transform: translateY(-1px); }
.stage-card.running { border-color: #f0c27f; box-shadow: 0 0 12px rgba(240,194,127,0.2); }
.stage-card.done    { border-color: #3fb950; }

/* Answer box */
.answer-box {
    background: linear-gradient(135deg, rgba(31,111,235,0.12) 0%, rgba(63,185,80,0.08) 100%);
    border: 1px solid #1f6feb;
    border-radius: 16px;
    padding: 24px 28px;
    margin-top: 20px;
    box-shadow: 0 0 30px rgba(31,111,235,0.15);
}

/* Confidence bar */
.conf-bar-bg { background:#21262d; border-radius:4px; height:8px; margin-top:6px; }
.conf-bar    { height:8px; border-radius:4px;
               background:linear-gradient(90deg,#238636,#3fb950); }

/* Badges */
.badge {
    display:inline-block; padding:2px 10px; border-radius:20px;
    font-size:0.75rem; font-weight:600; margin-right:6px;
}
.badge-blue   { background:rgba(31,111,235,0.2); color:#58a6ff; border:1px solid #1f6feb; }
.badge-green  { background:rgba(63,185,80,0.2);  color:#3fb950; border:1px solid #238636; }
.badge-yellow { background:rgba(240,194,127,0.2);color:#f0c27f; border:1px solid #9e6a03; }
.badge-purple { background:rgba(138,80,255,0.2); color:#bc8cff; border:1px solid #6e40c9; }

/* Chunk card */
.chunk-card {
    background:#161b22; border:1px solid #30363d; border-radius:8px;
    padding:10px 14px; margin:6px 0; font-size:0.82rem;
}
.chunk-meta { color:#8b949e; font-size:0.75rem; margin-bottom:6px; }

/* Headings */
h1 { background:linear-gradient(90deg,#58a6ff,#bc8cff);
     -webkit-background-clip:text; -webkit-text-fill-color:transparent; }

/* Input */
[data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input {
    background:#0d1117 !important; color:#e6edf3 !important;
    border:1px solid #30363d !important; border-radius:8px !important;
}
[data-testid="stTextArea"] textarea:focus, [data-testid="stTextInput"] input:focus {
    border-color:#58a6ff !important;
    box-shadow:0 0 0 2px rgba(88,166,255,0.15) !important;
}

/* Buttons */
.stButton > button {
    background:linear-gradient(135deg,#1f6feb,#388bfd) !important;
    color:#fff !important; border:none !important; border-radius:8px !important;
    font-weight:600 !important; padding:10px 24px !important;
    transition:all 0.2s ease !important;
}
.stButton > button:hover { transform:translateY(-1px); box-shadow:0 4px 16px rgba(31,111,235,0.4) !important; }

/* Expander */
[data-testid="stExpander"] {
    background:rgba(22,27,34,0.6) !important;
    border:1px solid #30363d !important;
    border-radius:10px !important;
}

/* Status dot */
.dot-green { color:#3fb950; font-size:1.1rem; }
.dot-red   { color:#f85149; font-size:1.1rem; }
.dot-yellow{ color:#f0c27f; font-size:1.1rem; }

/* Divider */
hr { border-color:#21262d !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────
def get_status():
    try:
        r = httpx.get(f"{API_BASE}/status", timeout=30)
        return r.json()
    except Exception:
        return None


def badge(text, kind="blue"):
    return f'<span class="badge badge-{kind}">{text}</span>'


def stage_icon(stage):
    return {
        "planning":      "📋",
        "retrieval":     "🔍",
        "compression":   "🗜️",
        "graph_reasoning":"🕸️",
        "multi_pass":    "🧠",
        "validation":    "✅",
        "answer":        "💬",
        "error":         "❌",
    }.get(stage, "⚙️")


def stage_label(stage):
    return {
        "planning":       "Query Planning & Decomposition",
        "retrieval":      "Agentic Hybrid Retrieval",
        "compression":    "Context Compression",
        "graph_reasoning":"Dynamic Graph Reasoning",
        "multi_pass":     "Multi-Pass LLM Reasoning",
        "validation":     "Self-RAG Validation",
        "answer":         "Final Answer",
    }.get(stage, stage.replace("_", " ").title())


# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 CogInfera")
    st.markdown("*Agentic Hierarchical Graph-Augmented RAG*")
    st.divider()

    # Status
    status = get_status()
    if status is None:
        st.markdown('<span class="dot-red">●</span> **API Offline** — start FastAPI first', unsafe_allow_html=True)
        st.code("py -3.14 -m uvicorn api.main:app --reload")
        st.stop()
    elif status["loaded"]:
        st.markdown(f'<span class="dot-green">●</span> **Index Loaded** — {status["chunk_count"]} chunks', unsafe_allow_html=True)
    else:
        st.markdown('<span class="dot-yellow">●</span> **No Document Indexed**', unsafe_allow_html=True)

    st.divider()

    # Ingest
    st.markdown("### 📄 Ingest a PDF")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
    if uploaded and st.button("🚀 Ingest Document", use_container_width=True):
        with st.spinner("Ingesting… this may take a few minutes"):
            try:
                r = httpx.post(
                    f"{API_BASE}/ingest",
                    files={"file": (uploaded.name, uploaded.read(), "application/pdf")},
                    timeout=600,
                )
                if r.status_code == 200:
                    res = r.json()
                    st.success(f"✅ {res['pages']} pages · {res['chunks']} chunks indexed")
                    if res.get("summary"):
                        st.info(res["summary"][:300] + "…")
                    st.rerun()
                else:
                    st.error(r.json().get("detail", "Ingest failed"))
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()

    # Settings
    st.markdown("### ⚙️ Options")
    show_intermediate = st.toggle("Show Intermediate Steps", value=True)
    show_chunks = st.toggle("Show Retrieved Chunks", value=False)

    st.divider()
    st.markdown(
        '<div style="color:#8b949e;font-size:0.75rem;">'
        'Model: qwen/qwen3.5-9b<br>'
        'Embedder: BAAI/bge-small-en<br>'
        'Reranker: ms-marco-MiniLM-L-6-v2'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Main Area ────────────────────────────────────────────────────────
st.markdown("# CogInfera")
st.markdown("*Ask anything about your ingested documents. The system will reason, retrieve, and validate its answer.*")
st.divider()

question = st.text_area(
    "Your question",
    placeholder="e.g. What is a deadlock and how can it be prevented?",
    height=90,
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 5])
with col1:
    submit = st.button("🔍 Ask", use_container_width=True)

if submit and not status["loaded"]:
    st.warning("Please ingest a document first using the sidebar.")
    submit = False

if submit and question.strip():
    st.divider()

    # Containers for each stage
    stage_containers = {}
    final_answer_placeholder = st.empty()

    # Pre-create stage containers if intermediate steps are on
    if show_intermediate:
        stages_order = ["planning", "retrieval", "compression", "graph_reasoning", "multi_pass", "validation"]
        for s in stages_order:
            stage_containers[s] = st.container()
            with stage_containers[s]:
                st.markdown(
                    f'<div class="stage-card" id="{s}">'
                    f'{stage_icon(s)} <strong>{stage_label(s)}</strong> '
                    f'<span style="color:#8b949e;font-size:0.8rem;float:right;">waiting…</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # Stream from API
    stage_data = {}
    try:
        with httpx.stream(
            "POST",
            f"{API_BASE}/query/stream",
            json={"question": question.strip()},
            # Disable read timeout — LLM calls can take minutes between chunks.
            # Keep connect/write timeouts reasonable.
            timeout=httpx.Timeout(connect=30.0, read=None, write=30.0, pool=30.0),
        ) as stream:
            for line in stream.iter_lines():
                if line.startswith(":"):   # SSE keepalive comment — skip
                    continue
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if raw == "[DONE]":
                    break

                event = json.loads(raw)
                stage = event.get("stage", "")
                s_status = event.get("status", "")
                data = event.get("data", {})
                stage_data[stage] = data

                if stage == "error":
                    st.error(data.get("message", "Unknown error"))
                    break

                if stage == "answer" and s_status == "done":
                    # Render final answer
                    answer_text = data.get("answer", "")
                    # Highlight citations like [[Pages X-Y]]
                    import re
                    highlighted = re.sub(
                        r"\[\[Pages? (\d+[-–]\d+)\]\]",
                        r'<span class="badge badge-blue">📄 p.\1</span>',
                        answer_text,
                    )
                    final_answer_placeholder.markdown(
                        f'<div class="answer-box">'
                        f'<h3 style="margin-top:0;color:#58a6ff;">💬 Answer</h3>'
                        f'<div style="color:#e6edf3;line-height:1.8;">{highlighted}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    continue

                if not show_intermediate or s_status != "done":
                    continue

                # Render each stage when done
                container = stage_containers.get(stage)
                if not container:
                    continue

                with container:
                    # Clear and re-render with real content
                    container.empty()

                    if stage == "planning":
                        complexity = data.get("complexity", "?")
                        sub_q = data.get("sub_queries", [])
                        color = {"simple": "green", "multi_hop": "yellow", "comparative": "purple"}.get(complexity, "blue")
                        html = (
                            f'<div class="stage-card done">'
                            f'{stage_icon(stage)} <strong>{stage_label(stage)}</strong> '
                            f'{badge(complexity, color)}<br><br>'
                        )
                        for i, q in enumerate(sub_q, 1):
                            html += f'<div style="color:#c9d1d9;margin:4px 0;">Sub-query {i}: <em>"{q}"</em></div>'
                        html += '</div>'
                        st.markdown(html, unsafe_allow_html=True)

                    elif stage == "retrieval":
                        count = data.get("count", 0)
                        top_chunks = data.get("top_chunks", [])
                        html = (
                            f'<div class="stage-card done">'
                            f'{stage_icon(stage)} <strong>{stage_label(stage)}</strong> '
                            f'{badge(f"{count} chunks retrieved", "green")}<br><br>'
                        )
                        html += '</div>'
                        st.markdown(html, unsafe_allow_html=True)
                        if show_chunks and top_chunks:
                            with st.expander(f"📄 Top {len(top_chunks)} Retrieved Chunks"):
                                for c in top_chunks:
                                    st.markdown(
                                        f'<div class="chunk-card">'
                                        f'<div class="chunk-meta">Pages {c["pages"]} · {c["section"]} · Score: {c["score"]}</div>'
                                        f'{c["text"]}…</div>',
                                        unsafe_allow_html=True,
                                    )

                    elif stage == "compression":
                        ctx = data.get("compressed_context", "")
                        html = (
                            f'<div class="stage-card done">'
                            f'{stage_icon(stage)} <strong>{stage_label(stage)}</strong> '
                            f'{badge(f"{len(ctx.split())} words", "purple")}</div>'
                        )
                        st.markdown(html, unsafe_allow_html=True)
                        with st.expander("🗜️ Compressed Context"):
                            st.markdown(f'<div style="color:#c9d1d9;font-size:0.85rem;white-space:pre-wrap;">{ctx}</div>', unsafe_allow_html=True)

                    elif stage == "graph_reasoning":
                        entities = data.get("entities", [])
                        rels = data.get("relationships", [])
                        paths = data.get("paths", [])
                        html = (
                            f'<div class="stage-card done">'
                            f'{stage_icon(stage)} <strong>{stage_label(stage)}</strong> '
                            f'{badge(f"{len(entities)} entities", "blue")} '
                            f'{badge(f"{len(rels)} relationships", "purple")} '
                            f'{badge(f"{len(paths)} paths", "yellow")}</div>'
                        )
                        st.markdown(html, unsafe_allow_html=True)
                        with st.expander("🕸️ Graph Details"):
                            if entities:
                                st.markdown("**Entities:**")
                                for e in entities:
                                    st.markdown(f"- `{e['name']}` ({e.get('type','?')}) — {e.get('description','')}")
                            if rels:
                                st.markdown("**Relationships:**")
                                for r in rels:
                                    st.markdown(f"- {r['source']} → **{r['relation']}** → {r['target']}")
                            if paths:
                                st.markdown("**Multi-hop Paths:**")
                                for p in paths:
                                    st.code(p)

                    elif stage == "multi_pass":
                        html = (
                            f'<div class="stage-card done">'
                            f'{stage_icon(stage)} <strong>{stage_label(stage)}</strong> '
                            f'{badge("Pass 1: Facts", "blue")} '
                            f'{badge("Pass 2: Relations", "purple")} '
                            f'{badge("Pass 3: Synthesis", "green")}</div>'
                        )
                        st.markdown(html, unsafe_allow_html=True)
                        with st.expander("🧠 Reasoning Passes"):
                            st.markdown("**Pass 1 — Fact Extraction:**")
                            st.markdown(data.get("facts", ""))
                            st.divider()
                            st.markdown("**Pass 2 — Relationship Mapping:**")
                            st.markdown(data.get("relationships", ""))

                    elif stage == "validation":
                        conf = data.get("confidence", 0)
                        grounded = data.get("is_grounded", False)
                        complete = data.get("is_complete", False)
                        needs_retry = data.get("needs_retry", False)
                        issues = data.get("issues", [])
                        g_badge = badge("Grounded ✓", "green") if grounded else badge("Not Grounded ✗", "yellow")
                        c_badge = badge("Complete ✓", "green") if complete else badge("Incomplete ✗", "yellow")
                        r_badge = badge("Refined", "yellow") if needs_retry else ""
                        html = (
                            f'<div class="stage-card done">'
                            f'{stage_icon(stage)} <strong>{stage_label(stage)}</strong> '
                            f'{g_badge} {c_badge} {r_badge}<br><br>'
                            f'<strong style="color:#8b949e;font-size:0.8rem;">Confidence</strong><br>'
                            f'<div class="conf-bar-bg"><div class="conf-bar" style="width:{int(conf*100)}%"></div></div>'
                            f'<div style="color:#8b949e;font-size:0.75rem;margin-top:3px;">{int(conf*100)}%</div>'
                        )
                        if issues:
                            html += '<br><strong style="color:#f0c27f;font-size:0.8rem;">Issues:</strong><ul style="color:#f0c27f;font-size:0.8rem;">'
                            for issue in issues:
                                html += f"<li>{issue}</li>"
                            html += "</ul>"
                        html += "</div>"
                        st.markdown(html, unsafe_allow_html=True)

    except httpx.ConnectError:
        st.error("❌ Cannot connect to API. Start the backend with: `py -3.14 -m uvicorn api.main:app --reload`")

elif submit and not question.strip():
    st.warning("Please enter a question.")
