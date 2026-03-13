"""
frontend/streamlit_app.py — RAGify AI Frontend
Makani Germany RAG Fashion Assistant

Run:
    streamlit run frontend/streamlit_app.py
"""

import json
import uuid
from typing import Optional

import requests
import streamlit as st

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="RAGify AI · Makani Germany",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS — white bg, black text, black buttons
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

:root {
    --white:    #FFFFFF;
    --black:    #0A0A0A;
    --grey-50:  #F7F7F7;
    --grey-100: #EFEFEF;
    --grey-200: #E0E0E0;
    --grey-400: #9E9E9E;
    --grey-600: #616161;
    --radius:   10px;
}

/* ── Base ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main .block-container {
    background: var(--white) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: var(--black) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    display: none !important;
    visibility: hidden !important;
}

/* ── Sidebar background ── */
[data-testid="stSidebar"] {
    background: var(--grey-50) !important;
    border-right: 1px solid var(--grey-200) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small {
    color: var(--black) !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Sidebar buttons — explicit white text on black ── */
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] .stButton button {
    background-color: var(--black) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    padding: 0.5rem 1rem !important;
    width: 100% !important;
    cursor: pointer !important;
}
[data-testid="stSidebar"] button p,
[data-testid="stSidebar"] .stButton button p {
    color: #FFFFFF !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    margin: 0 !important;
}
[data-testid="stSidebar"] button:hover,
[data-testid="stSidebar"] .stButton button:hover {
    opacity: 0.75 !important;
}

/* ── Main area buttons (chips + send) ── */
.stButton button {
    background-color: var(--black) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
}
.stButton button p {
    color: #FFFFFF !important;
    font-weight: 500 !important;
    margin: 0 !important;
}
.stButton button:hover { opacity: 0.75 !important; }

/* ── Text input ── */
.stTextInput > div > div > input {
    background: var(--white) !important;
    border: 1.5px solid var(--grey-200) !important;
    border-radius: var(--radius) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
    color: var(--black) !important;
    padding: 0.7rem 1rem !important;
    box-shadow: none !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--black) !important;
    box-shadow: none !important;
}
.stTextInput > div > div > input::placeholder {
    color: var(--grey-400) !important;
}

/* ── Chat bubbles ── */
.msg-row {
    display: flex;
    margin-bottom: 1rem;
    align-items: flex-end;
    gap: 0.6rem;
}
.msg-row.user { flex-direction: row-reverse; }
.msg-row.ai   { flex-direction: row; }

.avatar {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    font-weight: 700;
    flex-shrink: 0;
}
.avatar.user { background: var(--black); color: #FFFFFF; }
.avatar.ai   { background: var(--grey-200); color: var(--black); }

.bubble {
    max-width: 72%;
    padding: 0.8rem 1rem;
    border-radius: var(--radius);
    line-height: 1.65;
    font-size: 0.9rem;
    word-break: break-word;
}
.bubble.user {
    background: var(--black);
    color: #FFFFFF;
    border-bottom-right-radius: 3px;
}
.bubble.ai {
    background: var(--grey-50);
    color: var(--black);
    border: 1px solid var(--grey-200);
    border-bottom-left-radius: 3px;
}

/* typing cursor */
.cursor {
    display: inline-block;
    width: 2px;
    height: 0.9em;
    background: var(--black);
    margin-left: 2px;
    vertical-align: text-bottom;
    animation: blink 0.75s step-end infinite;
}
@keyframes blink { 50% { opacity: 0; } }

/* ── Source pills ── */
.sources-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin-top: 0.6rem;
}
.source-pill {
    font-size: 0.68rem;
    padding: 0.18rem 0.55rem;
    border: 1px solid var(--grey-200);
    border-radius: 20px;
    color: var(--grey-600);
    background: var(--white);
    white-space: nowrap;
    max-width: 200px;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ── Sidebar section labels ── */
.sm-label {
    font-size: 0.62rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--grey-400) !important;
    margin: 1.2rem 0 0.5rem !important;
    display: block;
}

/* ── Status badges ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.72rem;
    font-weight: 500;
    padding: 0.22rem 0.65rem;
    border-radius: 20px;
    margin-bottom: 0.5rem;
}
.badge-ok  { background: #F0FFF4; color: #276749; border: 1px solid #9AE6B4; }
.badge-err { background: #FFF5F5; color: #9B2C2C; border: 1px solid #FEB2B2; }

/* ── Source cards in sidebar ── */
.src-card {
    background: var(--white);
    border: 1px solid var(--grey-200);
    border-left: 3px solid var(--black);
    border-radius: 6px;
    padding: 0.55rem 0.7rem;
    margin-bottom: 0.4rem;
    font-size: 0.78rem;
    line-height: 1.45;
}
.src-title { font-weight: 500; color: var(--black) !important; }
.src-meta  { color: var(--grey-400) !important; font-size: 0.68rem; margin-top: 0.1rem; }

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--grey-200) !important;
    margin: 0.8rem 0 !important;
}
/* Always show sidebar collapse button */
[data-testid="stSidebarCollapseButton"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
def _init():
    defaults = {
        "session_id":    str(uuid.uuid4()),
        "messages":      [],
        "api_ok":        None,
        "api_checked":   False,
        "pending_input": "",
        "last_sources":  [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ─────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────
def check_health() -> bool:
    for timeout in (2, 5):
        try:
            r = requests.get(f"{API_BASE}/health", timeout=timeout)
            if 200 <= r.status_code < 300:
                return True
        except Exception:
            pass
    return False


def _extract_token(event_type: str, data) -> Optional[str]:
    """
    Extracts a text token from any SSE data shape our backend might send:
      event=token  data={"token": "..."}           ← api/app.py standard
      event=token  data={"type":"token","content":"..."} ← alternate
      event=token  data="raw string"               ← plain text
      event=message data={"content":"..."}         ← LangChain default
      event=message data="raw string"              ← raw fallback
    Returns None if this frame carries no displayable text.
    """
    if event_type in ("error",):
        return None
    if event_type == "done":
        return None

    if isinstance(data, dict):
        # Most explicit: {"token": "..."}
        if "token" in data:
            return data["token"]
        # Alternate: {"type": "token", "content": "..."}
        if data.get("type") == "token" and "content" in data:
            return data["content"]
        # LangChain AIMessageChunk often arrives as {"content": "..."}
        if "content" in data and isinstance(data["content"], str):
            return data["content"]
        # answer key (sync-style chunk)
        if "answer" in data and isinstance(data["answer"], str):
            return data["answer"]
    elif isinstance(data, str) and data:
        return data

    return None


def _extract_sources(event_type: str, data) -> Optional[list]:
    if event_type == "done" and isinstance(data, dict):
        return data.get("sources", [])
    return None


def stream_chat(message: str, session_id: str):
    """Yields (event_type, data) from SSE /chat. Skips blank lines."""
    with requests.post(
        f"{API_BASE}/chat",
        json={"message": message, "session_id": session_id},
        stream=True,
        timeout=90,
    ) as resp:
        resp.raise_for_status()
        event_type = "message"
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if raw.startswith("event:"):
                event_type = raw[6:].strip()
            elif raw.startswith("data:"):
                payload = raw[5:].strip()
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    data = payload
                yield event_type, data
                event_type = "message"


def call_chat_sync(message: str, session_id: str) -> dict:
    r = requests.post(
        f"{API_BASE}/chat/sync",
        json={"message": message, "session_id": session_id},
        timeout=90,
    )
    r.raise_for_status()
    return r.json()


def clear_history_api(session_id: str):
    try:
        requests.delete(
            f"{API_BASE}/chat/history",
            params={"session_id": session_id},
            timeout=5,
        )
    except Exception:
        pass

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<p style='font-size:1rem;font-weight:600;margin:0.8rem 0 0;font-family:Inter,sans-serif'>"
        "RAGify AI</p>"
        "<p style='font-size:0.7rem;color:#9E9E9E;margin:0.1rem 0 0.6rem;"
        "letter-spacing:.08em;text-transform:uppercase;font-family:Inter,sans-serif'>"
        "Makani Germany</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── API Status ──
    st.markdown("<span class='sm-label'>API Status</span>", unsafe_allow_html=True)

    if st.button("↻ Status prüfen", use_container_width=True, key="btn_health"):
        st.session_state.api_ok = check_health()
        st.session_state.api_checked = True

    if not st.session_state.api_checked:
        st.session_state.api_ok = check_health()
        st.session_state.api_checked = True

    if st.session_state.api_ok:
        st.markdown("<div class='badge badge-ok'>● Verbunden</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='badge badge-err'>● Nicht erreichbar</div>", unsafe_allow_html=True)
        st.caption("Backend starten:")
        st.code("uvicorn api.app:app --reload", language="bash")

    # ── Session Info ──
    st.markdown("<span class='sm-label'>Sitzung</span>", unsafe_allow_html=True)
    st.caption(f"ID: {st.session_state.session_id[:16]}…")
    st.caption(f"Nachrichten: {len(st.session_state.messages)}")

    if st.button("✕ Gespräch löschen", use_container_width=True, key="btn_clear"):
        clear_history_api(st.session_state.session_id)
        st.session_state.messages     = []
        st.session_state.session_id   = str(uuid.uuid4())
        st.session_state.last_sources = []
        st.rerun()

    # ── Last Sources ──
    if st.session_state.last_sources:
        st.markdown("<span class='sm-label'>Quellen (letzte Antwort)</span>", unsafe_allow_html=True)
        for src in st.session_state.last_sources[:5]:
            title = (
                src.get("title") or src.get("source")
                or src.get("metadata", {}).get("title", "Dokument")
            )
            score    = src.get("score") or src.get("rerank_score")
            doc_type = src.get("type") or src.get("metadata", {}).get("type", "")
            parts    = []
            if score    is not None: parts.append(f"{score:.3f}")
            if doc_type:             parts.append(doc_type.capitalize())
            meta = " · ".join(parts)
            st.markdown(
                f"<div class='src-card'>"
                f"<div class='src-title'>{title[:55]}</div>"
                f"<div class='src-meta'>{meta}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── About ──
    st.markdown("<span class='sm-label'>Info</span>", unsafe_allow_html=True)
    st.caption("RAG mit 571 Dokumenten — Produkte, Richtlinien, Website-Inhalte von Makani Germany.")

# ─────────────────────────────────────────────
# Main header
# ─────────────────────────────────────────────
st.markdown(
    "<div style='padding:1.8rem 0 1rem;border-bottom:1px solid #EFEFEF;margin-bottom:1.4rem'>"
    "<h1 style='font-size:1.7rem;font-weight:600;margin:0;letter-spacing:-.02em;"
    "font-family:Inter,sans-serif;color:#0A0A0A'>RAGify AI</h1>"
    "<p style='font-size:0.73rem;color:#9E9E9E;margin:.3rem 0 0;"
    "letter-spacing:.08em;text-transform:uppercase;font-family:Inter,sans-serif'>"
    "Ihr Modeassistent · Makani Germany</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Suggestion chips (only when no messages yet)
# ─────────────────────────────────────────────
SUGGESTIONS = [
    "Was sind eure Versandbedingungen?",
    "Kleider unter 100 €",
    "Rückgabebedingungen?",
    "Welche Größen habt ihr?",
    "Nachhaltige Kollektionen?",
    "Outfit für den Sommer",
]

if not st.session_state.messages:
    st.markdown(
        "<p style='font-size:0.88rem;color:#616161;margin-bottom:.8rem;"
        "font-family:Inter,sans-serif'>Womit kann ich Ihnen helfen?</p>",
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    for i, suggestion in enumerate(SUGGESTIONS):
        with cols[i % 3]:
            if st.button(suggestion, key=f"sug_{i}", use_container_width=True):
                st.session_state.pending_input = suggestion
                st.rerun()

# ─────────────────────────────────────────────
# Render helpers
# ─────────────────────────────────────────────
def _source_pills(sources: list) -> str:
    if not sources:
        return ""
    pills = "".join(
        f"<span class='source-pill'>"
        f"{(src.get('title') or src.get('source') or src.get('metadata', {}).get('title', 'Quelle'))[:35]}"
        f"</span>"
        for src in sources[:4]
    )
    return f"<div class='sources-row'>{pills}</div>"


def _render_history():
    for msg in st.session_state.messages:
        role    = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])
        label   = "U" if role == "user" else "AI"
        pills   = _source_pills(sources) if role == "ai" else ""
        st.markdown(
            f"<div class='msg-row {role}'>"
            f"<div class='avatar {role}'>{label}</div>"
            f"<div class='bubble {role}'>{content}{pills}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _stream_and_render(user_message: str):
    # Show user bubble
    st.session_state.messages.append({"role": "user", "content": user_message})
    st.markdown(
        f"<div class='msg-row user'>"
        f"<div class='avatar user'>U</div>"
        f"<div class='bubble user'>{user_message}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    placeholder = st.empty()
    collected: list[str] = []
    sources:   list      = []
    got_any_token        = False

    try:
        for ev, data in stream_chat(user_message, st.session_state.session_id):
            # Try to extract a text token from whatever format arrives
            tok = _extract_token(ev, data)
            if tok is not None:
                collected.append(tok)
                got_any_token = True
                partial = "".join(collected)
                placeholder.markdown(
                    f"<div class='msg-row ai'>"
                    f"<div class='avatar ai'>AI</div>"
                    f"<div class='bubble ai'>{partial}<span class='cursor'></span></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # Check for sources in done frame
            srcs = _extract_sources(ev, data)
            if srcs is not None:
                sources = srcs

        final = "".join(collected)

        # If streaming gave us nothing, fall back to sync
        if not got_any_token:
            raise ValueError("No tokens received from stream — falling back to sync")

    except Exception as exc:
        st.warning(f"Streaming: {exc} — lade synchron …")
        try:
            res     = call_chat_sync(user_message, st.session_state.session_id)
            final   = res.get("answer", "")
            sources = res.get("sources", [])
        except Exception as exc2:
            final = f"⚠ Fehler beim Laden der Antwort: {exc2}"

    # Final render without cursor
    pills = _source_pills(sources)
    placeholder.markdown(
        f"<div class='msg-row ai'>"
        f"<div class='avatar ai'>AI</div>"
        f"<div class='bubble ai'>{final}{pills}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.session_state.messages.append({"role": "ai", "content": final, "sources": sources})
    st.session_state.last_sources = sources

# ─────────────────────────────────────────────
# Chat history — rendered in a fixed container ABOVE the input
# ─────────────────────────────────────────────
chat_area = st.container()
with chat_area:
    _render_history()

# ─────────────────────────────────────────────
# Input row — fixed at bottom, messages above
# ─────────────────────────────────────────────

# Enter-key handler: fires when user presses Enter in the text box
def _on_enter():
    val = st.session_state.chat_input.strip()
    if val:
        st.session_state.pending_input = val
        st.session_state.chat_input = ""   # clear the box

st.markdown("<hr>", unsafe_allow_html=True)

col_in, col_btn = st.columns([6, 1])
with col_in:
    st.text_input(
        label="",
        placeholder="Produktname oder Frage auf Deutsch oder Englisch …",
        key="chat_input",
        label_visibility="collapsed",
        on_change=_on_enter,
    )
with col_btn:
    send_clicked = st.button("Senden", use_container_width=True, key="btn_send")

# ─────────────────────────────────────────────
# Send logic
# ─────────────────────────────────────────────
message_to_send: Optional[str] = None

if st.session_state.pending_input:
    message_to_send = st.session_state.pending_input
    st.session_state.pending_input = ""
elif send_clicked and st.session_state.get("chat_input", "").strip():
    message_to_send = st.session_state.chat_input.strip()
    st.session_state.chat_input = ""

if message_to_send:
    if st.session_state.api_ok is False:
        st.session_state.api_ok = check_health()  # one retry

    if st.session_state.api_ok is False:
        st.error(
            "⚠ API nicht erreichbar. "
            "Starten Sie: `uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload` "
            "und klicken Sie auf **↻ Status prüfen**."
        )
    else:
        with chat_area:
            _stream_and_render(message_to_send)
        st.rerun()

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown(
    "<p style='text-align:center;font-size:0.65rem;color:#BDBDBD;"
    "letter-spacing:.1em;text-transform:uppercase;margin-top:2.5rem;"
    "font-family:Inter,sans-serif'>"
    "RAGify AI · Makani Germany · Powered by RAG · Developed and designed by Akshay Vaghasiya</p>",
    unsafe_allow_html=True,
)