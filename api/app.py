"""
api/app.py — ShopMind AI FastAPI Backend
Makani Germany RAG AI Assistant

Endpoints:
  POST /chat           — Streaming SSE chat with conversation memory
  POST /chat/sync      — Synchronous chat (non-streaming fallback)
  GET  /chat/history   — Retrieve conversation history
  DELETE /chat/history — Clear conversation history
  GET  /health         — Health check + component status
  GET  /sources        — Last retrieved sources (for citations panel)
"""

import asyncio
import json
import logging
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Path setup — make sibling packages importable when running from project root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.rag_chain import RAGChain  # noqa: E402  (after sys.path fix)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("shopmind.api")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
rag_chain: Optional[RAGChain] = None
# In-memory store: session_id → list of source dicts from the last query
_last_sources: dict[str, list] = {}


# ---------------------------------------------------------------------------
# Lifespan (replaces @app.on_event which is deprecated)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain
    logger.info("🚀 Initialising ShopMind AI RAG pipeline …")
    try:
        rag_chain = RAGChain()
        logger.info("✅ RAG pipeline ready.")
    except Exception as exc:
        logger.critical("❌ Failed to initialise RAG pipeline: %s", exc, exc_info=True)
        raise RuntimeError("RAG pipeline failed to start") from exc
    yield
    logger.info("🛑 Shutting down ShopMind AI …")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ShopMind AI — Makani Germany",
    description="RAG-powered fashion assistant with streaming SSE support.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User query in German or English")
    session_id: Optional[str] = Field(
        default=None,
        description="Conversation session ID. Omit to start a new session.",
    )


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[dict]
    latency_ms: float


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[dict]


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    components: dict[str, str]


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------
def _sse_event(data: str | dict, event: str = "message") -> str:
    """Format a single SSE frame."""
    payload = json.dumps(data) if isinstance(data, dict) else data
    return f"event: {event}\ndata: {payload}\n\n"


def _sse_error(message: str) -> str:
    return _sse_event({"error": message}, event="error")


def _sse_done(session_id: str, sources: list) -> str:
    return _sse_event(
        {"session_id": session_id, "sources": sources, "done": True},
        event="done",
    )


# ---------------------------------------------------------------------------
# Streaming generator
# ---------------------------------------------------------------------------
async def _stream_rag_response(
    message: str, session_id: str
) -> AsyncGenerator[str, None]:
    """Yield SSE frames for a streaming RAG response."""
    assert rag_chain is not None, "RAG chain not initialised"

    collected_tokens: list[str] = []
    sources: list[dict] = []
    start = asyncio.get_event_loop().time()

    try:
        # RAGChain.stream() should be a generator/async-generator that yields:
        #   {"type": "token",   "content": "<str>"}
        #   {"type": "sources", "content": [<source dicts>]}   (final frame)
        # Adjust the interface below if your RAGChain differs.
        stream = rag_chain.stream(message)  # session_id not a param — history managed internally

        def _extract(chunk) -> tuple[str | None, list | None]:
            """
            Extract (token, sources) from whatever RAGChain.stream() yields.
            This chain yields plain strings — handle that first, then fall back
            to dict/object shapes for forward compatibility.
            """
            if chunk is None:
                return None, None
            # Plain string — most common for this chain
            if isinstance(chunk, str):
                return chunk, None
            # LangChain AIMessageChunk: has .content attribute
            if hasattr(chunk, "content"):
                return chunk.content, None
            # Dict shapes
            if isinstance(chunk, dict):
                if "token" in chunk:
                    return chunk["token"], None
                if chunk.get("type") == "token":
                    return chunk.get("content", ""), None
                if chunk.get("type") == "sources":
                    return None, chunk.get("content", [])
                if "answer" in chunk:
                    return chunk["answer"], chunk.get("sources")
                if "content" in chunk:
                    return chunk["content"], None
            return None, None

        # Handle both sync generators and async generators transparently
        if hasattr(stream, "__aiter__"):
            async for chunk in stream:
                tok, srcs = _extract(chunk)
                if tok:
                    collected_tokens.append(tok)
                    yield _sse_event({"token": tok}, event="token")
                if srcs is not None:
                    sources = srcs
        else:
            # Sync generator — run in thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            chunks = await loop.run_in_executor(None, list, stream)
            for chunk in chunks:
                tok, srcs = _extract(chunk)
                if tok:
                    collected_tokens.append(tok)
                    yield _sse_event({"token": tok}, event="token")
                if srcs is not None:
                    sources = srcs

        _last_sources[session_id] = sources

        latency_ms = round((asyncio.get_event_loop().time() - start) * 1000, 1)
        logger.info(
            "session=%s tokens=%d sources=%d latency=%.0fms",
            session_id,
            len(collected_tokens),
            len(sources),
            latency_ms,
        )

        yield _sse_done(session_id, sources)

    except Exception as exc:
        logger.error("Streaming error for session %s: %s", session_id, exc, exc_info=True)
        yield _sse_error(str(exc))
        yield _sse_done(session_id, [])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness + component status check."""
    components = {
        "rag_chain": "ok" if rag_chain is not None else "unavailable",
    }
    if rag_chain is not None:
        # Optional: expose sub-component health if RAGChain exposes it
        if hasattr(rag_chain, "health"):
            components.update(rag_chain.health())

    overall = "ok" if all(v == "ok" for v in components.values()) else "degraded"
    return HealthResponse(
        status=overall,
        timestamp=datetime.utcnow().isoformat() + "Z",
        components=components,
    )


@app.post("/chat", tags=["Chat"])
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint using Server-Sent Events (SSE).

    Stream format:
      event: token   — {"token": "<str>"}          (one per token)
      event: done    — {"session_id": "...", "sources": [...], "done": true}
      event: error   — {"error": "<message>"}
    """
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not ready.")

    session_id = request.session_id or str(uuid.uuid4())
    logger.info("chat_stream  session=%s  query=%r", session_id, request.message[:80])

    return StreamingResponse(
        _stream_rag_response(request.message, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",          # disable Nginx buffering
            "X-Session-Id": session_id,
        },
    )


@app.post("/chat/sync", response_model=ChatResponse, tags=["Chat"])
async def chat_sync(request: ChatRequest):
    """
    Non-streaming fallback.  Returns the full answer + sources in one JSON response.
    Useful for testing and non-SSE clients.
    """
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not ready.")

    session_id = request.session_id or str(uuid.uuid4())
    logger.info("chat_sync  session=%s  query=%r", session_id, request.message[:80])

    start = asyncio.get_event_loop().time()
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: rag_chain.query(request.message)
        )
    except Exception as exc:
        logger.error("Sync chat error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    latency_ms = round((asyncio.get_event_loop().time() - start) * 1000, 1)

    # query() may return a dict {"answer":..., "sources":...} or a plain string
    if isinstance(result, dict):
        answer  = result.get("answer", "") or result.get("response", "") or str(result)
        sources = result.get("sources", [])
    else:
        answer  = str(result) if result else ""
        sources = []

    _last_sources[session_id] = sources

    return ChatResponse(
        answer=answer,
        session_id=session_id,
        sources=sources,
        latency_ms=latency_ms,
    )


@app.get("/chat/history", response_model=HistoryResponse, tags=["Chat"])
async def get_history(session_id: str):
    """Return conversation history for a session."""
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not ready.")
    # RAGChain manages history internally — no public get_history method
    return HistoryResponse(session_id=session_id, messages=[])


@app.delete("/chat/history", tags=["Chat"])
async def clear_history(session_id: str):
    """Clear conversation memory for a session."""
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not ready.")

    try:
        loop = asyncio.get_event_loop()
        # clear_history() takes no arguments
        await loop.run_in_executor(None, rag_chain.clear_history)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    _last_sources.pop(session_id, None)
    return {"session_id": session_id, "cleared": True}


@app.get("/sources", tags=["Chat"])
async def get_last_sources(session_id: str):
    """
    Return the retrieved sources from the most recent query in this session.
    Handy for a citations sidebar in the frontend.
    """
    return {"session_id": session_id, "sources": _last_sources.get(session_id, [])}


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )