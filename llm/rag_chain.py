"""
llm/rag_chain.py

Full RAG chain — connects HybridRetriever to Claude claude-sonnet-4-6.

Flow:
  1. Receive user query + chat history
  2. Retrieve top-5 relevant chunks (hybrid search + reranking)
  3. Build structured prompt with context
  4. Call Claude claude-sonnet-4-6 via Anthropic API
  5. Stream response tokens back to caller
  6. Log query metadata to SQLite for analytics

Usage:
    chain = RAGChain()

    # Non-streaming
    result = chain.query("Was kostet die HALIA Tasche?")
    print(result["answer"])

    # Streaming
    for token in chain.stream("Was kostet die HALIA Tasche?"):
        print(token, end="", flush=True)
"""

import os
import json
import time
import sqlite3

from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain.schema import Document
from anthropic import Anthropic #For Claude Sonnet-4-6
# from openai import OpenAI (For Open AI - GPT-4o-mini)

from retrieval.retriever import HybridRetriever
from llm.prompt_templates import RAG_SYSTEM_PROMPT, build_rag_prompt

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH      = PROJECT_ROOT / "data" / "processed" / "query_logs.db"

# ── LLM config ─────────────────────────────────────────────────────────────────
CLAUDE_MODEL  = "claude-sonnet-4-6"
# OPENAI_MODEL  = "gpt-4o-mini"
MAX_TOKENS    = 1024
TEMPERATURE   = 0.3   # low = more factual, less creative — good for RAG


# ══════════════════════════════════════════════════════════════════════════════
# 1. QUERY LOGGER
# ══════════════════════════════════════════════════════════════════════════════

def init_db(db_path: Path = DB_PATH) -> None:
    """
    Creates the query_logs SQLite table if it doesn't exist.
    Called once on RAGChain initialisation.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT    NOT NULL,
            query         TEXT    NOT NULL,
            answer        TEXT,
            sources_used  TEXT,    -- JSON array of source_types
            latency_ms    INTEGER,
            input_tokens  INTEGER,
            output_tokens INTEGER,
            model         TEXT,
            error         TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_query(
    query:         str,
    answer:        str,
    sources_used:  list[str],
    latency_ms:    int,
    input_tokens:  int,
    output_tokens: int,
    error:         str | None = None,
    db_path:       Path = DB_PATH,
) -> None:
    """
    Logs a query and its response to SQLite.
    Non-blocking — failures are silently ignored so they
    never interrupt the user-facing response.
    """
    try:
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            INSERT INTO query_logs
                (timestamp, query, answer, sources_used,
                 latency_ms, input_tokens, output_tokens, model, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(),
                query,
                answer,
                json.dumps(sources_used, ensure_ascii=False),
                latency_ms,
                input_tokens,
                output_tokens,
                CLAUDE_MODEL,
                # OPENAI_MODEL,
                error,
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass  # Never let logging break the main flow


# ══════════════════════════════════════════════════════════════════════════════
# 2. RAG CHAIN
# ══════════════════════════════════════════════════════════════════════════════

class RAGChain:
    """
    Full RAG pipeline: retrieve → prompt → generate → log.

    Attributes:
        retriever:  HybridRetriever instance (FAISS + BM25 + reranker)
        client:     Anthropic API client
        history:    Conversation history list (last N turns)
    """

    def __init__(self):
        print("🔧 Initialising RAGChain ...")

        # Initialise retriever (loads FAISS, BM25, cross-encoder)
        self.retriever = HybridRetriever()

        # Initialise Anthropic client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        # api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found in .env\n"
                "Add it and try again."
            )
        self.client = Anthropic(api_key=api_key)
        # self.client = OpenAI(api_key=api_key)

        # Conversation history — stores last 4 turns (8 messages)
        self.history: list[dict] = []

        # Initialise query log database
        init_db()

        print(f"✅ RAGChain ready — model: {CLAUDE_MODEL}\n")
        # print(f"✅ RAGChain ready — model: {OPENAI_MODEL}\n")

    def _detect_source_filter(self, query: str) -> str | None:
        """Route query to relevant source type based on keywords.
        Returns a source_type string to filter retrieval, or None to search all."""
        q = query.lower()

        if any(w in q for w in [
            "retoure", "zurück", "widerruf", "rücksend", "rückgabe",
            "return", "refund", "withdraw", "rückerstatt"
        ]):
            return "returns_policy"

        if any(w in q for w in [
            "versand", "lieferung", "lieferzeit", "versandkosten", "zoll",
            "shipping", "delivery", "deliver", "ship", "postage"
        ]):
            return "shipping_policy"

        if any(w in q for w in [
            "tasche", "handtasche", "geldbeutel", "produkt", "preis", "farbe", "modell",
            "bag", "wallet", "price", "product", "colour", "color", "model"
        ]):
            return "product_catalog"

        if any(w in q for w in [
            "datenschutz", "daten", "privacy", "data", "verantwortlich"
        ]):
            return "privacy_policy"

        return None
            
    def _get_sources(self, docs: list[tuple[Document, float]]) -> list[str]:
        """Extract unique source types from retrieved docs."""
        seen = set()
        sources = []
        for doc, _ in docs:
            st = doc.metadata.get("source_type", "unknown")
            if st not in seen:
                seen.add(st)
                sources.append(st)
        return sources


    # ── Non-streaming query ────────────────────────────────────────────────────

    def query(
        self,
        user_query:    str,
        source_filter: str | None = None,
        top_k:         int = 5,
    ) -> dict:
        """
        Run a single RAG query and return the full response.

        Args:
            user_query:    User's question string.
            source_filter: Optional filter by source_type.
            top_k:         Number of chunks to retrieve.

        Returns:
            Dict with keys:
              answer        — Claude's response string
              sources       — list of source_types used
              retrieved     — list of (Document, score) tuples
              latency_ms    — total response time
              input_tokens  — tokens in prompt
              output_tokens — tokens in response
        """
        start_time = time.time()
        error      = None
        answer     = ""

        try:
            # 1. Retrieve relevant chunks
            source_filter = self._detect_source_filter(user_query)
            if source_filter:
                print(f"  🎯 Routing to: {source_filter}")
            retrieved_docs = self.retriever.retrieve(
                query=user_query,
                top_k=top_k,
                source_filter=source_filter,
            )
            
            # retrieved_docs = self.retriever.retrieve(
            #     query=user_query,
            #     top_k=top_k,
            #     source_filter=source_filter,
            # )

            # 2. Build prompt
            messages = build_rag_prompt(
                query=user_query,
                retrieved_docs=retrieved_docs,
                chat_history=self.history,
            )

            # 3. Call Claude/OpenAI
            response = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                system=RAG_SYSTEM_PROMPT,
                messages=messages,
            )

            answer        = response.content[0].text
            input_tokens  = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            # response = self.client.chat.completions.create(
            #     model=OPENAI_MODEL,
            #     max_tokens=MAX_TOKENS,
            #     temperature=TEMPERATURE,
            #     messages=[{"role": "system", "content": RAG_SYSTEM_PROMPT}] + messages,
            # )
            # answer        = response.choices[0].message.content
            # input_tokens  = response.usage.prompt_tokens
            # output_tokens = response.usage.completion_tokens

            # 4. Update conversation history
            self.history.append({"role": "user",      "content": user_query})
            self.history.append({"role": "assistant",  "content": answer})

            # Keep only last 8 messages (4 turns)
            if len(self.history) > 8:
                self.history = self.history[-8:]

        except Exception as e:
            error         = str(e)
            answer        = f"Es tut mir leid, es ist ein Fehler aufgetreten: {e}"
            input_tokens  = 0
            output_tokens = 0
            retrieved_docs = []

        latency_ms   = int((time.time() - start_time) * 1000)
        sources_used = self._get_sources(retrieved_docs) if retrieved_docs else []

        # 5. Log to SQLite
        log_query(
            query=user_query,
            answer=answer,
            sources_used=sources_used,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            error=error,
        )

        return {
            "answer":        answer,
            "sources":       sources_used,
            "retrieved":     retrieved_docs,
            "latency_ms":    latency_ms,
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
        }


    # ── Streaming query ────────────────────────────────────────────────────────

    def stream(
        self,
        user_query:    str,
        source_filter: str | None = None,
        top_k:         int = 5,
    ):
        """
        Stream a RAG response token by token.

        Yields string tokens as they arrive from Claude.
        Logs the complete query after streaming finishes.

        Usage:
            for token in chain.stream("Was kostet die HALIA?"):
                print(token, end="", flush=True)
        """
        start_time = time.time()
        full_answer = ""
        error       = None

        try:
            # 1. Retrieve
            source_filter = self._detect_source_filter(user_query)
            if source_filter:
                print(f"  🎯 Routing to: {source_filter}")
            retrieved_docs = self.retriever.retrieve(
                query=user_query,
                top_k=top_k,
                source_filter=source_filter,
            )
            # retrieved_docs = self.retriever.retrieve(
            #     query=user_query,
            #     top_k=top_k,
            #     source_filter=source_filter,
            # )

            # 2. Build prompt
            messages = build_rag_prompt(
                query=user_query,
                retrieved_docs=retrieved_docs,
                chat_history=self.history,
            )

            # 3. Stream from Claude
            input_tokens  = 0
            output_tokens = 0

            with self.client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                system=RAG_SYSTEM_PROMPT,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    full_answer += text
                    yield text

                # Get token usage after stream completes
                final_message = stream.get_final_message()
                input_tokens  = final_message.usage.input_tokens
                output_tokens = final_message.usage.output_tokens

            # stream = self.client.chat.completions.create(
            #     model=OPENAI_MODEL,
            #     max_tokens=MAX_TOKENS,
            #     temperature=TEMPERATURE,
            #     messages=[{"role": "system", "content": RAG_SYSTEM_PROMPT}] + messages,
            #     stream=True,
            # )
            # for chunk in stream:
            #     text = chunk.choices[0].delta.content or ""
            #     if text:
            #         full_answer += text
            #         yield text
            
            # 4. Update history
            self.history.append({"role": "user",      "content": user_query})
            self.history.append({"role": "assistant",  "content": full_answer})
            if len(self.history) > 8:
                self.history = self.history[-8:]

        except Exception as e:
            error       = str(e)
            full_answer = f"Fehler: {e}"
            yield full_answer
            retrieved_docs = []
            input_tokens   = 0
            output_tokens  = 0

        latency_ms   = int((time.time() - start_time) * 1000)
        sources_used = self._get_sources(retrieved_docs) if retrieved_docs else []

        # 5. Log after stream completes
        log_query(
            query=user_query,
            answer=full_answer,
            sources_used=sources_used,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            error=error,
        )

    def clear_history(self) -> None:
        """Reset conversation history — start a new session."""
        self.history = []
        print("🗑️  Conversation history cleared.")


# ══════════════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  RAG-AI-ASSISTANT · RAG Chain Test")
    print("=" * 60 + "\n")

    chain = RAGChain()

    # ── Test 1: Product query ──────────────────────────────────────────────────
    print("── Test 1: Product query ─────────────────────────────────")
    print("Query: Welche schwarzen Handtaschen habt ihr unter 120 Euro?\n")
    print("Answer (streaming):\n")

    for token in chain.stream("Welche schwarzen Handtaschen habt ihr unter 120 Euro?"):
        print(token, end="", flush=True)

    print("\n")

    # ── Test 2: Policy query ───────────────────────────────────────────────────
    print("── Test 2: Shipping query ────────────────────────────────")
    print("Query: Wie lange dauert der Versand nach Österreich?\n")
    print("Answer (streaming):\n")

    for token in chain.stream("Wie lange dauert der Versand nach Österreich?"):
        print(token, end="", flush=True)

    print("\n")

    # ── Test 3: Follow-up (tests conversation memory) ─────────────────────────
    print("── Test 3: Follow-up question (conversation memory) ──────")
    print("Query: Und was kostet der Versand dahin?\n")
    print("Answer (streaming):\n")

    for token in chain.stream("Und was kostet der Versand dahin?"):
        print(token, end="", flush=True)

    print("\n")

    # ── Test 4: English query ──────────────────────────────────────────────────
    print("── Test 4: English query ─────────────────────────────────")
    print("Query: What is your return policy?\n")
    print("Answer (streaming):\n")

    for token in chain.stream("What is your return policy?"):
        print(token, end="", flush=True)

    print("\n")

    # ── Show log summary ───────────────────────────────────────────────────────
    print("── Query log summary ─────────────────────────────────────")
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT query, latency_ms, input_tokens, output_tokens "
        "FROM query_logs ORDER BY id DESC LIMIT 4"
    ).fetchall()
    conn.close()

    for row in rows:
        query, latency, inp, out = row
        print(f"  [{latency}ms | {inp}↑ {out}↓ tokens] {query[:50]}")

    print("\n" + "=" * 60)
    print("  ✅ RAG Chain test complete")
    print("=" * 60)