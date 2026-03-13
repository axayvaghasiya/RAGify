"""
retrieval/retriever.py

Advanced RAG retrieval pipeline — the core ML layer of the system.

Three-stage retrieval:
  Stage 1 — Parallel search
    a) Dense search  : FAISS cosine similarity (semantic meaning)
    b) Sparse search : BM25 keyword matching (exact terms)

  Stage 2 — RRF Fusion
    Combine FAISS + BM25 ranked lists into a single ranked list
    using Reciprocal Rank Fusion (RRF).

  Stage 3 — Cross-Encoder Reranking
    Re-score top-20 fused results with a cross-encoder model
    that reads (query, chunk) pairs — much more accurate than
    bi-encoder similarity alone.

Why this matters:
  - FAISS alone misses exact product names ("MC NANI")
  - BM25 alone misses semantic intent ("günstige Tasche")
  - Together they catch both — RRF merges them fairly
  - Cross-encoder reranking pushes the truly relevant chunks to top-5
"""

import json
import os
import numpy as np

from pathlib import Path
from dotenv import load_dotenv
from langchain.schema import Document
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import faiss

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
INDEX_PATH    = PROCESSED_DIR / "faiss_index" / "index.faiss"
METADATA_PATH = PROCESSED_DIR / "faiss_index" / "metadata.json"

# ── Model config ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL   = "text-embedding-3-small"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Retrieval config ───────────────────────────────────────────────────────────
DENSE_TOP_K    = 20   # candidates from FAISS
SPARSE_TOP_K   = 20   # candidates from BM25
RRF_K          = 60   # RRF constant — 60 is standard from the paper
RERANK_TOP_N   = 20   # how many to send to cross-encoder
FINAL_TOP_K    = 5    # final chunks returned to RAG chain


# ══════════════════════════════════════════════════════════════════════════════
# 1. RETRIEVER CLASS
# ══════════════════════════════════════════════════════════════════════════════

class HybridRetriever:
    """
    Hybrid retriever combining dense (FAISS) + sparse (BM25) search
    with RRF fusion and cross-encoder reranking.

    Usage:
        retriever = HybridRetriever()
        results = retriever.retrieve("Welche Handtaschen habt ihr in schwarz?")
        for doc, score in results:
            print(score, doc.page_content[:100])
    """

    def __init__(self):
        print("🔧 Initialising HybridRetriever ...")

        # Load FAISS index
        self.index, self.metadata = self._load_faiss_index()

        # Build BM25 index from same chunks
        self.bm25, self.bm25_chunks = self._build_bm25_index()

        # Load cross-encoder
        self.cross_encoder = self._load_cross_encoder()

        # OpenAI client for query embedding
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        print("✅ HybridRetriever ready\n")


    # ── Loaders ────────────────────────────────────────────────────────────────

    def _load_faiss_index(self) -> tuple[faiss.Index, list[dict]]:
        """Load FAISS index and metadata from disk."""
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {INDEX_PATH}\n"
                "Run: python3 embeddings/embedder.py first."
            )

        index = faiss.read_index(str(INDEX_PATH))

        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        print(f"  📂 FAISS index loaded — {index.ntotal} vectors")
        return index, metadata


    def _build_bm25_index(self) -> tuple[BM25Okapi, list[dict]]:
        """
        Build BM25 sparse index from chunk texts.

        Tokenization: lowercase split on whitespace.
        Simple but effective for German product/policy text.
        BM25 will match exact tokens like product names, prices,
        policy terms that semantic search can miss.
        """
        texts = [entry["page_content"] for entry in self.metadata]

        # Tokenize: lowercase, split on whitespace
        tokenized = [text.lower().split() for text in texts]

        bm25 = BM25Okapi(tokenized)
        print(f"  📂 BM25 index built  — {len(texts)} documents")
        return bm25, self.metadata


    def _load_cross_encoder(self) -> CrossEncoder:
        """
        Load cross-encoder model for reranking.

        ms-marco-MiniLM-L-6-v2 is trained on passage ranking.
        Small (22M params), fast on CPU, strong reranking quality.
        Downloads ~80MB on first run, cached locally after.
        """
        print(f"  🤖 Loading cross-encoder: {CROSS_ENCODER_MODEL}")
        model = CrossEncoder(CROSS_ENCODER_MODEL)
        print(f"  ✅ Cross-encoder loaded")
        return model


    # ── Query embedding ────────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string using OpenAI.
        Returns normalized vector ready for FAISS inner product search.
        """
        response = self.openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query],
        )
        vector = np.array([response.data[0].embedding], dtype=np.float32)
        faiss.normalize_L2(vector)
        return vector


    # ── Stage 1a: Dense search ─────────────────────────────────────────────────

    def _dense_search(self, query: str, top_k: int = DENSE_TOP_K) -> list[tuple[int, float]]:
        """
        FAISS cosine similarity search.

        Returns:
            List of (index_position, score) tuples, highest score first.
        """
        query_vector = self._embed_query(query)
        scores, indices = self.index.search(query_vector, k=top_k)

        results = [
            (int(idx), float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx != -1  # FAISS returns -1 for empty slots
        ]
        return results


    # ── Stage 1b: Sparse search ────────────────────────────────────────────────

    def _sparse_search(self, query: str, top_k: int = SPARSE_TOP_K) -> list[tuple[int, float]]:
        """
        BM25 keyword search.

        Returns:
            List of (index_position, score) tuples, highest score first.
        """
        tokenized_query = query.lower().split()
        scores          = self.bm25.get_scores(tokenized_query)

        # Get top_k indices sorted by score descending
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [
            (int(idx), float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0  # skip zero-score results
        ]
        return results


    # ── Stage 2: RRF Fusion ────────────────────────────────────────────────────

    def _rrf_fusion(
        self,
        dense_results:  list[tuple[int, float]],
        sparse_results: list[tuple[int, float]],
        k: int = RRF_K,
    ) -> list[tuple[int, float]]:
        """
        Reciprocal Rank Fusion — combines two ranked lists into one.

        Formula: RRF(d) = Σ 1 / (k + rank(d))
          where rank is 1-indexed position in each list.

        k=60 is the value from the original RRF paper (Cormack 2009).
        Higher k reduces the impact of top-ranked documents.

        Args:
            dense_results:  [(index, score), ...] from FAISS
            sparse_results: [(index, score), ...] from BM25
            k:              RRF constant

        Returns:
            Fused list of (index, rrf_score) sorted by rrf_score desc.
        """
        rrf_scores: dict[int, float] = {}

        # Add RRF scores from dense results
        for rank, (idx, _) in enumerate(dense_results, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank)

        # Add RRF scores from sparse results
        for rank, (idx, _) in enumerate(sparse_results, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank)

        # Sort by fused score descending
        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return fused


    # ── Stage 3: Cross-encoder reranking ──────────────────────────────────────

    def _rerank(
        self,
        query:      str,
        candidates: list[tuple[int, float]],
        top_n:      int = RERANK_TOP_N,
    ) -> list[tuple[int, float]]:
        """
        Cross-encoder reranking of top candidates.

        The cross-encoder reads the full (query, passage) pair together,
        giving much more accurate relevance scores than bi-encoder
        similarity. Slower but only runs on top_n candidates.

        Args:
            query:      Original user query string.
            candidates: [(index, rrf_score), ...] from RRF fusion.
            top_n:      How many candidates to rerank.

        Returns:
            Reranked list of (index, cross_encoder_score), top_n items.
        """
        # Take top_n candidates for reranking
        top_candidates = candidates[:top_n]

        # Build (query, passage) pairs
        pairs = [
            (query, self.metadata[idx]["page_content"])
            for idx, _ in top_candidates
        ]

        # Score all pairs — cross-encoder returns raw logits
        scores = self.cross_encoder.predict(pairs)

        # Combine indices with new scores and sort
        reranked = sorted(
            [(idx, float(score)) for (idx, _), score in zip(top_candidates, scores)],
            key=lambda x: x[1],
            reverse=True,
        )

        return reranked


    # ── Main retrieve method ───────────────────────────────────────────────────

    def retrieve(
        self,
        query:        str,
        top_k:        int  = FINAL_TOP_K,
        source_filter: str | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Full hybrid retrieval pipeline for a single query.

        Pipeline:
          1. Dense search  (FAISS)
          2. Sparse search (BM25)
          3. RRF fusion
          4. Cross-encoder reranking
          5. Return top_k Documents with scores

        Args:
            query:         User's question string.
            top_k:         Number of chunks to return (default 5).
            source_filter: Optional — filter by source_type before search.
                           e.g. "product_catalog", "returns_policy"

        Returns:
            List of (Document, score) tuples, best first.
        """
        # Stage 1: Parallel search
        dense_results  = self._dense_search(query)
        sparse_results = self._sparse_search(query)

        # Optional source_type filter — apply after retrieval
        if source_filter:
            dense_results  = [
                (idx, score) for idx, score in dense_results
                if self.metadata[idx]["metadata"].get("source_type") == source_filter
            ]
            sparse_results = [
                (idx, score) for idx, score in sparse_results
                if self.metadata[idx]["metadata"].get("source_type") == source_filter
            ]

        # Stage 2: RRF fusion
        fused = self._rrf_fusion(dense_results, sparse_results)

        # Stage 3: Cross-encoder reranking
        reranked = self._rerank(query, fused)

        # Build Document objects for top_k results
        results = []
        for idx, score in reranked[:top_k]:
            entry = self.metadata[idx]
            doc   = Document(
                page_content=entry["page_content"],
                metadata=entry["metadata"],
            )
            results.append((doc, score))

        return results


# ══════════════════════════════════════════════════════════════════════════════
# TEST — run directly to verify retrieval quality
# ══════════════════════════════════════════════════════════════════════════════

def print_results(query: str, results: list[tuple[Document, float]]) -> None:
    """Pretty-print retrieval results for a test query."""
    print(f"\n{'─' * 60}")
    print(f"Query: \"{query}\"")
    print(f"{'─' * 60}")
    for rank, (doc, score) in enumerate(results, 1):
        source_type = doc.metadata.get("source_type", "unknown")
        source      = doc.metadata.get("source", "")
        print(f"\n[{rank}] score={score:.4f}  type={source_type}")
        print(f"     {doc.page_content[:200].strip()} ...")


if __name__ == "__main__":
    print("=" * 60)
    print("  RAG-AI-ASSISTANT · Hybrid Retriever Test")
    print("=" * 60 + "\n")

    # Initialise retriever (loads all indexes + models)
    retriever = HybridRetriever()

    # ── Test queries — one per source type ────────────────────────────────────

    # 1. Product search (German)
    results = retriever.retrieve("Schwarze Handtasche unter 100 Euro")
    print_results("Schwarze Handtasche unter 100 Euro", results)

    # 2. Policy question (German)
    results = retriever.retrieve("Wie lange dauert die Lieferung nach Deutschland?")
    print_results("Wie lange dauert die Lieferung nach Deutschland?", results)

    # 3. Returns question (German)
    results = retriever.retrieve("Kann ich einen Artikel zurückgeben?")
    print_results("Kann ich einen Artikel zurückgeben?", results)

    # 4. Cross-lingual — English query, German index
    results = retriever.retrieve("What is the return policy?")
    print_results("What is the return policy? (English query → German index)", results)

    # 5. Source filter test — only search products
    results = retriever.retrieve(
        "Umhängetasche Leder",
        source_filter="product_catalog"
    )
    print_results("Umhängetasche Leder [filtered: product_catalog only]", results)

    print("\n" + "=" * 60)
    print("  ✅ Retriever test complete")
    print("=" * 60)