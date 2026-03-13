"""
embeddings/embedder.py

Converts chunks into vector embeddings using OpenAI text-embedding-3-small
and stores them in a FAISS index on disk.

Flow:
  1. Load chunks from data/processed/chunks.json
  2. Batch-embed all chunk texts via OpenAI API
  3. Build FAISS index from vectors
  4. Save FAISS index + chunk metadata to data/processed/

Run once — subsequent steps load from disk, no re-embedding needed.
To re-embed (e.g. after adding new data), delete data/processed/faiss_index/
and re-run this script.
"""

import os
import json
import time
import numpy as np

from pathlib import Path
from dotenv import load_dotenv
from langchain.schema import Document
from openai import OpenAI
import faiss

# ── Load environment variables ─────────────────────────────────────────────────
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
PROCESSED_DIR  = Path(__file__).resolve().parent.parent / "data" / "processed"
CHUNKS_PATH    = PROCESSED_DIR / "chunks.json"
INDEX_DIR      = PROCESSED_DIR / "faiss_index"
INDEX_PATH     = INDEX_DIR / "index.faiss"
METADATA_PATH  = INDEX_DIR / "metadata.json"

# ── Embedding config ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM   = 1536          # dimensions for text-embedding-3-small
BATCH_SIZE      = 100           # chunks per API call — max 2048, 100 is safe
RETRY_ATTEMPTS  = 3             # retry failed batches
RETRY_DELAY     = 5             # seconds between retries


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD CHUNKS
# ══════════════════════════════════════════════════════════════════════════════

def load_chunks(path: Path = CHUNKS_PATH) -> list[Document]:
    """
    Loads chunks saved by chunking.py.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"chunks.json not found at {path}\n"
            "Run: python3 ingestion/chunking.py first."
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = [
        Document(page_content=d["page_content"], metadata=d["metadata"])
        for d in data
    ]

    print(f"📂 Loaded {len(chunks)} chunks from {path.name}")
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# 2. EMBED CHUNKS IN BATCHES
# ══════════════════════════════════════════════════════════════════════════════

def embed_chunks(
    chunks: list[Document],
    client: OpenAI,
) -> np.ndarray:
    """
    Sends chunks to OpenAI in batches and returns a numpy array of embeddings.

    Args:
        chunks: List of Document chunks to embed.
        client: OpenAI client instance.

    Returns:
        numpy array of shape (len(chunks), EMBEDDING_DIM)
    """
    texts = [chunk.page_content for chunk in chunks]
    all_embeddings = []

    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\n🔢 Embedding {len(texts)} chunks in {total_batches} batches ...")
    print(f"   Model : {EMBEDDING_MODEL}")
    print(f"   Batch : {BATCH_SIZE} chunks per request\n")

    for batch_num, start in enumerate(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[start : start + BATCH_SIZE]
        batch_idx   = batch_num + 1

        # Retry logic for transient API errors
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                response = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch_texts,
                )

                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                print(
                    f"  ✅ Batch {batch_idx}/{total_batches} "
                    f"({len(batch_texts)} chunks) "
                    f"— total embedded: {len(all_embeddings)}"
                )
                break  # success — exit retry loop

            except Exception as e:
                if attempt < RETRY_ATTEMPTS:
                    print(
                        f"  ⚠️  Batch {batch_idx} attempt {attempt} failed: {e}"
                        f" — retrying in {RETRY_DELAY}s ..."
                    )
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"  ❌ Batch {batch_idx} failed after {RETRY_ATTEMPTS} attempts: {e}")
                    raise

        # Small delay between batches — avoids rate limit on free tier
        if start + BATCH_SIZE < len(texts):
            time.sleep(0.5)

    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    print(f"\n✅ Embedding complete — shape: {embeddings_array.shape}")
    return embeddings_array


# ══════════════════════════════════════════════════════════════════════════════
# 3. BUILD FAISS INDEX
# ══════════════════════════════════════════════════════════════════════════════

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Builds a FAISS index from the embedding vectors.

    Uses IndexFlatIP (Inner Product) with normalized vectors,
    which is equivalent to cosine similarity search.

    Why cosine similarity?
    - Standard for semantic text search
    - Not affected by vector magnitude differences
    - Works well across languages (important for German text)

    Args:
        embeddings: numpy array of shape (n_chunks, EMBEDDING_DIM)

    Returns:
        FAISS index ready for similarity search.
    """
    print(f"\n🏗️  Building FAISS index ...")
    print(f"   Vectors : {embeddings.shape[0]}")
    print(f"   Dimensions: {embeddings.shape[1]}")
    print(f"   Similarity: cosine (via normalized inner product)")

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)

    # IndexFlatIP = exact search using inner product
    # For 571 chunks this is fast — no approximation needed
    # (Switch to IndexIVFFlat for 100k+ chunks)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)

    print(f"   ✅ Index built — {index.ntotal} vectors stored")
    return index


# ══════════════════════════════════════════════════════════════════════════════
# 4. SAVE INDEX + METADATA
# ══════════════════════════════════════════════════════════════════════════════

def save_index(
    index: faiss.Index,
    chunks: list[Document],
    index_dir: Path = INDEX_DIR,
) -> None:
    """
    Saves the FAISS index and chunk metadata to disk.

    Two files are saved:
      - index.faiss    : the vector index (binary)
      - metadata.json  : chunk text + metadata, indexed to match vectors

    The position of each chunk in metadata.json matches its
    vector position in the FAISS index — index position 42
    corresponds to metadata.json entry 42.
    """
    index_dir.mkdir(parents=True, exist_ok=True)

    # Save FAISS binary index
    faiss.write_index(index, str(INDEX_PATH))
    print(f"💾 Saved FAISS index → {INDEX_PATH}")

    # Save metadata — text + all metadata fields per chunk
    metadata = [
        {
            "page_content": chunk.page_content,
            "metadata":     chunk.metadata,
        }
        for chunk in chunks
    ]

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"💾 Saved metadata    → {METADATA_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. LOAD INDEX (used by retriever.py later)
# ══════════════════════════════════════════════════════════════════════════════

def load_index() -> tuple[faiss.Index, list[dict]]:
    """
    Loads the FAISS index and metadata from disk.
    Called by retriever.py — not needed during embedding.

    Returns:
        Tuple of (faiss_index, metadata_list)
    """
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_PATH}\n"
            "Run: python3 embeddings/embedder.py first."
        )

    index = faiss.read_index(str(INDEX_PATH))

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"📂 Loaded FAISS index — {index.ntotal} vectors")
    print(f"📂 Loaded metadata    — {len(metadata)} entries")

    return index, metadata


# ══════════════════════════════════════════════════════════════════════════════
# 6. VERIFY — quick sanity check search
# ══════════════════════════════════════════════════════════════════════════════

def verify_index(
    index: faiss.Index,
    metadata: list[dict],
    client: OpenAI,
) -> None:
    """
    Runs a quick test search to verify the index works correctly.
    Searches for a German fashion query and prints top 3 results.
    """
    print("\n── Index verification ────────────────────────────────────")
    test_query = "Handtasche schwarze Umhängetasche"
    print(f"  Test query: '{test_query}'")

    # Embed the query
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[test_query],
    )
    query_vector = np.array(
        [response.data[0].embedding], dtype=np.float32
    )
    faiss.normalize_L2(query_vector)

    # Search top 3
    scores, indices = index.search(query_vector, k=3)

    print(f"\n  Top 3 results:")
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        entry       = metadata[idx]
        source_type = entry["metadata"].get("source_type", "unknown")
        preview     = entry["page_content"][:120].replace("\n", " ")
        print(f"\n  [{rank}] score={score:.4f}  type={source_type}")
        print(f"       {preview} ...")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  RAG-AI-ASSISTANT · Embedder")
    print(f"  Model: {EMBEDDING_MODEL}")
    print("=" * 60)

    # Check for existing index
    if INDEX_PATH.exists():
        print(f"\n⚠️  Existing FAISS index found at {INDEX_PATH}")
        print("   Delete data/processed/faiss_index/ to re-embed.\n")
        confirm = input("   Re-embed anyway? (y/N): ").strip().lower()
        if confirm != "y":
            print("   Loading existing index ...")
            index, metadata = load_index()
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            verify_index(index, metadata, client)
            exit(0)

    # Initialise OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in .env\n"
            "Add it to your .env file and try again."
        )
    client = OpenAI(api_key=api_key)

    # 1. Load chunks
    chunks = load_chunks()

    # 2. Embed
    embeddings = embed_chunks(chunks, client)

    # 3. Build index
    index = build_faiss_index(embeddings)

    # 4. Save
    save_index(index, chunks)

    # 5. Verify
    verify_index(index, metadata=json.load(open(METADATA_PATH, encoding="utf-8")), client=client)

    print("\n" + "=" * 60)
    print("  ✅ Embedding pipeline complete")
    print(f"     {index.ntotal} vectors stored in FAISS")
    print(f"     Index saved → data/processed/faiss_index/")
    print("=" * 60)