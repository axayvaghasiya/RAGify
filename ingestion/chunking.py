"""
ingestion/chunking.py

Splits raw documents into smaller chunks ready for embedding.
Different chunk strategies per source type:
  - products       → small chunks (already compact, keep whole)
  - policies       → medium chunks (legal text needs some context)
  - website pages  → medium chunks
  - default        → medium chunks

Each chunk inherits all metadata from its parent document,
plus adds chunk-specific fields (chunk_index, chunk_total).
"""

import re
import json
import html

from pathlib import Path
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ── Paths ──────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
DOCUMENTS_PATH = PROCESSED_DIR / "documents.json"
CHUNKS_PATH    = PROCESSED_DIR / "chunks.json"


# ══════════════════════════════════════════════════════════════════════════════
# CHUNK SIZE STRATEGY
# Different content types need different chunk sizes.
#
# products   → 400 / 40 overlap
#   Products are already short structured text. Small chunks keep
#   one product per chunk — no mixing two products in one chunk.
#
# policies   → 800 / 100 overlap
#   Legal/policy text has long sentences with important context.
#   Larger chunks prevent splitting mid-clause.
#
# website    → 600 / 80 overlap
#   Page content varies. Medium size balances context vs precision.
#
# default    → 600 / 80 overlap
# ══════════════════════════════════════════════════════════════════════════════

CHUNK_CONFIG = {
    "product_catalog": {"chunk_size": 400,  "chunk_overlap": 40},
    "returns_policy":  {"chunk_size": 800,  "chunk_overlap": 100},
    "shipping_policy": {"chunk_size": 800,  "chunk_overlap": 100},
    "privacy_policy":  {"chunk_size": 800,  "chunk_overlap": 100},
    "terms_policy":    {"chunk_size": 800,  "chunk_overlap": 100},
    "website":         {"chunk_size": 600,  "chunk_overlap": 80},
    "default":         {"chunk_size": 600,  "chunk_overlap": 80},
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. TEXT CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """
    Cleans raw text before chunking.

    Steps:
      1. Decode HTML entities  (&amp; → &,  &nbsp; → space, etc.)
      2. Remove leftover HTML tags if any slipped through
      3. Normalize whitespace
      4. Strip leading/trailing whitespace
    """
    # 1. Decode HTML entities — fixes &amp; &nbsp; &quot; etc.
    text = html.unescape(text)

    # 2. Remove any leftover HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # 3. Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)   # max 2 consecutive newlines
    text = re.sub(r'[ \t]+', ' ', text)       # collapse spaces and tabs

    # 4. Strip
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# 2. LOAD SAVED DOCUMENTS
# ══════════════════════════════════════════════════════════════════════════════

def load_documents_from_json(path: Path = DOCUMENTS_PATH) -> list[Document]:
    """
    Loads documents saved by load_documents.py.
    Reads from data/processed/documents.json.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"documents.json not found at {path}\n"
            "Run: python3 ingestion/load_documents.py first."
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = [
        Document(page_content=d["page_content"], metadata=d["metadata"])
        for d in data
    ]

    print(f"📂 Loaded {len(docs)} documents from {path.name}")
    return docs


# ══════════════════════════════════════════════════════════════════════════════
# 3. CHUNK A SINGLE DOCUMENT
# ══════════════════════════════════════════════════════════════════════════════

def chunk_document(doc: Document) -> list[Document]:
    """
    Splits a single Document into chunks using the appropriate
    chunk size for its source_type.

    Special case: product_catalog documents are already compact.
    If the whole product fits within chunk_size, keep it as one chunk.

    Args:
        doc: A LangChain Document with page_content and metadata.

    Returns:
        List of chunk Documents, each with inherited + chunk metadata.
    """
    source_type = doc.metadata.get("source_type", "default")
    config      = CHUNK_CONFIG.get(source_type, CHUNK_CONFIG["default"])

    # Clean the text first
    cleaned_text = clean_text(doc.page_content)

    # Products: if content fits in one chunk, don't split
    if source_type == "product_catalog":
        if len(cleaned_text) <= config["chunk_size"]:
            chunk = Document(
                page_content=cleaned_text,
                metadata={
                    **doc.metadata,
                    "chunk_index": 0,
                    "chunk_total": 1,
                }
            )
            return [chunk]

    # All other docs: use RecursiveCharacterTextSplitter
    # Split order: paragraphs → sentences → words → characters
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        length_function=len,
    )

    raw_chunks = splitter.split_text(cleaned_text)

    # Filter out chunks that are too short to be useful
    raw_chunks = [c.strip() for c in raw_chunks if len(c.strip()) > 50]

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        chunk = Document(
            page_content=chunk_text,
            metadata={
                **doc.metadata,           # inherit all parent metadata
                "chunk_index": i,         # position within parent doc
                "chunk_total": len(raw_chunks),  # total chunks from parent
            }
        )
        chunks.append(chunk)

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# 4. CHUNK ALL DOCUMENTS
# ══════════════════════════════════════════════════════════════════════════════

def chunk_all_documents(docs: list[Document]) -> list[Document]:
    """
    Chunks all documents and returns a flat list of chunk Documents.

    Args:
        docs: List of Documents from load_documents.py

    Returns:
        Flat list of all chunks across all documents.
    """
    all_chunks = []

    # Track counts per source type for the summary
    counts: dict[str, int] = {}

    for doc in docs:
        source_type = doc.metadata.get("source_type", "default")
        chunks      = chunk_document(doc)
        all_chunks.extend(chunks)
        counts[source_type] = counts.get(source_type, 0) + len(chunks)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  ✅ Total chunks created: {len(all_chunks)}")
    for source_type, count in sorted(counts.items()):
        config = CHUNK_CONFIG.get(source_type, CHUNK_CONFIG["default"])
        print(
            f"     {source_type:<20} {count:>4} chunks  "
            f"(size={config['chunk_size']}, overlap={config['chunk_overlap']})"
        )
    print(f"{'=' * 60}\n")

    return all_chunks


# ══════════════════════════════════════════════════════════════════════════════
# 5. SAVE CHUNKS
# ══════════════════════════════════════════════════════════════════════════════

def save_chunks(chunks: list[Document], output_path: Path = CHUNKS_PATH) -> None:
    """
    Saves all chunks to data/processed/chunks.json.
    Overwrites any existing file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serialized = [
        {
            "page_content": chunk.page_content,
            "metadata":     chunk.metadata,
        }
        for chunk in chunks
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, ensure_ascii=False, indent=2)

    print(f"💾 Saved {len(chunks)} chunks → {output_path}")


def load_chunks_from_json(path: Path = CHUNKS_PATH) -> list[Document]:
    """
    Loads saved chunks from data/processed/chunks.json.
    Used by embedder.py — no need to re-chunk every run.
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
# 6. QUICK TEST — inspect chunk quality
# ══════════════════════════════════════════════════════════════════════════════

def inspect_chunks(chunks: list[Document], samples_per_type: int = 2) -> None:
    """
    Prints sample chunks per source type so you can visually
    verify chunk quality — check for clean text, good boundaries,
    no HTML entities, sensible length.
    """
    print("\n── Chunk inspection ──────────────────────────────────────")

    seen: dict[str, int] = {}

    for chunk in chunks:
        source_type = chunk.metadata.get("source_type", "unknown")

        if seen.get(source_type, 0) >= samples_per_type:
            continue

        seen[source_type] = seen.get(source_type, 0) + 1

        print(f"\n[{source_type.upper()}]  "
              f"chunk {chunk.metadata.get('chunk_index', '?')}/"
              f"{chunk.metadata.get('chunk_total', '?')}  "
              f"({len(chunk.page_content)} chars)")
        print(f"  {chunk.page_content[:300].strip()}")
        print(f"  ...")

    # Chunk length statistics
    lengths = [len(c.page_content) for c in chunks]
    print(f"\n── Chunk length stats ────────────────────────────────────")
    print(f"  Min    : {min(lengths)} chars")
    print(f"  Max    : {max(lengths)} chars")
    print(f"  Average: {sum(lengths) // len(lengths)} chars")
    print(f"  Total  : {len(chunks)} chunks")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  RAG-AI-ASSISTANT · Chunking Pipeline")
    print("=" * 60)

    # 1. Load documents saved by load_documents.py
    docs = load_documents_from_json()

    # 2. Chunk everything
    print("\n✂️  Chunking documents ...")
    chunks = chunk_all_documents(docs)

    # 3. Save to disk
    save_chunks(chunks)

    # 4. Inspect quality
    inspect_chunks(chunks)