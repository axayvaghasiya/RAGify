# RAGify – Enterprise Retrieval-Augmented Generation (RAG) System (For E-Commerce)

🎥 **Demo Video:** https://www.loom.com/share/96d9f497f55941ddacec2d4d6de873f0

> Production-grade **Retrieval-Augmented Generation (RAG)** system built over a real German Shopify merchant dataset (Makani Germany).  
> The system combines hybrid retrieval, cross-encoder reranking, and LLM reasoning to answer customer questions in **German and English**.

[![Python](https://img.shields.io/badge/Python-3.14-blue)](https://python.org)
[![Claude](https://img.shields.io/badge/LLM-Claude%20Sonnet%204-orange)](https://anthropic.com)
[![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-green)](https://github.com/facebookresearch/faiss)
[![RAGAS](https://img.shields.io/badge/Eval-RAGAS-purple)](https://github.com/explodinggradients/ragas)

---

## Key Features

• **Hybrid retrieval:** FAISS + BM25 + Reciprocal Rank Fusion  
• **Cross-encoder reranking:** `ms-marco-MiniLM-L-6-v2`  
• **Query routing:** source-aware retrieval (product / shipping / returns)  
• **Streaming responses:** SSE streaming via FastAPI  
• **Multilingual retrieval:** German ↔ English cross-lingual search  
• **Evaluation pipeline:** RAGAS metrics (Faithfulness, Relevancy, Precision)  
• **Source attribution:** showing the document used to generate answers

<!-- ## Live Demo

🚀 **[Try RAGify AI](https://project-ragify.streamlit.app)**

Ask questions like:
- *"Welche schwarzen Handtaschen habt ihr unter 120 Euro?"*
- *"What is your return policy?"*
- *"Wie lange dauert der Versand nach Österreich?"*

--- -->

## Demo Video:
**[Loom](https://www.loom.com/share/96d9f497f55941ddacec2d4d6de873f0)

## RAGAS Evaluation Results

Evaluated on 20 golden Q&A pairs (10 German + 10 English) using the RAGAS framework.

| Metric | Without Routing | With Query Routing | Improvement |
|--------|----------------|-------------------|-------------|
| **Faithfulness** | 0.90 | 0.88 | ≈ same |
| **Answer Relevancy** | 0.60 | **0.76** | +26%    |
| **Context Precision** | 0.50 | **0.68** | +36%   |

> **Faithfulness 0.88** — the LLM stays grounded in retrieved context with minimal hallucination.  
> Adding keyword-based query routing improved answer relevancy by 26% and context precision by 36% by directing queries to the correct source type before retrieval.

---

## Architecture

```
         User Query
             │
             ▼
┌─────────────────────────────────┐
│      Query Router               │  keyword-based intent detection
│  → product_catalog              │  routes to correct source type
│  → shipping_policy              │  before search
│  → returns_policy               │
└────────────┬────────────────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
┌────────┐      ┌──────────┐
│ FAISS  │      │  BM25    │   Stage 1: Parallel Search
│ Dense  │      │ Sparse   │   FAISS = semantic similarity
│ Search │      │ Search   │   BM25  = exact keyword match
└────┬───┘      └─────┬────┘
     │                │
     └──────┬─────────┘
            ▼
    ┌───────────────┐
    │  RRF Fusion   │   Stage 2: Reciprocal Rank Fusion
    │   k = 60      │   merges ranked lists fairly
    └───────┬───────┘
            ▼
    ┌───────────────┐
    │ Cross-Encoder │   Stage 3: Reranking
    │  Reranking    │   ms-marco-MiniLM-L-6-v2
    │  top-20 → 5   │   reads (query, chunk) pairs
    └───────┬───────┘
            ▼
    ┌───────────────┐
    │ Claude Sonnet │   Generation
    │     4.6       │   streaming + conversation memory
    │  + citations  │   answers in DE or EN
    └───────────────┘
```

---

## Data Sources

| Source | Documents | Chunks | Language |
|--------|-----------|--------|----------|
| Product catalog (CSV) | 455 | 461 | German |
| Website pages (scraped) | 6 | 46 | German |
| Shipping policy | 1 | 5 | German |
| Returns policy | 1 | 3 | German |
| Privacy policy | 1 | 35 | German |
| Terms & conditions | 1 | 21 | German |
| **Total** | **465** | **571** | **DE + EN** |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Embeddings | OpenAI `text-embedding-3-small` (1536 dim) |
| Vector store | FAISS `IndexFlatIP` (cosine similarity) |
| Sparse search | BM25 (`rank-bm25`) |
| Fusion | Reciprocal Rank Fusion (k=60) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Claude claude-sonnet-4-6 (Anthropic) |
| Framework | LangChain LCEL |
| API | FastAPI + SSE streaming |
| UI | Streamlit |
| Evaluation | RAGAS |
| Logging | SQLite |

---

## Project Structure

```
rag-ai-assistant/
├── ingestion/
│   ├── load_documents.py   # scraper + policy loader + product CSV
│   └── chunking.py         # source-aware chunking (400-800 chars)
├── embeddings/
│   └── embedder.py         # batch embedding + FAISS index builder
├── retrieval/
│   └── retriever.py        # HybridRetriever: FAISS + BM25 + RRF + reranking
├── llm/
│   ├── prompt_templates.py # German RAG system prompt with XML structure
│   └── rag_chain.py        # streaming chain + query routing + memory + logging
├── api/
│   └── app.py              # FastAPI with SSE streaming endpoint
├── frontend/
│   └── streamlit_app.py    # chat UI with citations + session management
├── evaluation/
│   ├── golden_dataset.json # 20 Q&A pairs (10 DE + 10 EN)
│   ├── evaluate_rag.py     # RAGAS evaluation pipeline
│   └── report.md           # generated evaluation report
└── data/
    ├── raw/                # gitignored — merchant files
    └── processed/          # gitignored — chunks, FAISS index, logs
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key (embeddings — `text-embedding-3-small`)
- Anthropic API key (generation — Claude claude-sonnet-4-6)

### Setup

```bash
# Clone
git clone https://github.com/axayvaghasiya/rag-ai-assistant.git
cd rag-ai-assistant

# Virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Environment variables
cp .env.example .env
# Add your OPENAI_API_KEY and ANTHROPIC_API_KEY to .env
```

### Run the pipeline

```bash
# Step 1 — Load documents (policies + website + products)
python3 ingestion/load_documents.py

# Step 2 — Chunk documents
python3 ingestion/chunking.py

# Step 3 — Embed and build FAISS index (~$0.002)
python3 embeddings/embedder.py

# Step 4 — Test retrieval
PYTHONPATH=. python3 retrieval/retriever.py

# Step 5 — Test RAG chain
PYTHONPATH=. python3 llm/rag_chain.py
```

### Run the app

```bash
# Terminal 1 — Start FastAPI backend
PYTHONPATH=. uvicorn api.app:app --reload --port 8000

# Terminal 2 — Start Streamlit frontend
PYTHONPATH=. streamlit run frontend/streamlit_app.py
```

Open **http://localhost:8501**

### Run evaluation

```bash
PYTHONPATH=. python3 evaluation/evaluate_rag.py
# Results saved to evaluation/results.json and evaluation/report.md
```

---

## Key Design Decisions

**Why hybrid search?**  
FAISS alone misses exact product names ("MC NANI"). BM25 alone misses semantic intent ("affordable bag"). Hybrid search catches both. RRF fusion (k=60, Cormack 2009) merges ranked lists without requiring score normalisation.

**Why cross-encoder reranking?**  
Bi-encoder similarity (FAISS) embeds query and chunks independently — fast but approximate. The cross-encoder reads the full `(query, chunk)` pair together, giving far more accurate relevance scores. Running it on only top-20 candidates keeps latency acceptable.

**Why query routing?**  
Searching all 571 chunks for a returns question surfaces irrelevant product chunks. Keyword-based routing directs the query to the relevant source type first, improving context precision by 36%.

**Why Claude claude-sonnet-4-6?**  
200k context window handles long policy documents. Strong multilingual performance on German text. Streaming API enables token-by-token response delivery.

**Why German data kept as-is?**  
`text-embedding-3-small` is multilingual — it handles cross-lingual retrieval natively. An English question ("What is your return policy?") correctly retrieves German policy chunks. This is verified in the evaluation dataset.

---

## Sample Interactions

**Product search (German)**
```
User: Welche schwarzen Handtaschen habt ihr unter 120 Euro?

RAGify: Unter 120 Euro haben wir:
  • HALIA - BLACK — 99,90 €
  • OLINA - BLACK — 119,90 €
[Quelle: Produkt]
```

**Policy question (English)**
```
User: What is your return policy?

RAGify: You have 14 days to withdraw from the contract.
We will refund all payments within 14 days using the same
payment method. Return shipping costs are borne by the customer.
[Source: Retoure, AGB]
```

**Conversation memory**
```
User: Wie lange dauert der Versand nach Österreich?
RAGify: 3–4 Werktage. [Quelle: Versandrichtlinie]

User: Und was kostet der Versand dahin?
RAGify: Der Versand nach Österreich ist kostenlos. [Quelle: Website]
```

---

## Cost

| Operation | Cost |
|-----------|------|
| Embed 571 chunks | ~$0.002 |
| Per RAG query (Claude) | ~$0.003 |
| Full RAGAS evaluation (20 Q) | ~$0.05 |
| **Total project cost** | **< $1** |

---

## Roadmap

- [ ] Pinecone migration (persistent vector store)
- [ ] HyDE query expansion (hypothetical document embeddings)
- [ ] Multi-query retrieval (3 rephrasings per query)
- [ ] Analytics dashboard (query volume, latency, token cost)
- [ ] Docker Compose deployment
- [ ] Automated index refresh when merchant data changes

---

## About

Built by **Akshay Vaghasiya** — Senior Shopify Engineer transitioning to AI Engineering.

@ 2026 Akshay Vaghasiya. All rights reserved.

This project demonstrates end-to-end RAG system design with production-grade retrieval, quantitative evaluation, and real merchant data — not a toy dataset.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/akshayvaghasiya)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/axayvaghasiya)
