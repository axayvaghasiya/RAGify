# RAGify AI — RAGAS Evaluation Report

**Generated:** 2026-03-13 17:12 UTC  
**Dataset:** 20 questions (10 German + 10 English)  
**RAG Model:** claude-sonnet-4-6  
**Embedding Model:** text-embedding-3-small  
**Retrieval:** Hybrid BM25 + FAISS + Cross-Encoder Reranking  

---

## Aggregate Scores

| Metric | Score |
|--------|-------|
| **Faithfulness** | 0.8833 |
| **Answer Relevancy** | 0.7604 |
| **Context Precision** | 0.6767 |

> **Faithfulness** measures whether the answer is grounded in the retrieved context (no hallucination).  
> **Answer Relevancy** measures whether the answer addresses the question asked.  
> **Context Precision** measures whether the retrieved chunks are relevant to the question.

---

## Performance by Language

| Language | Questions | Avg Latency |
|----------|-----------|-------------|
| German (DE) | 10 | 4260ms |
| English (EN) | 10 | 4268ms |
| **Overall** | 20 | 4264ms |

---

## Performance by Category

| Category | Questions | Avg Latency |
|----------|-----------|-------------|
| Payment | 1 | 4195ms |
| Privacy | 1 | 2718ms |
| Product | 4 | 5046ms |
| Returns | 5 | 4873ms |
| Shipping | 9 | 3757ms |

---

## Per-Question Results

| # | Language | Category | Question | Latency | Sources |
|---|----------|----------|----------|---------|---------|
| 1 | DE | product | Welche schwarzen Handtaschen habt ihr unter 150 Eu... | 6163ms | product_catalog |
| 2 | DE | shipping | Wie lange dauert die Lieferung nach Deutschland? | 3060ms | shipping_policy |
| 3 | DE | shipping | Was sind die Versandkosten nach Österreich? | 3629ms | shipping_policy |
| 4 | DE | returns | Wie lange habe ich Zeit, einen Artikel zurückzugeb... | 5382ms | returns_policy |
| 5 | DE | returns | Wer trägt die Rücksendekosten? | 4508ms | returns_policy |
| 6 | DE | payment | Welche Zahlungsmethoden akzeptiert ihr? | 4195ms | website, shipping_policy, privacy_policy |
| 7 | DE | shipping | Versendet ihr in die Schweiz? | 3058ms | website, privacy_policy, shipping_policy |
| 8 | DE | shipping | Was passiert wenn mein Paket nicht zugestellt werd... | 5955ms | website, terms_policy |
| 9 | DE | product | Habt ihr Geldbeutel im Sortiment? | 3933ms | product_catalog |
| 10 | DE | privacy | Wer ist die verantwortliche Stelle für den Datensc... | 2718ms | privacy_policy |
| 11 | EN | returns | What is the return window at Makani Germany? | 3873ms | returns_policy |
| 12 | EN | shipping | Do you ship to Switzerland? | 4524ms | shipping_policy |
| 13 | EN | product | What black handbags do you have available? | 4711ms | product_catalog |
| 14 | EN | shipping | How long does delivery take to Germany? | 2806ms | shipping_policy |
| 15 | EN | returns | Who pays for return shipping? | 6919ms | returns_policy |
| 16 | EN | shipping | Is shipping free to Germany? | 3379ms | shipping_policy |
| 17 | EN | shipping | What happens if my package cannot be delivered? | 4325ms | shipping_policy |
| 18 | EN | product | Do you offer wallets? | 5377ms | product_catalog |
| 19 | EN | shipping | What is the shipping cost to Austria? | 3085ms | shipping_policy |
| 20 | EN | returns | How will I be refunded after a return? | 3684ms | returns_policy |

---

## Architecture

```
Query
  │
  ├── Dense Search (FAISS cosine similarity)
  │     └── text-embedding-3-small embeddings
  │
  ├── Sparse Search (BM25 keyword matching)
  │
  ├── RRF Fusion (k=60, Cormack 2009)
  │
  ├── Cross-Encoder Reranking
  │     └── cross-encoder/ms-marco-MiniLM-L-6-v2
  │
  └── Claude claude-sonnet-4-6 Generation
        └── Streaming with conversation memory
```

---

*Evaluated using [RAGAS](https://github.com/explodinggradients/ragas) framework.*
