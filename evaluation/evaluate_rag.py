"""
evaluation/evaluate_rag.py

RAGAS evaluation of the RAG pipeline against a golden dataset.

Metrics computed:
  - faithfulness       : Is the answer grounded in the retrieved context?
  - answer_relevancy   : Does the answer address the question?
  - context_precision  : Are the retrieved chunks actually relevant?

Output:
  - Console table with per-question and aggregate scores
  - evaluation/results.json  — full results for the README
  - evaluation/report.md     — formatted markdown report

Run:
    PYTHONPATH=. python3 evaluation/evaluate_rag.py
"""

import os
import json
import time

from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from langchain.schema import Document

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
GOLDEN_PATH    = PROJECT_ROOT / "evaluation" / "golden_dataset.json"
RESULTS_PATH   = PROJECT_ROOT / "evaluation" / "results.json"
REPORT_PATH    = PROJECT_ROOT / "evaluation" / "report.md"

# ── RAGAS uses OpenAI for its own internal scoring ─────────────────────────────
# This is separate from your RAG chain's LLM.
# RAGAS needs an LLM to judge faithfulness and relevancy.
RAGAS_MODEL = "gpt-4o-mini"


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD GOLDEN DATASET
# ══════════════════════════════════════════════════════════════════════════════

def load_golden_dataset(path: Path = GOLDEN_PATH) -> list[dict]:
    """Load the 20-question golden Q&A dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Golden dataset not found at {path}\n"
            "Make sure golden_dataset.json is in the evaluation/ folder."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"📂 Loaded {len(data)} golden Q&A pairs")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# 2. RUN RAG PIPELINE ON GOLDEN QUESTIONS
# ══════════════════════════════════════════════════════════════════════════════

def run_rag_on_dataset(golden_data: list[dict]) -> list[dict]:
    """
    Runs each golden question through the full RAG pipeline.
    Collects: question, answer, contexts, ground_truth.

    This is what RAGAS evaluates.
    """
    # Import here to avoid slow startup if just checking args
    from retrieval.retriever import HybridRetriever
    from llm.rag_chain import RAGChain

    print("\n🔧 Initialising RAG pipeline ...")
    chain = RAGChain()

    results = []
    total   = len(golden_data)

    print(f"\n🚀 Running {total} questions through RAG pipeline ...\n")

    for i, item in enumerate(golden_data, 1):
        question    = item["question"]
        ground_truth = item["ground_truth"]
        language    = item.get("language", "de")
        category    = item.get("category", "general")

        print(f"  [{i:02d}/{total}] [{language.upper()}] {question[:60]}...")

        start = time.time()

        # Reset history between questions — each is independent
        chain.clear_history()

        # Get full result (non-streaming) with retrieved docs
        result = chain.query(question)

        latency = int((time.time() - start) * 1000)

        # Extract context strings from retrieved docs
        contexts = [
            doc.page_content
            for doc, _ in result.get("retrieved", [])
        ]

        results.append({
            "question":    question,
            "answer":      result["answer"],
            "contexts":    contexts,
            "ground_truth": ground_truth,
            "language":    language,
            "category":    category,
            "latency_ms":  latency,
            "sources":     result.get("sources", []),
        })

        print(f"         ✅ {latency}ms | sources: {result.get('sources', [])}")

    print(f"\n✅ Pipeline run complete — {len(results)} answers collected")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 3. RUN RAGAS EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def run_ragas_evaluation(rag_results: list[dict]) -> dict:
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.metrics import faithfulness, answer_relevancy, context_precision

    print("\n📊 Running RAGAS evaluation ...")
    print(f"   Evaluation model: {RAGAS_MODEL}\n")

    openai_api_key = os.getenv("OPENAI_API_KEY")

    ragas_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=RAGAS_MODEL,
            api_key=openai_api_key,
            temperature=0,
        )
    )

    ragas_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=openai_api_key,
        )
    )

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["ground_truth"],
        )
        for r in rag_results
    ]

    dataset = EvaluationDataset(samples=samples)

    scores = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    return scores


# ══════════════════════════════════════════════════════════════════════════════
# 4. SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def save_results(
    rag_results:  list[dict],
    ragas_scores: dict,
) -> None:
    """Save full results to JSON and markdown report."""

    df = ragas_scores.to_pandas()
    scores_dict = {
        "faithfulness":      float(df["faithfulness"].mean()),
        "answer_relevancy":  float(df["answer_relevancy"].mean()),
        "context_precision": float(df["context_precision"].mean()),
    }

    # ── JSON results ───────────────────────────────────────────────────────────
    output = {
        "timestamp":    datetime.utcnow().isoformat(),
        "total_questions": len(rag_results),
        "aggregate_scores": {
            "faithfulness":      round(float(scores_dict.get("faithfulness", 0)), 4),
            "answer_relevancy":  round(float(scores_dict.get("answer_relevancy", 0)), 4),
            "context_precision": round(float(scores_dict.get("context_precision", 0)), 4),
        },
        "per_question": rag_results,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Results saved → {RESULTS_PATH}")

    # ── Markdown report ────────────────────────────────────────────────────────
    agg = output["aggregate_scores"]

    # Per-category breakdown
    by_category: dict[str, list] = {}
    for r in rag_results:
        cat = r.get("category", "general")
        by_category.setdefault(cat, []).append(r)

    # Per-language breakdown
    de_results = [r for r in rag_results if r.get("language") == "de"]
    en_results = [r for r in rag_results if r.get("language") == "en"]

    avg_latency    = sum(r["latency_ms"] for r in rag_results) // len(rag_results)
    de_avg_latency = sum(r["latency_ms"] for r in de_results) // max(len(de_results), 1)
    en_avg_latency = sum(r["latency_ms"] for r in en_results) // max(len(en_results), 1)

    report = f"""# RAGify AI — RAGAS Evaluation Report

**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  
**Dataset:** {len(rag_results)} questions (10 German + 10 English)  
**RAG Model:** claude-sonnet-4-6  
**Embedding Model:** text-embedding-3-small  
**Retrieval:** Hybrid BM25 + FAISS + Cross-Encoder Reranking  

---

## Aggregate Scores

| Metric | Score |
|--------|-------|
| **Faithfulness** | {agg['faithfulness']:.4f} |
| **Answer Relevancy** | {agg['answer_relevancy']:.4f} |
| **Context Precision** | {agg['context_precision']:.4f} |

> **Faithfulness** measures whether the answer is grounded in the retrieved context (no hallucination).  
> **Answer Relevancy** measures whether the answer addresses the question asked.  
> **Context Precision** measures whether the retrieved chunks are relevant to the question.

---

## Performance by Language

| Language | Questions | Avg Latency |
|----------|-----------|-------------|
| German (DE) | {len(de_results)} | {de_avg_latency}ms |
| English (EN) | {len(en_results)} | {en_avg_latency}ms |
| **Overall** | {len(rag_results)} | {avg_latency}ms |

---

## Performance by Category

| Category | Questions | Avg Latency |
|----------|-----------|-------------|
"""
    for cat, items in sorted(by_category.items()):
        avg_lat = sum(r["latency_ms"] for r in items) // len(items)
        report += f"| {cat.capitalize()} | {len(items)} | {avg_lat}ms |\n"

    report += f"""
---

## Per-Question Results

| # | Language | Category | Question | Latency | Sources |
|---|----------|----------|----------|---------|---------|
"""
    for i, r in enumerate(rag_results, 1):
        q        = r["question"][:50] + "..." if len(r["question"]) > 50 else r["question"]
        sources  = ", ".join(r.get("sources", []))
        report  += f"| {i} | {r['language'].upper()} | {r['category']} | {q} | {r['latency_ms']}ms | {sources} |\n"

    report += """
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
"""

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"💾 Report saved  → {REPORT_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# PRINT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(rag_results: list[dict], ragas_scores: dict) -> None:
    """Print a clean summary table to the console."""
    
    df = ragas_scores.to_pandas()
    scores_dict = {
        "faithfulness":      float(df["faithfulness"].mean()),
        "answer_relevancy":  float(df["answer_relevancy"].mean()),
        "context_precision": float(df["context_precision"].mean()),
    }

    agg = {
        "faithfulness":      float(scores_dict.get("faithfulness", 0)),
        "answer_relevancy":  float(scores_dict.get("answer_relevancy", 0)),
        "context_precision": float(scores_dict.get("context_precision", 0)),
    }

    avg_latency = sum(r["latency_ms"] for r in rag_results) // len(rag_results)

    print("\n" + "=" * 60)
    print("  RAGAS EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Questions evaluated : {len(rag_results)}")
    print(f"  Average latency     : {avg_latency}ms")
    print()
    print(f"  Faithfulness        : {agg['faithfulness']:.4f}")
    print(f"  Answer Relevancy    : {agg['answer_relevancy']:.4f}")
    print(f"  Context Precision   : {agg['context_precision']:.4f}")
    print("=" * 60)

    # Highlight any weak spots
    for metric, score in agg.items():
        if score < 0.7:
            print(f"  ⚠️  {metric} is below 0.70 — consider tuning retrieval")
        elif score >= 0.85:
            print(f"  ✅ {metric} is strong (≥0.85)")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  RAGify AI · RAGAS Evaluation")
    print("=" * 60)

    rag_cache = PROJECT_ROOT / "evaluation" / "rag_results_cache.json"

    # Load from cache if available — avoids re-running 20 Claude queries
    if rag_cache.exists():
        print("📂 Found cached RAG results — skipping pipeline run")
        with open(rag_cache, "r", encoding="utf-8") as f:
            rag_results = json.load(f)
    else:
        # 1. Load golden dataset
        golden_data = load_golden_dataset()

        # 2. Run RAG pipeline on all questions
        rag_results = run_rag_on_dataset(golden_data)

        # 3. Cache results to disk
        rag_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(rag_cache, "w", encoding="utf-8") as f:
            json.dump(rag_results, f, ensure_ascii=False, indent=2)
        print(f"💾 RAG results cached → {rag_cache}")

    # 4. Run RAGAS metrics
    ragas_scores = run_ragas_evaluation(rag_results)

    # 5. Save results + report
    save_results(rag_results, ragas_scores)

    # 6. Print summary
    print_summary(rag_results, ragas_scores)

    # 5. Print summary
    print_summary(rag_results, ragas_scores)