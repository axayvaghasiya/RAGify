"""
llm/prompt_templates.py

Prompt templates for the RAG chain.

Two templates:
  1. RAG_SYSTEM_PROMPT  — system message defining the assistant's role,
                          constraints, and citation format
  2. build_rag_prompt() — assembles the full prompt with retrieved context
                          and conversation history

Design principles:
  - Answer ONLY from provided context (no hallucination)
  - Always cite sources by type (policy, product, website)
  - Respond in the same language as the user's question
  - Be concise and helpful — this is a fashion store assistant
"""

from langchain.schema import Document

# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════

RAG_SYSTEM_PROMPT = """Du bist ein hilfreicher KI-Assistent für Makani Germany, \
einen deutschen Fashion- und Handtaschen-Shop.

<rolle>
Du beantwortest Fragen von Kunden zu Produkten, Versand, Retouren, \
Zahlungsmethoden und Shop-Richtlinien.
Du hilfst Kunden, das richtige Produkt zu finden.
</rolle>

<regeln>
1. Beantworte Fragen NUR auf Basis des bereitgestellten Kontexts.
2. Wenn die Antwort nicht im Kontext steht, sage ehrlich: \
"Diese Information habe ich leider nicht. Bitte kontaktiere uns direkt."
3. Antworte in der Sprache, in der die Frage gestellt wurde \
(Deutsch oder Englisch).
4. Zitiere immer die Quelle deiner Antwort am Ende \
(z.B. [Quelle: Versandrichtlinie] oder [Quelle: Produkt]).
5. Sei freundlich, präzise und kurz — maximal 3-4 Sätze wenn möglich.
6. Erfinde KEINE Produktdetails, Preise oder Richtlinien.
</regeln>

<kontext_format>
Der Kontext besteht aus mehreren Abschnitten mit [TYP] Markierungen:
  [PRODUKT]        — Produktinformationen aus dem Katalog
  [VERSAND]        — Versand- und Zahlungsrichtlinien
  [RETOURE]        — Rückgabe- und Widerrufsrichtlinien
  [DATENSCHUTZ]    — Datenschutzerklärung
  [AGB]            — Allgemeine Geschäftsbedingungen
  [WEBSITE]        — Informationen von der Website
</kontext_format>"""


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE TYPE → LABEL MAPPING
# ══════════════════════════════════════════════════════════════════════════════

SOURCE_LABELS = {
    "product_catalog": "PRODUKT",
    "shipping_policy": "VERSAND",
    "returns_policy":  "RETOURE",
    "privacy_policy":  "DATENSCHUTZ",
    "terms_policy":    "AGB",
    "website":         "WEBSITE",
}


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_rag_prompt(
    query:           str,
    retrieved_docs:  list[tuple[Document, float]],
    chat_history:    list[dict] | None = None,
) -> list[dict]:
    """
    Builds the full message list for the Claude API call.

    Structure:
      [
        {"role": "system",    "content": RAG_SYSTEM_PROMPT},
        {"role": "user",      "content": "previous question"},   # if history
        {"role": "assistant", "content": "previous answer"},     # if history
        {"role": "user",      "content": context + current query}
      ]

    Args:
        query:          Current user question.
        retrieved_docs: List of (Document, score) from retriever.
        chat_history:   Optional list of previous turns
                        [{"role": "user"|"assistant", "content": "..."}]

    Returns:
        List of message dicts ready for Anthropic API.
    """
    # ── Build context block ───────────────────────────────────────────────────
    context_parts = []

    for doc, score in retrieved_docs:
        source_type = doc.metadata.get("source_type", "website")
        label       = SOURCE_LABELS.get(source_type, "INFO")

        # Add product title as header if available
        title = doc.metadata.get("title", "")
        if title and title != "nan":
            header = f"[{label}] {title}"
        else:
            header = f"[{label}]"

        context_parts.append(f"{header}\n{doc.page_content.strip()}")

    context_block = "\n\n---\n\n".join(context_parts)

    # ── Build user message with context ───────────────────────────────────────
    user_message = f"""<kontext>
{context_block}
</kontext>

Frage: {query}"""

    # ── Assemble messages list ─────────────────────────────────────────────────
    messages = []

    # Add conversation history (last 4 turns = 8 messages max)
    if chat_history:
        for turn in chat_history[-8:]:
            messages.append({
                "role":    turn["role"],
                "content": turn["content"],
            })

    # Add current user message with context
    messages.append({
        "role":    "user",
        "content": user_message,
    })

    return messages


# ══════════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Create mock retrieved docs for testing
    mock_docs = [
        (
            Document(
                page_content="Produkt: HALIA - BLACK\nPreis: 99.9 EUR\nKategorie: Handtaschen",
                metadata={"source_type": "product_catalog", "title": "HALIA - BLACK"}
            ),
            0.85
        ),
        (
            Document(
                page_content="Versandkostenfrei ab 49,90€ innerhalb Deutschlands.",
                metadata={"source_type": "shipping_policy"}
            ),
            0.72
        ),
    ]

    messages = build_rag_prompt(
        query="Wie viel kostet die HALIA Tasche?",
        retrieved_docs=mock_docs,
    )

    print("── System Prompt ─────────────────────────────────────────")
    print(RAG_SYSTEM_PROMPT[:300] + "...")

    print("\n── Built Messages ────────────────────────────────────────")
    for msg in messages:
        print(f"\n[{msg['role'].upper()}]")
        print(msg["content"][:400])