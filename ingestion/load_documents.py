"""
ingestion/load_documents.py

Loads all raw data sources into a unified list of LangChain Documents.
Each document carries metadata so we can filter and cite sources later.

Sources handled:
  - Policy .txt files  (returns, shipping, privacy, terms)
  - Website pages      (scraped from makani-germany.de)
  - Product CSV        (added later when merchant provides it)
"""

import os
import re
import time
import requests
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain.schema import Document

# ── Load environment variables from .env ──────────────────────────────────────
load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


# ══════════════════════════════════════════════════════════════════════════════
# 1. POLICY TEXT FILES
# ══════════════════════════════════════════════════════════════════════════════

def load_policy_files(raw_dir: Path = RAW_DIR) -> list[Document]:
    """
    Loads all .txt policy files from data/raw/.
    Each file becomes one Document with rich metadata.

    Returns:
        List of LangChain Document objects, one per policy file.
    """
    documents = []

    # Map filenames to human-readable document types
    policy_type_map = {
        "returns_policy.txt":  "returns_policy",
        "shipping_policy.txt": "shipping_policy",
        "privacy_policy.txt":  "privacy_policy",
        "terms_policy.txt":    "terms_policy",
    }

    txt_files = list(raw_dir.glob("*.txt"))

    if not txt_files:
        print(f"⚠️  No .txt files found in {raw_dir}")
        return documents

    for filepath in txt_files:
        filename = filepath.name
        doc_type = policy_type_map.get(filename, "policy")

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                print(f"⚠️  Skipping empty file: {filename}")
                continue

            doc = Document(
                page_content=content,
                metadata={
                    "source":      str(filepath),
                    "filename":    filename,
                    "source_type": doc_type,
                    "language":    "de",         # German merchant data
                    "domain":      "store_policy",
                }
            )
            documents.append(doc)
            print(f"✅ Loaded policy: {filename} ({len(content):,} chars)")

        except Exception as e:
            print(f"❌ Failed to load {filename}: {e}")

    return documents


# ══════════════════════════════════════════════════════════════════════════════
# 2. WEBSITE SCRAPER
# ══════════════════════════════════════════════════════════════════════════════

# Pages to scrape from makani-germany.de
# Add or remove URLs here as needed — no code changes required elsewhere
TARGET_URLS = [
    "https://makani-germany.de/",
    "https://makani-germany.de/pages/uber-uns",
    "https://makani-germany.de/pages/faq",
    "https://makani-germany.de/pages/kontaktformular",
    "https://makani-germany.de/collections",
    "https://makani-germany.de/collections/all",
]

# HTML tags that contain meaningful content (skip nav, footer, scripts)
CONTENT_TAGS = ["p", "h1", "h2", "h3", "h4", "li", "span", "div"]

# Tags to remove entirely before parsing
NOISE_TAGS = [
    "script", "style", "nav", "footer", "header",
    "noscript", "iframe", "form", "button"
]


def _clean_text(text: str) -> str:
    """Remove excess whitespace and blank lines from scraped text."""
    text = re.sub(r'\n{3,}', '\n\n', text)   # max 2 consecutive newlines
    text = re.sub(r'[ \t]+', ' ', text)       # collapse spaces/tabs
    return text.strip()


def _scrape_single_page(url: str, session: requests.Session) -> Document | None:
    """
    Scrapes a single URL and returns a Document, or None if it fails.

    Args:
        url:     The page URL to scrape.
        session: Shared requests.Session for connection reuse.

    Returns:
        A Document with page text and metadata, or None on failure.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove noise tags in place
        for tag in soup(NOISE_TAGS):
            tag.decompose()

        # Extract page title for metadata
        title_tag = soup.find("title")
        page_title = title_tag.get_text(strip=True) if title_tag else url

        # Get main content area if available, else use full body
        main = (
            soup.find("main")
            or soup.find("article")
            or soup.find(id="content")
            or soup.find(class_="content")
            or soup.body
        )

        if not main:
            print(f"⚠️  No content found at: {url}")
            return None

        raw_text = main.get_text(separator="\n")
        clean_text = _clean_text(raw_text)

        # Skip pages with very little content (likely 404s or redirects)
        if len(clean_text) < 100:
            print(f"⚠️  Too little content at: {url} — skipping")
            return None

        return Document(
            page_content=clean_text,
            metadata={
                "source":      url,
                "page_title":  page_title,
                "source_type": "website",
                "language":    "de",
                "domain":      "store_website",
            }
        )

    except requests.exceptions.HTTPError as e:
        print(f"⚠️  HTTP error scraping {url}: {e}")
        return None
    except requests.exceptions.Timeout:
        print(f"⚠️  Timeout scraping {url}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error scraping {url}: {e}")
        return None


def load_website_pages(urls: list[str] = TARGET_URLS) -> list[Document]:
    """
    Scrapes a list of URLs and returns Documents.
    Includes polite delay between requests.

    Args:
        urls: List of URLs to scrape.

    Returns:
        List of Document objects, one per successfully scraped page.
    """
    documents = []

    print(f"\n🌐 Scraping {len(urls)} pages from makani-germany.de ...")

    with requests.Session() as session:
        for i, url in enumerate(urls):
            doc = _scrape_single_page(url, session)
            if doc:
                documents.append(doc)
                print(f"✅ Scraped: {url} ({len(doc.page_content):,} chars)")

            # Polite delay — don't hammer the server
            if i < len(urls) - 1:
                time.sleep(1.5)

    return documents


# ══════════════════════════════════════════════════════════════════════════════
# 3. PRODUCT CSV  (ready for when merchant sends the file)
# ══════════════════════════════════════════════════════════════════════════════

def load_product_csv(csv_path: Path | None = None) -> list[Document]:
    """
    Loads Shopify product CSV export and converts each product into a Document.

    Shopify CSV columns used:
        Title, Body (HTML), Vendor, Type, Tags,
        Variant Price, Variant SKU, Image Src

    Args:
        csv_path: Path to the products CSV. Defaults to data/raw/products.csv

    Returns:
        List of Documents, one per product row.
    """
    if csv_path is None:
        csv_path = RAW_DIR / "products.csv"

    if not csv_path.exists():
        print(f"ℹ️  No product CSV found at {csv_path} — skipping.")
        print("    Drop products.csv into data/raw/ when the merchant sends it.")
        return []

    documents = []

    try:
        df = pd.read_csv(csv_path)
        print(f"\n📦 Loading product CSV: {len(df)} rows found")

        # Shopify exports one row per variant — keep only the first row per product
        # (identified by a non-empty Title column)
        df = df[df["Title"].notna() & (df["Title"].str.strip() != "")]

        for _, row in df.iterrows():
            # Strip HTML tags from Body (HTML) column
            body_html = str(row.get("Body (HTML)", ""))
            body_text = BeautifulSoup(body_html, "html.parser").get_text(
                separator=" "
            ).strip()

            # Build a clean natural-language product description
            parts = []

            title = str(row.get("Title", "")).strip()
            if title:
                parts.append(f"Produkt: {title}")

            vendor = str(row.get("Vendor", "")).strip()
            if vendor and vendor != "nan":
                parts.append(f"Marke: {vendor}")

            product_type = str(row.get("Type", "")).strip()
            if product_type and product_type != "nan":
                parts.append(f"Kategorie: {product_type}")

            tags = str(row.get("Tags", "")).strip()
            if tags and tags != "nan":
                parts.append(f"Tags: {tags}")

            price = str(row.get("Variant Price", "")).strip()
            if price and price != "nan":
                parts.append(f"Preis: {price} EUR")

            if body_text:
                parts.append(f"Beschreibung: {body_text}")

            page_content = "\n".join(parts)

            if len(page_content.strip()) < 20:
                continue  # skip essentially empty rows

            doc = Document(
                page_content=page_content,
                metadata={
                    "source":      str(csv_path),
                    "source_type": "product_catalog",
                    "language":    "de",
                    "domain":      "product",
                    "title":       title,
                    "vendor":      vendor,
                    "product_type": product_type,
                    "price":       price,
                    "tags":        tags,
                    "sku":         str(row.get("Variant SKU", "")).strip(),
                }
            )
            documents.append(doc)

        print(f"✅ Loaded {len(documents)} products from CSV")

    except Exception as e:
        print(f"❌ Failed to load product CSV: {e}")

    return documents


# ══════════════════════════════════════════════════════════════════════════════
# 4. MASTER LOADER  — calls everything above
# ══════════════════════════════════════════════════════════════════════════════

def load_all_documents() -> list[Document]:
    """
    Master function — loads all available data sources.
    Safe to call at any point: skips sources that aren't ready yet.

    Returns:
        Combined list of all Documents across all sources.
    """
    print("=" * 60)
    print("  RAG-AI-ASSISTANT · Document Loader")
    print("  Merchant: Makani Germany (Fashion/Apparel)")
    print("  Language: German (de)")
    print("=" * 60)

    all_documents = []

    # 1. Policy files
    print("\n📄 Loading policy files ...")
    policy_docs = load_policy_files()
    all_documents.extend(policy_docs)

    # 2. Website
    print("\n🌐 Loading website pages ...")
    website_docs = load_website_pages()
    all_documents.extend(website_docs)

    # 3. Product CSV (skips gracefully if not yet available)
    print("\n📦 Loading product catalog ...")
    product_docs = load_product_csv()
    all_documents.extend(product_docs)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  ✅ Total documents loaded: {len(all_documents)}")
    print(f"     Policy files : {len(policy_docs)}")
    print(f"     Website pages: {len(website_docs)}")
    print(f"     Products     : {len(product_docs)}")
    print("=" * 60 + "\n")

    return all_documents


# ══════════════════════════════════════════════════════════════════════════════
# QUICK TEST  — run this file directly to verify everything loads
# ══════════════════════════════════════════════════════════════════════════════

import json

def save_documents(docs: list[Document], output_path: Path) -> None:
    """Save documents to JSON so we don't re-scrape every run."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    serialized = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in docs
    ]
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Saved {len(docs)} documents → {output_path}")


def load_saved_documents(input_path: Path) -> list[Document]:
    """Load previously saved documents from JSON. Skips scraping."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    docs = [
        Document(page_content=d["page_content"], metadata=d["metadata"])
        for d in data
    ]
    print(f"📂 Loaded {len(docs)} documents from cache → {input_path}")
    return docs


if __name__ == "__main__":
    output_path = Path("data/processed/documents.json")
    
    # If saved file exists, load from disk — skip scraping
    # if output_path.exists():
    #     print("📂 Found existing documents.json — loading from cache.")
    #     print("   Delete data/processed/documents.json to re-scrape.\n")
    #     docs = load_saved_documents(output_path)
    # else:
    #     docs = load_all_documents()
    #     save_documents(docs, output_path)

     # Always reload everything and overwrite
    docs = load_all_documents()
    save_documents(docs, output_path)

    # Preview
    seen_types = set()
    print("\n── Sample documents ──────────────────────────────────────")
    for doc in docs:
        source_type = doc.metadata.get("source_type", "unknown")
        if source_type not in seen_types:
            seen_types.add(source_type)
            print(f"\n[{source_type.upper()}]")
            print(f"  Source : {doc.metadata.get('source', '')}")
            print(f"  Preview: {doc.page_content[:200].strip()} ...")