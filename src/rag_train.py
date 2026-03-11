"""
Build and save the RAG knowledge-base index.

Run:
    python -m src.rag_train
    python -m src.rag_train --data_dir data/processed
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import PROCESSED_DIR, RAG_DIR, ensure_dirs
from .rag import RAG_DOCS_PATH, RAG_INDEX_PATH, build_rag_from_processed_data, reset_retriever


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build TF-IDF RAG index from processed healthcare data"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(PROCESSED_DIR),
        help="Directory containing train/val/test .jsonl files",
    )
    args = parser.parse_args()

    ensure_dirs()
    RAG_DIR.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    print(f"Building RAG index from: {data_dir}")

    retriever = build_rag_from_processed_data(data_dir)
    retriever.save(RAG_INDEX_PATH, RAG_DOCS_PATH)

    # Invalidate any cached singleton
    reset_retriever()

    print(f"\nRAG index ready.")
    print(f"  Documents indexed : {retriever.doc_count}")
    print(f"  Index saved to    : {RAG_INDEX_PATH}")
    print(f"  Docs saved to     : {RAG_DOCS_PATH}")
    print("\nTest a retrieval:")
    hits = retriever.retrieve("hypertension blood pressure", top_k=2)
    for h in hits:
        print(f"  [{h['score']:.3f}] ({h['task']}) {h['input'][:120]}")


if __name__ == "__main__":
    main()
