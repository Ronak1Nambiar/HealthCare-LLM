"""
RAG (Retrieval-Augmented Generation) module for HealthCare LLM.

Uses TF-IDF cosine similarity to retrieve relevant healthcare documents,
then injects them as grounded context for the LLM.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from .config import PROCESSED_DIR, RAG_DIR

RAG_INDEX_PATH = RAG_DIR / "rag_index.pkl"
RAG_DOCS_PATH = RAG_DIR / "rag_docs.json"


class RAGRetriever:
    """CPU-friendly TF-IDF retriever over indexed healthcare documents."""

    def __init__(self) -> None:
        self._vectorizer: Any = None
        self._matrix: Any = None
        self._docs: list[dict[str, str]] = []

    # ── Build ──────────────────────────────────────────────────────────────

    def build(self, documents: list[dict[str, str]]) -> None:
        """
        Build a TF-IDF index from a list of document dicts.
        Each dict must have a 'text' key; other keys are stored as metadata.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        if not documents:
            raise ValueError("Cannot build RAG index from empty document list.")

        self._docs = documents
        texts = [d["text"] for d in documents]
        self._vectorizer = TfidfVectorizer(
            max_features=20_000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(texts)
        print(
            f"[RAG] Built TF-IDF index: {len(documents)} docs, "
            f"{self._matrix.shape[1]} features"
        )

    # ── Retrieve ───────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Return top-k docs most relevant to *query* (by cosine similarity)."""
        if not self.is_built:
            return []

        import numpy as np

        query_vec = self._vectorizer.transform([query])
        scores = (self._matrix @ query_vec.T).toarray().flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0.0:
                results.append({**self._docs[idx], "score": float(scores[idx])})
        return results

    def format_context(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve docs and format them as a numbered context block
        suitable for injection into a prompt.
        """
        hits = self.retrieve(query, top_k=top_k)
        if not hits:
            return ""
        lines = ["Retrieved knowledge base context:"]
        for i, doc in enumerate(hits, 1):
            snippet = doc["input"][:400].strip()
            task_tag = doc.get("task", "")
            lines.append(f"[RAG-{i}] ({task_tag}) {snippet}")
        return "\n".join(lines)

    # ── Persist ────────────────────────────────────────────────────────────

    def save(
        self,
        index_path: Path = RAG_INDEX_PATH,
        docs_path: Path = RAG_DOCS_PATH,
    ) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(index_path, "wb") as fh:
            pickle.dump(
                {"vectorizer": self._vectorizer, "matrix": self._matrix}, fh
            )
        with open(docs_path, "w", encoding="utf-8") as fh:
            json.dump(self._docs, fh, ensure_ascii=True)
        print(f"[RAG] Saved index -> {index_path}")
        print(f"[RAG] Saved docs  -> {docs_path}")

    def load(
        self,
        index_path: Path = RAG_INDEX_PATH,
        docs_path: Path = RAG_DOCS_PATH,
    ) -> bool:
        """Load saved index. Returns True on success, False if files missing."""
        if not index_path.exists() or not docs_path.exists():
            return False
        with open(index_path, "rb") as fh:
            data = pickle.load(fh)
        self._vectorizer = data["vectorizer"]
        self._matrix = data["matrix"]
        with open(docs_path, encoding="utf-8") as fh:
            self._docs = json.load(fh)
        return True

    # ── Status ─────────────────────────────────────────────────────────────

    @property
    def is_built(self) -> bool:
        return self._vectorizer is not None and len(self._docs) > 0

    @property
    def doc_count(self) -> int:
        return len(self._docs)


# ── Module-level singleton ─────────────────────────────────────────────────

_retriever: RAGRetriever | None = None


def get_retriever() -> RAGRetriever:
    """Return the module-level retriever, lazy-loading from disk if needed."""
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
        _retriever.load()
    return _retriever


def reset_retriever() -> None:
    """Force the singleton to be re-created on next access (after re-indexing)."""
    global _retriever
    _retriever = None


# ── Build helper ───────────────────────────────────────────────────────────

def build_rag_from_processed_data(data_dir: Path = PROCESSED_DIR) -> RAGRetriever:
    """
    Build a RAG index from the processed JSONL files produced by data_prep.
    Documents are constructed from instruction + input + output text.
    """
    documents: list[dict[str, str]] = []
    for split in ("train.jsonl", "val.jsonl", "test.jsonl"):
        path = data_dir / split
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                instruction = row.get("instruction", "")
                inp = row.get("input", "")
                out = row.get("output", "")
                # Combine all text for TF-IDF; keep raw fields as metadata
                combined = f"{instruction} {inp} {out}"
                documents.append(
                    {
                        "text": combined,
                        "task": row.get("task", ""),
                        "input": inp,
                        "output": out,
                        "id": row.get("id", ""),
                    }
                )

    if not documents:
        raise ValueError(
            "No processed data found in "
            f"{data_dir}. Run `python -m src.data_prep` first."
        )

    retriever = RAGRetriever()
    retriever.build(documents)
    return retriever
