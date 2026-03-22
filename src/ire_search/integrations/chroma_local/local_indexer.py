"""
Local vector index using ChromaDB (persistent) + Sentence-Transformers embeddings.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "documents"


def _distance_to_similarity(distance: float) -> float:
    """Map Chroma distance (lower is better) to a 0–1-ish score for UI."""
    if distance is None:
        return 0.0
    return float(1.0 / (1.0 + max(0.0, distance)))


class LocalIndexer:
    """
    Persist embeddings in `.chroma_db/` (or a custom path) and query by semantic similarity.
    """

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        collection_name: str = DEFAULT_COLLECTION,
        model_name: str = DEFAULT_MODEL,
    ):
        self.collection_name = collection_name
        self.model_name = model_name
        base = Path(persist_directory) if persist_directory else Path.cwd() / ".chroma_db"
        self.persist_directory = base.resolve()
        self._client = None
        self._collection = None
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _get_collection(self):
        if self._collection is None:
            import chromadb

            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_directory))
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_documents(
        self,
        documents: Iterable[Dict[str, Any]],
        batch_size: int = 32,
    ) -> int:
        """
        Index documents. Each item should have:
          - doc_id: str (unique)
          - text or content: str (body to embed)
          - metadata: optional dict (values must be Chroma-serializable: str, int, float, bool)
        """
        model = self._get_model()
        collection = self._get_collection()

        ids: List[str] = []
        texts: List[str] = []
        metas: List[Dict[str, Any]] = []

        for doc in documents:
            did = str(doc.get("doc_id", "")).strip()
            body = doc.get("text") or doc.get("content") or ""
            body = str(body)[:50_000]
            if not did or not body.strip():
                continue
            meta = dict(doc.get("metadata") or {})
            for k, v in list(meta.items()):
                if v is None:
                    del meta[k]
                elif isinstance(v, (dict, list)):
                    meta[k] = str(v)
                elif not isinstance(v, (str, int, float, bool)):
                    meta[k] = str(v)
            ids.append(did)
            texts.append(str(body))
            metas.append(meta)

        if not ids:
            return 0

        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start : start + batch_size]
            batch_texts = texts[start : start + batch_size]
            batch_metas = metas[start : start + batch_size]
            embeddings = model.encode(
                batch_texts,
                batch_size=min(batch_size, len(batch_texts)),
                show_progress_bar=False,
                convert_to_numpy=True,
            ).tolist()
            collection.upsert(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_metas,
            )

        return len(ids)

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Return ranked hits with similarity_score, distance, metadata, snippet."""
        if not query or not str(query).strip():
            return []
        model = self._get_model()
        collection = self._get_collection()
        q_emb = model.encode([query.strip()], convert_to_numpy=True).tolist()
        raw = collection.query(
            query_embeddings=q_emb,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        out: List[Dict[str, Any]] = []
        ids_list = raw.get("ids") or [[]]
        docs_list = raw.get("documents") or [[]]
        meta_list = raw.get("metadatas") or [[]]
        dist_list = raw.get("distances") or [[]]
        row_ids = ids_list[0] if ids_list else []
        row_docs = docs_list[0] if docs_list else []
        row_metas = meta_list[0] if meta_list else []
        row_dists = dist_list[0] if dist_list else []

        for i, did in enumerate(row_ids):
            dist = row_dists[i] if i < len(row_dists) else 0.0
            doc_text = row_docs[i] if i < len(row_docs) else ""
            meta = row_metas[i] if i < len(row_metas) else {}
            out.append(
                {
                    "doc_id": did,
                    "distance": float(dist),
                    "similarity_score": _distance_to_similarity(float(dist)),
                    "metadata": meta or {},
                    "snippet": (doc_text or "")[:400],
                }
            )
        return out

    def reset(self) -> None:
        """Clear all vectors: drop the collection and recreate (avoids file locks on Windows)."""
        self._model = None
        self._collection = None
        import chromadb

        self.persist_directory.mkdir(parents=True, exist_ok=True)
        if self._client is None:
            self._client = chromadb.PersistentClient(path=str(self.persist_directory))
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def wipe_persist_directory(self) -> None:
        """Remove the entire `.chroma_db` folder. Call when no client is active (e.g. app exit)."""
        self._collection = None
        self._client = None
        self._model = None
        if self.persist_directory.exists():
            shutil.rmtree(self.persist_directory)
