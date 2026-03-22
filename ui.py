"""
Local-first semantic search UI (Streamlit).

Run from project root:
  pip install -e ".[ui]"
  streamlit run ui.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st

from ire_search.core.file_crawler import crawl_to_list
from ire_search.integrations.chroma_local import LocalIndexer


def _indexer() -> LocalIndexer:
    persist = ROOT / ".chroma_db"
    return LocalIndexer(persist_directory=persist)


def main() -> None:
    st.set_page_config(page_title="Semantic Search (local)", layout="wide")
    st.title("Local semantic search")
    st.caption("ChromaDB + Sentence-Transformers (`all-MiniLM-L6-v2`). Data stays in `.chroma_db/`.")

    if "indexer" not in st.session_state:
        st.session_state.indexer = _indexer()

    indexer: LocalIndexer = st.session_state.indexer

    with st.sidebar:
        st.subheader("Index")
        folder = st.text_input(
            "Folder path to index",
            placeholder=str(ROOT),
            help="Absolute path to a folder containing .pdf, .docx, .pptx, .md",
        )
        if st.button("Index now", type="primary"):
            root = Path(folder.strip() or str(ROOT)).expanduser()
            if not root.is_dir():
                st.error(f"Not a directory: {root}")
            else:
                with st.spinner("Crawling and embedding… (first run downloads the model)"):
                    try:
                        docs = crawl_to_list(root)
                    except Exception as e:
                        st.error(f"Crawl failed: {e}")
                        docs = []
                    if not docs:
                        st.warning("No documents indexed (empty or unsupported files).")
                    else:
                        n = indexer.add_documents(docs)
                        st.success(f"Indexed {n} document(s) from {len(docs)} file(s) found.")
        st.divider()
        st.subheader("Danger zone")
        if st.button("Reset local vector index", type="secondary"):
            indexer.reset()
            st.session_state.indexer = _indexer()
            st.success("Removed `.chroma_db` and recreated an empty index.")

    q = st.text_input("Search", placeholder="Describe what you are looking for…")
    if st.button("Search", type="primary") and q.strip():
        with st.spinner("Searching…"):
            hits = indexer.search(q.strip(), top_k=12)
        if not hits:
            st.info("No results (index may be empty).")
        else:
            for h in hits:
                meta = h.get("metadata") or {}
                title = meta.get("filename") or meta.get("rel_path") or h.get("doc_id", "")
                path = meta.get("source_path", "")
                score = h.get("similarity_score", 0.0)
                snip = h.get("snippet", "")
                with st.container():
                    st.markdown(f"### `{title}`")
                    st.caption(f"Similarity: **{score:.4f}** (distance {h.get('distance', 0):.4f})")
                    if path:
                        st.caption(path)
                    st.write(snip[:800] + ("…" if len(snip) > 800 else ""))
                    st.divider()


if __name__ == "__main__":
    main()
