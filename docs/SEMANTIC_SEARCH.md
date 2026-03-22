# Local semantic search

## Overview

- **`LocalIndexer`** — [`src/ire_search/integrations/chroma_local/local_indexer.py`](../src/ire_search/integrations/chroma_local/local_indexer.py): Chroma persistent client (default path `.chroma_db/` at the process working directory), embeddings via `sentence-transformers` model **`all-MiniLM-L6-v2`**.
- **`file_crawler`** — [`src/ire_search/core/file_crawler.py`](../src/ire_search/core/file_crawler.py): recursive crawl of `.pdf` (via **pymupdf4llm**), `.docx` / `.pptx` / `.md` (via **markitdown**).
- **`ui.py`** — Streamlit app: folder path, **Index now**, search cards with **similarity score**, **Reset** (clears the Chroma collection).

## Install

```bash
pip install -e ".[semantic]"   # Chroma + ST + crawlers
pip install -e ".[ui]"        # above + Streamlit
```

## Run the UI

From the repository root:

```bash
streamlit run ui.py
```

## Reset behavior

`LocalIndexer.reset()` drops the Chroma **collection** and recreates it (avoids file-lock issues on Windows with SQLite under `.chroma_db/`). To delete the entire folder, use `wipe_persist_directory()` only when no client holds files open (e.g. after exiting the app).

## PyInstaller / .exe

Bundling **torch**, **sentence-transformers**, and **chromadb** into a single `--onefile` binary is possible but produces a **very large** artifact and often needs hidden imports.

- Prefer **`pyinstaller --onedir ui.py`** first, or a `.spec` with `collect_all` for `torch` and `chromadb`.
- Test on the target OS; first run may still download the Hugging Face model unless you ship the cache.

Example starting point (expect iteration):

```bash
pip install pyinstaller
pyinstaller --onedir --name ire-semantic ui.py
```

`--windowed` / `--onefile` add complexity; validate imports with `pyi-archive_viewer` if the app fails at startup.
