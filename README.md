# IRE Assignment 1 — Information Retrieval Engine

> **Pluggable IR system with Boolean, TF, TF-IDF, and BM25 scoring**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Architecture](https://img.shields.io/badge/Pattern-Strategy-purple.svg)](docs/ARCHITECTURE.md)
[![Scorers](https://img.shields.io/badge/Scorers-4-green.svg)](docs/SCORING_STRATEGIES.md)
[![Tests](https://img.shields.io/badge/Tests-pytest-orange.svg)](tests/)

---

## How to run (step by step)

**Prerequisites:** Python 3.9+ and a terminal in the **project root** (folder that contains `cli/`, `src/`, `ui.py`).

### 1. Virtual environment (recommended)

**Windows (PowerShell):**

```powershell
cd path\to\IRE_Assignment1
python -m venv env
.\env\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
cd path/to/IRE_Assignment1
python3 -m venv env
source env/bin/activate
```

### 2. Install the package

```bash
pip install -U pip
pip install -e .
pip install -r requirements.txt
```

- **Lexical IR only** — this is enough for `cli/build.py`, `query.py`, `evaluate.py`, and tests.
- **Semantic UI (Chroma + Streamlit)** — also run: `pip install -e ".[ui]"` (large download: PyTorch + sentence-transformers on first use).

### 3. Run lexical search (BM25 / TF-IDF / Boolean)

```bash
# Build a small index (fast test)
python cli/build.py -x 4 -y 1 -z 1 -optim 0 --limit 100

# Query
python cli/query.py -x 4 -y 1 -z 1 -q T -optim 0 --query "machine learning"

# Interactive queries
python cli/query.py -x 4 -y 1 -z 1 -q T -optim 0 --interactive
```

Optional shorthand from project root: `python main.py query -x 4 -y 1 -z 1 -q T -optim 0 --query "test"`.

### 4. Run tests

```bash
python -m pytest tests/ -q
```

With semantic extras: `python -m pytest tests/test_semantic.py -v`.

### 5. Run local semantic search (Streamlit)

Indexes **PDF / Word / PowerPoint / Markdown** from a folder into **ChromaDB** (`.chroma_db/`).

```bash
pip install -e ".[ui]"
streamlit run ui.py
```

Paste a **folder path** in the sidebar → **Index now** → search in natural language. **Reset** clears the vector index. See [docs/SEMANTIC_SEARCH.md](docs/SEMANTIC_SEARCH.md).

---

## Quick Start (short)

```bash
pip install -r requirements.txt

# Build a BM25 index
python cli/build.py -x 4 -y 1 -z 1 -optim 0 --limit 100

# Query it
python cli/query.py -x 4 -y 1 -z 1 -q T -optim 0 --query "machine learning"

# Interactive mode
python cli/query.py -x 4 -y 1 -z 1 -q T -optim 0 --interactive
```

## Architecture

The system uses the **Strategy Pattern** — one `SelfIndexer` class composed with pluggable scorers and storage backends:

```python
from ire_search import create_indexer

# 24 valid combinations via factory
indexer = create_indexer(x=4, y=1, z=1)  # BM25 + JSON + no compression
indexer.create_index('my_index', documents)
results = indexer.query('machine learning')
```

| Component | Options | Description |
|-----------|---------|-------------|
| **Scorer** (`x`) | 1=Boolean, 2=TF, 3=TF-IDF, **4=BM25** | Ranking algorithm |
| **Storage** (`y`) | 1=JSON, 2=SQLite | Persistence backend |
| **Compression** (`z`) | 1=None, 2=Elias γ/δ, 3=zlib | Index compression |

Identifier format: `SelfIndex_i{x}d{y}c{z}o{optim}`

## Commands

### Build
```bash
python cli/build.py -x <scorer> -y <storage> -z <compression> -optim <opt> [--limit N]
```

### Query
```bash
python cli/query.py -x 3 -y 1 -z 1 -q T -optim 0 --query "neural networks"
python cli/query.py -x 1 -y 1 -z 1 -q T -optim 0 --query '"cat" AND "dog"'
```

### Evaluate
```bash
python cli/evaluate.py -x 3 -y 1 -z 1 -optim 0         # Benchmark
python cli/generate_plots.py                              # Plots
```

### Test
```bash
python -m pytest tests/ -v
```

## Local semantic search (optional)

Uses **ChromaDB** (persistent store in `.chroma_db/`) and **Sentence-Transformers** (`all-MiniLM-L6-v2`) for embeddings. Crawls `.pdf`, `.docx`, `.pptx`, `.md` from a folder via `markitdown` / `pymupdf4llm`.

```bash
pip install -e ".[ui]"
streamlit run ui.py
```

Install semantic stack only: `pip install -e ".[semantic]"`.

See [docs/SEMANTIC_SEARCH.md](docs/SEMANTIC_SEARCH.md) for details and **PyInstaller** packaging caveats.

## Programmatic API

```python
from ire_search.core.engine import SearchEngine

engine = SearchEngine(x=4, y=1, z=1)
engine.build(documents)
response = engine.search("neural networks", top_k=5)

for result in response:
    print(f"{result.rank}. {result.doc_id}: {result.title}")
```

## Dataset

- **100,000 documents**: 50K Wikipedia + 50K news articles
- **256 test queries**: single-term, multi-term, Boolean, phrase, complex
- [Download pre-built indices](https://drive.google.com/drive/folders/126VycOfgOjit1S0xZIkBR9-jWo8lbMsp?usp=sharing)

## Documentation

| Guide | Contents |
|-------|----------|
| [Architecture](docs/ARCHITECTURE.md) | System design, Mermaid diagrams |
| [Scoring Strategies](docs/SCORING_STRATEGIES.md) | Boolean/TF/TF-IDF/BM25 math |
| [Storage Backends](docs/STORAGE_BACKENDS.md) | JSON vs SQLite + compression |
| [Compression](docs/COMPRESSION.md) | Elias γ/δ and zlib algorithms |
| [Query Processing](docs/QUERY_PROCESSING.md) | TAAT/DAAT, Shunting-yard |
| [API Reference](docs/API_REFERENCE.md) | All public classes and methods |
| [Extending](docs/EXTENDING.md) | Adding scorers, storage, RAG |
| [Evaluation](docs/EVALUATION.md) | Benchmarking and metrics |
| [User Guide](docs/USER_GUIDE.md) | End-to-end tutorial |
| [Data Guide](docs/DATA_GUIDE.md) | Custom data integration |
| [Interview study guide](notebooks/IRE_Interview_Guide.ipynb) | End-to-end prep: architecture, Q&A, STAR, CLI |
| [Semantic search](docs/SEMANTIC_SEARCH.md) | ChromaDB, Streamlit UI, PyInstaller notes |

## Key Findings

| Metric | Best | Config |
|--------|------|--------|
| Lowest latency | 2.80ms | Boolean + Skip Pointers |
| Highest throughput | 357 QPS | Boolean (TAAT) |
| Smallest disk | 164 MB | Elias compression (3.97×) |
| Best relevance | — | BM25 (new) |

## Project Structure

```
├── main.py             # Optional entrypoint → cli/
├── ui.py               # Streamlit local semantic search UI
├── cli/                # build.py, query.py, evaluate.py, plots
├── src/ire_search/     # Installable package
│   ├── core/           # SelfIndexer, SearchEngine, preprocessor, file_crawler
│   ├── scoring/        # Boolean / TF / TF-IDF / BM25
│   ├── storage/        # JSON + SQLite backends
│   ├── compression/    # Elias + zlib
│   ├── io/             # Parquet / news loaders
│   ├── evaluation/     # MetricsCollector (benchmarks)
│   └── integrations/chroma_local/  # LocalIndexer + Chroma
├── tests/              # pytest suite
├── docs/               # Documentation
└── indices/            # Built index files
```

---

**Version**: 2.0 — Refactored with Strategy Pattern + BM25
