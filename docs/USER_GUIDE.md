# User Guide

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build a BM25 index
python cli/build.py -x 4 -y 1 -z 1 -optim 0 --limit 100

# 3. Query it
python cli/query.py -x 4 -y 1 -z 1 -q T -optim 0 --query "machine learning"

# 4. Interactive mode
python cli/query.py -x 4 -y 1 -z 1 -q T -optim 0 --interactive
```

## Build Options

```bash
python cli/build.py -x <scorer> -y <storage> -z <compression> -optim <opt> [--limit N] [--force]
```

| Flag | Values | Description |
|------|--------|-------------|
| `-x` | 1, 2, 3, 4 | Boolean, TF, TF-IDF, BM25 |
| `-y` | 1, 2 | JSON, SQLite |
| `-z` | 1, 2, 3 | None, Elias, zlib |
| `-optim` | 0, sp, th, es | None, SkipPointers, Threshold, EarlyStop |
| `--limit` | int | Max documents per source (testing) |
| `--force` | flag | Skip rebuild confirmation |

## Query Options

```bash
python cli/query.py -x <scorer> -y <storage> -z <compression> -q <mode> -optim <opt> [--query "..."] [--interactive] [--top-k N]
```

### Boolean Query Syntax
```
"machine" AND "learning"              # Both terms
"cat" OR "dog"                        # Either term
"data" AND NOT "science"              # Exclude
("cat" OR "dog") AND "pet"            # Grouping
"machine learning"                    # Phrase (exact sequence)
```

### Ranked Query Syntax
Just type natural language — the scorer handles tokenization:
```
machine learning neural networks
```

## All 24 Configurations

| x | y | z | Identifier | Description |
|---|---|---|------------|-------------|
| 1 | 1 | 1 | `SelfIndex_i1d1c1o0` | Boolean + JSON |
| 1 | 1 | 2 | `SelfIndex_i1d1c2o0` | Boolean + JSON + Elias |
| 1 | 1 | 3 | `SelfIndex_i1d1c3o0` | Boolean + JSON + zlib |
| 1 | 2 | 1 | `SelfIndex_i1d2c1o0` | Boolean + SQLite |
| ... | ... | ... | ... | ... |
| 4 | 2 | 3 | `SelfIndex_i4d2c3o0` | BM25 + SQLite + zlib |

## Programmatic Usage

```python
from ire_search.core.engine import SearchEngine

engine = SearchEngine(x=4, y=1, z=1)
engine.load()

response = engine.search("neural networks", top_k=5)
print(f"Found {len(response)} results in {response.elapsed_ms:.1f}ms")
for r in response:
    print(f"  {r.rank}. {r.doc_id}: {r.title}")
```

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Just strategy tests
python -m pytest tests/test_strategies.py -v

# With coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```
