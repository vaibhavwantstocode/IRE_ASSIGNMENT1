# API Reference

## Core API

### `create_indexer(x, y, z, optim, **kwargs)`
Factory function — the primary way to create indexers.

```python
from ire_search import create_indexer

indexer = create_indexer(x=3, y=1, z=1, optim='0')
indexer = create_indexer(x=4, y=1, z=1, k1=1.5, b=0.8)  # BM25 with custom params
```

| Parameter | Type | Values | Default |
|-----------|------|--------|---------|
| `x` | int | 1=Boolean, 2=TF, 3=TF-IDF, 4=BM25 | required |
| `y` | int | 1=JSON, 2=SQLite | required |
| `z` | int | 1=None, 2=Elias, 3=zlib | required |
| `optim` | str | '0', 'sp', 'th', 'es' | '0' |
| `**kwargs` | — | BM25: `k1`, `b` | — |

---

### `SelfIndexer`

The unified indexer class.

#### `create_index(index_id, documents, limit=None)`
Build an index from documents.

```python
docs = [{'doc_id': 'news_1', 'title': 'Title', 'content': 'text here'}]
indexer.create_index('my_index', docs)
```

#### `load_index(index_id)`
Load a previously built index.

#### `query(query_str, mode='TAAT', top_k=10)`
Query the index. Returns `List[str]` of document IDs.

#### `update_index(index_id, remove_docs, add_docs)`
Incrementally add/remove documents.

#### `get_stats()`
Returns dict with `num_documents`, `num_terms`, `scorer_type`, `identifier`.

---

### `SearchEngine`

High-level orchestrator.

```python
from ire_search.core.engine import SearchEngine

engine = SearchEngine(x=4, y=1, z=1)
engine.build(documents)
response = engine.search('machine learning', top_k=10)

for result in response:
    print(f"{result.rank}. {result.doc_id} — {result.title}")
```

#### `build(documents, index_id=None, limit=None)`
Build and save an index.

#### `load(index_id=None)`
Load an existing index.

#### `search(query, top_k=10, mode='TAAT')`
Returns `SearchResponse` with `results`, `elapsed_ms`, `scorer_type`.

#### `compare_scorers(query, scorer_types=None, top_k=10)`
Run the same query with multiple scorers. Returns `Dict[str, SearchResponse]`.

---

## Scoring Strategies

All implement `ScoringStrategy` ABC:

| Method | Signature | Returns |
|--------|-----------|---------|
| `build_postings` | `(doc_id, tokens) → Dict[str, list]` | Term → posting |
| `compute_metadata` | `(index, num_docs) → Dict` | IDF/norms/etc. |
| `score_query` | `(query, index, metadata, docs, mode, top_k) → List[str]` | Ranked doc_ids |

## Storage Backends

All implement `StorageBackend` ABC:

| Method | Signature |
|--------|-----------|
| `save(index_id, data)` | Save index data |
| `load(index_id) → dict` | Load index data |
| `delete(index_id)` | Delete index |
| `list_indices() → List[str]` | List available indices |
