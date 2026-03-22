# Extending the System

The Strategy Pattern makes adding new components straightforward.

## Adding a New Scorer

1. **Create a class** implementing `ScoringStrategy`:

```python
# src/ire_search/scoring/strategies.py
class MyScorer(ScoringStrategy):
    index_type = 'CUSTOM'
    
    def build_postings(self, doc_id, tokens):
        postings = {}
        for i, token in enumerate(tokens):
            if token not in postings:
                postings[token] = [doc_id, 0, []]
            postings[token][1] += 1
            postings[token][2].append(i)
        return postings
    
    def compute_metadata(self, inverted_index, num_documents):
        return {}  # Your global stats here
    
    def score_query(self, query_str, inverted_index, metadata,
                    documents, mode='TAAT', top_k=10):
        # Your scoring logic here
        return ranked_doc_ids
```

2. **Register in factory**:

```python
# strategies.py → get_scorer()
def get_scorer(x, **kwargs):
    ...
    elif x == 5:
        return MyScorer(**kwargs)
```

3. **Use it**:
```python
indexer = create_indexer(x=5, y=1, z=1)
```

## Adding a New Storage Backend

1. Implement `StorageBackend`:

```python
class RedisStorage(StorageBackend):
    def save(self, index_id, data):
        ...
    def load(self, index_id):
        ...
    def delete(self, index_id):
        ...
    def list_indices(self):
        ...
```

2. Register in `get_storage()`.

## Adding a New Compression Method

Compression is handled within `StorageBackend`, so:
1. Create a compressor class (like `EliasCompressor`)
2. Add it as an option in `JSONStorage.__init__`

## RAG Integration Points

The `SearchEngine` class is designed for future RAG pipelines:

```python
from ire_search.core.engine import SearchEngine

class RAGEngine(SearchEngine):
    def __init__(self, llm_client, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm_client
    
    def retrieve_and_generate(self, query, top_k=5):
        response = self.search(query, top_k=top_k)
        context = "\n".join(r.title for r in response)
        return self.llm.generate(f"Context: {context}\nQuery: {query}")
```
