# Data Integration Guide

## Document Format

The indexer expects documents as dictionaries:

```python
{
    'doc_id': 'unique_id',           # Required — unique identifier
    'title': 'Document Title',       # Optional — stored for display
    'content': 'Full text here...',  # Option A — raw text (preprocessed automatically)
    'tokens': ['token1', 'token2'],  # Option B — pre-tokenized (faster)
}
```

> [!IMPORTANT]
> If using Elias compression (`z=2`), doc_ids must follow the format `news_X` or `wiki_Y` where X/Y are integers.

## Adding Custom Data

### From CSV/TSV

```python
import csv
from ire_search import create_indexer

def load_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            yield {
                'doc_id': f'csv_{i}',
                'title': row.get('title', ''),
                'content': row.get('text', row.get('body', '')),
            }

indexer = create_indexer(x=4, y=1, z=1)
indexer.create_index('my_csv_index', load_csv('data.csv'))
```

### From JSON Lines

```python
import json

def load_jsonl(path):
    with open(path, 'r') as f:
        for line in f:
            doc = json.loads(line)
            yield {
                'doc_id': doc['id'],
                'title': doc.get('title', ''),
                'content': doc['text'],
            }
```

### From a Database

```python
import sqlite3

def load_from_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT id, title, content FROM documents")
    for row in cursor:
        yield {'doc_id': str(row[0]), 'title': row[1], 'content': row[2]}
    conn.close()
```

## Preprocessing

Text preprocessing is handled by [core/preprocessor.py](../src/ire_search/core/preprocessor.py):

1. **Lowercasing**
2. **Tokenization** (split on non-alphanumeric)
3. **Stopword removal** (NLTK English stopwords)
4. **Stemming** (Porter Stemmer)

To skip preprocessing, provide pre-tokenized `tokens` instead of `content`.

## Using Preprocessed Cache

For large corpora, preprocess once and cache:

```bash
python scripts/preprocess_corpus.py --limit 50000
```

This creates a Parquet cache that `build.py` loads automatically.
