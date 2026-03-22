# Storage Backends

All backends implement the `StorageBackend` ABC from [storage/backends.py](../src/ire_search/storage/backends.py).

## JSON Storage (`y=1`)

Stores indices as `.json` files in the `indices/` directory.

**Pros**: Human-readable, easy debugging, fast for small indices
**Cons**: Loads entire index into RAM, large file sizes

### Compression Support
| `z` | Method | How It Works |
|-----|--------|-------------|
| 1 | None | Raw JSON |
| 2 | Elias γ/δ | Posting lists encoded as base64 bitstrings |
| 3 | zlib | Entire JSON compressed with zlib |

### File Format
```json
{
  "identifier": "SelfIndex_i3d1c1o0",
  "inverted_index": { "term": [[doc_id, tf, [positions]], ...] },
  "documents": { "doc_id": {"title": "..."} },
  "num_documents": 50000,
  "idf_scores": { "term": 3.14 },
  "doc_norms": { "doc_id": 12.5 }
}
```

## SQLite Storage (`y=2`)

Stores indices as `.db` files using SQLite.

**Pros**: Disk-backed (low RAM), supports large indices, concurrent-safe
**Cons**: Slower for full-index scans, binary format

### Schema
```sql
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE postings (term TEXT, data BLOB);
CREATE TABLE documents (doc_id TEXT PRIMARY KEY, title TEXT);
CREATE INDEX idx_postings_term ON postings(term);
```

## Choosing a Backend

| Scenario | Recommended |
|----------|------------|
| Development/debugging | JSON (`y=1`) |
| Large corpus (>100K docs) | SQLite (`y=2`) |
| Low-memory server | SQLite (`y=2`) + zlib (`z=3`) |
| Maximum speed | JSON (`y=1`) + no compression (`z=1`) |
