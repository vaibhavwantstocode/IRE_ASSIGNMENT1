# Compression

Compression is transparent — it's handled inside `StorageBackend.save()` and `StorageBackend.load()`.

## Elias Gamma/Delta (`z=2`)

Source: [compression/elias.py](../src/ire_search/compression/elias.py)

### How It Works
1. **Delta-encode document IDs** — store gaps instead of absolute IDs
2. **Elias Gamma** for small values (TF, small gaps)
3. **Elias Delta** for larger values (positions, large gaps)
4. Pack into bitstring → base64 for JSON storage

### Algorithm
```
Original postings:  [news_1, news_5, news_12]
Doc ID numbers:     [1,      5,      12]
Delta-encoded gaps: [1,      4,      7]
Elias Gamma(1):     "1"
Elias Gamma(4):     "00100"
Elias Gamma(7):     "00111"
```

### When to Use
Best for indices where most values are small (typical IR workloads).

## zlib (`z=3`)

Uses Python's built-in `zlib` library for general-purpose compression.

### How It Works
1. Serialize index data to JSON string
2. Compress with `zlib.compress(data, level=6)`
3. Store as binary blob

### Compression Ratios
| Data Type | Typical Ratio |
|-----------|---------------|
| Raw JSON → zlib | 3-5× smaller |
| Elias | 2-4× smaller (posting-specific) |

## Comparison

| Feature | None (`z=1`) | Elias (`z=2`) | zlib (`z=3`) |
|---------|-------------|---------------|-------------|
| Speed (build) | ⚡ Fastest | Moderate | Fast |
| Speed (load) | ⚡ Fastest | Moderate | Fast |
| Size reduction | — | 2-4× | 3-5× |
| Best for | Dev/testing | Production | Archives |
