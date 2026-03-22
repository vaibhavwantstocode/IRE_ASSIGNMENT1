# Scoring Strategies

All scorers implement the `ScoringStrategy` ABC in [scoring/strategies.py](../src/ire_search/scoring/strategies.py).

## 1. Boolean Scorer (`x=1`)

**Pure set-based retrieval** — documents match or they don't; no ranking.

### Operators
| Operator | Meaning | Example |
|----------|---------|---------|
| `AND` | Intersection | `"cat" AND "dog"` |
| `OR` | Union | `"cat" OR "dog"` |
| `NOT` | Complement | `"cat" AND NOT "fish"` |
| `"..."` | Phrase | `"machine learning"` |
| `()` | Grouping | `("cat" OR "dog") AND NOT "fish"` |

### Query Parsing
Uses the **Shunting-yard algorithm** to convert infix boolean expressions to Reverse Polish Notation (RPN), then evaluates with a stack:

```
Input:  "cat" AND ("dog" OR "fish")
Tokens: [cat, AND, (, dog, OR, fish, )]
RPN:    [cat, dog, fish, OR, AND]
```

## 2. TF Scorer (`x=2`)

**Term Frequency** — ranks by how often query terms appear, normalized by document length.

$$\text{score}(q, d) = \sum_{t \in q} \frac{\text{tf}(t, d)}{\|d\|}$$

Where `‖d‖` is the L2 norm of the TF vector: $\|d\| = \sqrt{\sum_t \text{tf}(t, d)^2}$

### Postings Format
```python
[doc_id, tf, [pos1, pos2, ...]]
# Example: ['news_42', 3, [10, 25, 89]]
```

## 3. TF-IDF Scorer (`x=3`)

**Term Frequency × Inverse Document Frequency** — upweights rare terms.

$$\text{score}(q, d) = \sum_{t \in q} \frac{\text{tf}(t, d) \times \text{idf}(t)}{\|d\|_{\text{tfidf}}}$$

$$\text{idf}(t) = \log\left(\frac{N}{\text{df}(t)}\right)$$

Where:
- `N` = total documents
- `df(t)` = number of documents containing term `t`
- `‖d‖_tfidf` = L2 norm of the TF-IDF vector

## 4. BM25 Scorer (`x=4`)

**Okapi BM25** — the industry-standard probabilistic ranking function.

$$\text{score}(q, d) = \sum_{t \in q} \text{idf}(t) \cdot \frac{\text{tf}(t, d) \cdot (k_1 + 1)}{\text{tf}(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

### Parameters
| Parameter | Default | Effect |
|-----------|---------|--------|
| `k1` | 1.2 | TF saturation — higher = more weight on frequency |
| `b` | 0.75 | Length normalization — 0 = none, 1 = full |

### Key Properties
- **TF saturation**: Unlike raw TF, frequent terms hit diminishing returns
- **Length normalization**: Long documents are penalized for containing more terms by chance
- **No IDF for single-doc terms**: `idf = log((N - df + 0.5) / (df + 0.5) + 1)`

### Postings Format (BM25-specific)
```python
[doc_id, tf, doc_length, [pos1, pos2, ...]]
# Example: ['news_42', 3, 157, [10, 25, 89]]
```

## Comparison

| Scorer | Use Case | Speed | Quality |
|--------|----------|-------|---------|
| Boolean | Exact filtering, facets | ⚡ Fastest | No ranking |
| TF | Simple frequency ranking | ⚡ Fast | Basic |
| TF-IDF | Academic IR, balanced | ⚡ Fast | Good |
| BM25 | Production search engines | ⚡ Fast | **Best** |

## Usage

```python
from ire_search import create_indexer

# BM25 with custom parameters
indexer = create_indexer(x=4, y=1, z=1, k1=1.5, b=0.8)
```
