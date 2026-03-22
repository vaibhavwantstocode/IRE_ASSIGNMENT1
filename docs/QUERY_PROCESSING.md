# Query Processing

## Query Modes

### TAAT (Term-at-a-Time)
Processes one query term at a time, accumulating scores across all documents.

```
Query: "machine learning"
Step 1: Get all docs for "machine"  → {d1: 0.3, d2: 0.5, d4: 0.2}
Step 2: Get all docs for "learning" → {d1: 0.4, d2: 0.1, d4: 0.6}
Step 3: Sum scores                  → {d1: 0.7, d4: 0.8, d2: 0.6}
```

**Pros**: Simple, cache-friendly for posting list access
**Cons**: Must process all postings

### DAAT (Document-at-a-Time)
Processes one document at a time using synchronized cursors.

**Pros**: Can prune early (skip low-scoring docs)
**Cons**: Requires simultaneous iteration of posting lists

## Boolean Query Parsing

The Boolean scorer uses the **Shunting-yard algorithm** (Dijkstra, 1961) to parse expressions.

### Pipeline
```
Input string → Tokenizer → Shunting-yard → RPN → Stack evaluator → Doc set
```

### Example
```
Input:  ("cat" OR "dog") AND NOT "fish"
Tokens: [(, cat, OR, dog, ), AND, NOT, fish]
RPN:    [cat, dog, OR, fish, NOT, AND]
```

### Operator Precedence
| Priority | Operator | Type |
|----------|----------|------|
| 3 | `NOT` | Unary |
| 2 | `AND` | Binary |
| 1 | `OR` | Binary |

## Phrase Matching

Phrase queries (quoted strings) use **positional intersection**:

```
Query: "machine learning"
Step 1: Get positions of "machine"  in doc → [0, 5, 12]
Step 2: Get positions of "learning" in doc → [1, 8, 13]
Step 3: Check if any position[learning] == position[machine] + 1
         → 0+1=1 ✓ (match at positions 0,1)
         → 12+1=13 ✓ (match at positions 12,13)
```

## Skip Pointers

Optimization for AND queries. Skip pointers allow jumping over large sections of posting lists when looking for common documents.

```
Postings for "cat":  [1, 3, 5, 8, 12, 15, 20, ...]
Skip pointers:       [5 →, 15 →, ...]  (every √n entries)

When intersecting with "dog" postings at doc_id=10:
  Instead of scanning 1,3,5,8,12... we skip to 15
```
