"""
IRE Search Engine — Information Retrieval with Pluggable Strategies

Primary API:
    from ire_search import SelfIndexer, create_indexer

    indexer = create_indexer(x=3, y=1, z=1)  # TF-IDF + JSON
    indexer.create_index('my_index', documents)
    results = indexer.query('machine learning')

Strategy classes:
    from ire_search.scoring import BooleanScorer, TFScorer, TFIDFScorer, BM25Scorer
    from ire_search.storage import JSONStorage, SQLiteStorage
"""

from .core.indexer import SelfIndexer, create_indexer, get_index_identifier
from .scoring import (
    ScoringStrategy,
    BooleanScorer,
    TFScorer,
    TFIDFScorer,
    BM25Scorer,
    get_scorer,
)
from .storage import StorageBackend, JSONStorage, SQLiteStorage, get_storage

__all__ = [
    "SelfIndexer",
    "create_indexer",
    "get_index_identifier",
    "ScoringStrategy",
    "BooleanScorer",
    "TFScorer",
    "TFIDFScorer",
    "BM25Scorer",
    "StorageBackend",
    "JSONStorage",
    "SQLiteStorage",
    "get_scorer",
    "get_storage",
]
