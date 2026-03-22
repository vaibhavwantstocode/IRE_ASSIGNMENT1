"""
SearchEngine — High-level Orchestrator for the IR System

Provides a single entry point for building, querying, and managing indices.
Designed for future extensibility (RAG, vector search, hybrid ranking).

Usage:
    engine = SearchEngine()
    
    # Build an index
    engine.build(x=3, y=1, z=1, documents=docs)
    
    # Search
    results = engine.search('machine learning', top_k=10)
    
    # Compare scorers
    comparison = engine.compare_scorers('neural networks', scorers=[1,2,3,4])
"""

import time
from typing import Dict, List, Optional, Any, Iterable

from .indexer import SelfIndexer, create_indexer, get_index_identifier


class SearchResult:
    """A single search result with metadata."""
    
    __slots__ = ('doc_id', 'score', 'rank', 'title')
    
    def __init__(self, doc_id: str, score: float = 0.0, rank: int = 0,
                 title: str = ''):
        self.doc_id = doc_id
        self.score = score
        self.rank = rank
        self.title = title
    
    def __repr__(self):
        return f"SearchResult(rank={self.rank}, doc_id='{self.doc_id}', score={self.score:.4f})"
    
    def to_dict(self) -> dict:
        return {
            'doc_id': self.doc_id, 'score': self.score,
            'rank': self.rank, 'title': self.title,
        }


class SearchResponse:
    """Container for search results with timing and metadata."""
    
    def __init__(self, query: str, results: List[SearchResult],
                 elapsed_ms: float, scorer_type: str, total_docs: int = 0):
        self.query = query
        self.results = results
        self.elapsed_ms = elapsed_ms
        self.scorer_type = scorer_type
        self.total_docs = total_docs
    
    def __repr__(self):
        return (f"SearchResponse(query='{self.query}', "
                f"results={len(self.results)}, "
                f"time={self.elapsed_ms:.2f}ms)")
    
    def __len__(self):
        return len(self.results)
    
    def __iter__(self):
        return iter(self.results)
    
    def to_dict(self) -> dict:
        return {
            'query': self.query,
            'results': [r.to_dict() for r in self.results],
            'elapsed_ms': self.elapsed_ms,
            'scorer_type': self.scorer_type,
            'total_docs': self.total_docs,
        }


class SearchEngine:
    """
    High-level search engine orchestrating indexing and retrieval.
    
    This is the recommended entry point for applications integrating
    the IR system. It manages index lifecycle and provides a clean
    search API.
    
    Future extensions:
    - Hybrid search (BM25 + vector similarity)
    - RAG pipeline integration
    - Query expansion/suggestion
    - Result re-ranking
    - Caching layer
    
    Example:
        engine = SearchEngine(x=3, y=1, z=1)
        engine.build(documents)
        response = engine.search('machine learning')
        for result in response:
            print(f"{result.rank}. {result.doc_id} ({result.score:.4f})")
    """
    
    def __init__(self, x: int = 3, y: int = 1, z: int = 1,
                 optim: str = '0', **scorer_kwargs):
        """
        Initialize the search engine.
        
        Args:
            x: Scorer type (1=Boolean, 2=TF, 3=TF-IDF, 4=BM25)
            y: Storage backend (1=JSON, 2=SQLite)
            z: Compression (1=None, 2=Elias, 3=zlib)
            optim: Optimization strategy
            **scorer_kwargs: Extra args for scorer (e.g., k1, b for BM25)
        """
        self.x = x
        self.y = y
        self.z = z
        self.optim = optim
        self.scorer_kwargs = scorer_kwargs
        
        self._indexer: Optional[SelfIndexer] = None
        self._identifier: Optional[str] = None
        self._is_loaded = False
    
    @property
    def identifier(self) -> str:
        """Get the current index identifier."""
        if self._identifier is None:
            self._identifier = get_index_identifier(self.x, self.y, self.z, self.optim)
        return self._identifier
    
    @property
    def indexer(self) -> SelfIndexer:
        """Get or create the underlying indexer."""
        if self._indexer is None:
            self._indexer = create_indexer(
                x=self.x, y=self.y, z=self.z,
                optim=self.optim, **self.scorer_kwargs
            )
        return self._indexer
    
    def build(self, documents: Iterable[Dict], index_id: Optional[str] = None,
              limit: Optional[int] = None):
        """
        Build an index from documents.
        
        Args:
            documents: Iterable of dicts with 'doc_id', 'title', 'content'/'tokens'
            index_id: Custom identifier (defaults to auto-generated)
            limit: Max documents to index
        """
        ident = index_id or self.identifier
        self.indexer.create_index(ident, documents, limit=limit)
        self._identifier = ident
        self._is_loaded = True
    
    def load(self, index_id: Optional[str] = None):
        """
        Load a previously built index.
        
        Args:
            index_id: Identifier of the index (defaults to auto-generated)
        """
        ident = index_id or self.identifier
        self.indexer.load_index(ident)
        self._identifier = ident
        self._is_loaded = True
    
    def search(self, query: str, top_k: int = 10,
               mode: str = 'TAAT') -> SearchResponse:
        """
        Search the index and return structured results.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            mode: 'TAAT' or 'DAAT'
            
        Returns:
            SearchResponse with ranked results and timing
        """
        if not self._is_loaded:
            self.load()
        
        start = time.perf_counter()
        raw_results = self.indexer.query(query, mode=mode, top_k=top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Wrap raw results in SearchResult objects
        results = []
        for rank, doc_id in enumerate(raw_results, 1):
            title = self.indexer.documents.get(doc_id, {}).get('title', '')
            results.append(SearchResult(
                doc_id=doc_id, score=0.0, rank=rank, title=title
            ))
        
        return SearchResponse(
            query=query, results=results, elapsed_ms=elapsed_ms,
            scorer_type=self.indexer.scorer.index_type,
            total_docs=self.indexer.num_documents,
        )
    
    def compare_scorers(self, query: str, scorer_types: List[int] = None,
                        top_k: int = 10) -> Dict[str, SearchResponse]:
        """
        Run the same query with multiple scorers for comparison.
        
        Args:
            query: Search query string
            scorer_types: List of scorer type codes (default: [1,2,3,4])
            top_k: Results per scorer
            
        Returns:
            Dict mapping scorer name → SearchResponse
        """
        if scorer_types is None:
            scorer_types = [2, 3, 4]  # Skip Boolean for free-text queries
        
        comparison = {}
        for x in scorer_types:
            engine = SearchEngine(x=x, y=self.y, z=self.z, optim=self.optim)
            try:
                engine.load()
                response = engine.search(query, top_k=top_k)
                comparison[response.scorer_type] = response
            except FileNotFoundError:
                print(f"  Skipping x={x} (index not built)")
        
        return comparison
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if not self._is_loaded:
            self.load()
        stats = self.indexer.get_stats()
        stats['identifier'] = self.identifier
        return stats
    
    def __repr__(self):
        status = 'loaded' if self._is_loaded else 'not loaded'
        return f"SearchEngine(x={self.x}, y={self.y}, z={self.z}, {status})"
