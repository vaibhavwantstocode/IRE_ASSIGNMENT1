"""
Unified SelfIndexer — Single Indexer with Pluggable Strategies

Replaces the previous 15+ indexer classes with one composable class:
    indexer = SelfIndexer(scorer, storage)
    
Or use the factory for backward-compatible construction:
    indexer = create_indexer(x=3, y=1, z=2, optim='0')

The unified indexer delegates:
- Scoring/ranking → ScoringStrategy (Boolean, TF, TF-IDF, BM25)
- Persistence     → StorageBackend (JSON, SQLite)
- Compression     → handled within StorageBackend
"""

import os
from collections import defaultdict
from typing import Dict, List, Iterable, Optional, Any

from ..scoring import ScoringStrategy, get_scorer
from ..storage import StorageBackend, get_storage
from .preprocessor import preprocess_text


# Compression type mapping
COMPRESSION_MAP = {1: 'NONE', 2: 'CODE', 3: 'CLIB'}
OPTIM_MAP = {'0': 'Null', 'sp': 'Skipping', 'th': 'Thresholding', 'es': 'EarlyStopping'}


class SelfIndexer:
    """
    Unified Information Retrieval Indexer.
    
    Composes a ScoringStrategy and StorageBackend to support any
    combination of scoring, storage, and compression.
    
    Usage:
        # Direct composition
        from ire_search.scoring import TFIDFScorer
        from ire_search.storage import JSONStorage
        indexer = SelfIndexer(TFIDFScorer(), JSONStorage(compression_type='CODE'))
        
        # Or via factory
        indexer = create_indexer(x=3, y=1, z=2, optim='0')
    """
    
    def __init__(self, scorer: ScoringStrategy, storage: StorageBackend,
                 optim: str = 'Null'):
        self.scorer = scorer
        self.storage = storage
        self.optim = optim
        
        # Index state
        self.inverted_index = defaultdict(list)
        self.documents = {}
        self.metadata = {}
        self.num_documents = 0
        self.identifier_short = ''
    
    def create_index(self, index_id: str, documents: Iterable[Dict],
                     limit: Optional[int] = None):
        """
        Build an index from a stream of documents.
        
        Args:
            index_id: Identifier for this index (used for save path)
            documents: Iterable of dicts with 'doc_id', 'title', 'content'/'tokens'
            limit: Optional max number of documents to index
        """
        self.identifier_short = index_id
        scorer_type = self.scorer.index_type
        print(f"\n--- Building {scorer_type} index '{index_id}' ---")
        
        doc_count = 0
        for doc in documents:
            if limit and doc_count >= limit:
                break
            
            doc_id = doc['doc_id']
            self.documents[doc_id] = {'title': doc.get('title', 'No Title')}
            
            # Get tokens
            if 'tokens' in doc:
                tokens = doc['tokens']
            else:
                tokens = preprocess_text(doc.get('content', ''))
            
            # Build postings using the scoring strategy
            postings = self.scorer.build_postings(doc_id, tokens)
            for term, posting in postings.items():
                self.inverted_index[term].append(posting)
            
            doc_count += 1
            if doc_count % 10000 == 0:
                print(f"  Processed {doc_count:,} documents...")
        
        self.num_documents = doc_count
        
        # Compute global metadata (IDF, norms, etc.)
        print(f"  Computing metadata for {len(self.inverted_index):,} terms...")
        self.metadata = self.scorer.compute_metadata(
            self.inverted_index, self.num_documents
        )
        
        print(f"  {scorer_type} index complete: {doc_count:,} docs, "
              f"{len(self.inverted_index):,} terms")
        
        # Save to storage
        self._save_index(index_id)
    
    def _save_index(self, index_id: str):
        """Save index using the storage backend."""
        save_data = {
            'inverted_index': dict(self.inverted_index),
            'documents': self.documents,
            'num_documents': self.num_documents,
            'scorer_type': self.scorer.index_type,
        }
        # Merge metadata
        save_data.update(self.metadata)
        
        self.storage.save(index_id, save_data)
    
    def load_index(self, index_id: str):
        """
        Load a previously built index from storage.
        
        Args:
            index_id: Identifier of the index to load
        """
        self.identifier_short = index_id
        print(f"\n--- Loading index '{index_id}' ---")
        
        data = self.storage.load(index_id)
        
        self.inverted_index = data.get('inverted_index', defaultdict(list))
        if not isinstance(self.inverted_index, defaultdict):
            self.inverted_index = defaultdict(list, self.inverted_index)
        
        self.documents = data.get('documents', {})
        self.num_documents = data.get('num_documents', len(self.documents))
        
        # Extract metadata (everything except the core fields)
        core_keys = {'inverted_index', 'documents', 'num_documents',
                     'identifier', 'compression', 'compression_stats',
                     'scorer_type', 'version'}
        self.metadata = {k: v for k, v in data.items() if k not in core_keys}
        
        print(f"  Loaded: {len(self.inverted_index):,} terms, "
              f"{len(self.documents):,} docs")
    
    def query(self, query_str: str, mode: str = 'TAAT', top_k: int = 10) -> List[str]:
        """
        Query the index and return ranked results.
        
        Args:
            query_str: Raw query string
            mode: 'TAAT' or 'DAAT' for ranked queries; ignored for Boolean
            top_k: Number of top results (ignored for Boolean)
            
        Returns:
            List of document IDs ranked by relevance
        """
        if not self.inverted_index:
            print("Warning: Index not loaded.")
            return []
        
        return self.scorer.score_query(
            query_str, self.inverted_index, self.metadata,
            self.documents, mode=mode, top_k=top_k
        )
    
    def update_index(self, index_id: str, remove_docs: Iterable[Dict],
                     add_docs: Iterable[Dict]):
        """
        Incrementally update the index.
        
        Args:
            index_id: Index identifier
            remove_docs: Documents to remove (must have 'doc_id')
            add_docs: Documents to add (must have 'doc_id', 'content'/'tokens')
        """
        # Remove documents
        for doc in remove_docs:
            doc_id = doc['doc_id']
            if doc_id in self.documents:
                del self.documents[doc_id]
                self.num_documents -= 1
                for term in list(self.inverted_index.keys()):
                    self.inverted_index[term] = [
                        p for p in self.inverted_index[term] if p[0] != doc_id
                    ]
                    if not self.inverted_index[term]:
                        del self.inverted_index[term]
        
        # Add new documents
        for doc in add_docs:
            doc_id = doc['doc_id']
            self.documents[doc_id] = {'title': doc.get('title', 'No Title')}
            
            if 'tokens' in doc:
                tokens = doc['tokens']
            else:
                tokens = preprocess_text(doc.get('content', ''))
            
            postings = self.scorer.build_postings(doc_id, tokens)
            for term, posting in postings.items():
                self.inverted_index[term].append(posting)
            self.num_documents += 1
        
        # Recompute metadata
        self.metadata = self.scorer.compute_metadata(
            self.inverted_index, self.num_documents
        )
        
        self._save_index(index_id)
    
    def delete_index(self, index_id: str):
        """Delete an index from storage."""
        self.storage.delete(index_id)
    
    def list_indices(self) -> List[str]:
        """List all available indices."""
        return self.storage.list_indices()
    
    def list_indexed_files(self, index_id: str = None) -> List[str]:
        """List all document IDs in the current index."""
        return list(self.documents.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            'num_documents': self.num_documents,
            'num_terms': len(self.inverted_index),
            'scorer_type': self.scorer.index_type,
            'identifier': self.identifier_short,
        }


# =============================================================================
# Factory Function
# =============================================================================

def create_indexer(x: int, y: int = 1, z: int = 1, optim: str = '0',
                   **kwargs) -> SelfIndexer:
    """
    Create a SelfIndexer with the specified configuration.
    
    This is the recommended way to create indexers — it replaces the
    old 40-line if/elif chain in build.py.
    
    Args:
        x: Index type (1=Boolean, 2=TF, 3=TF-IDF, 4=BM25)
        y: Datastore (1=JSON, 2=SQLite)
        z: Compression (1=None, 2=Elias, 3=Zlib)
        optim: Optimization ('0', 'sp', 'th', 'es')
        **kwargs: Additional scorer args (e.g., k1, b for BM25)
    
    Returns:
        Configured SelfIndexer instance
        
    Examples:
        # Boolean, JSON, no compression
        indexer = create_indexer(x=1, y=1, z=1)
        
        # TF-IDF, SQLite, Elias compression
        indexer = create_indexer(x=3, y=2, z=2)
        
        # BM25 with custom parameters
        indexer = create_indexer(x=4, y=1, z=1, k1=1.5, b=0.8)
    """
    compression_type = COMPRESSION_MAP.get(z, 'NONE')
    optim_name = OPTIM_MAP.get(optim, 'Null')
    
    scorer = get_scorer(x, **kwargs)
    storage = get_storage(y, compression_type=compression_type)
    
    return SelfIndexer(scorer=scorer, storage=storage, optim=optim_name)


def get_index_identifier(x: int, y: int, z: int, optim: str = '0') -> str:
    """
    Generate the standard index identifier string.
    
    Format: SelfIndex_i{x}d{y}c{z}o{optim}
    
    Args:
        x, y, z, optim: Configuration parameters
    
    Returns:
        Identifier string (e.g., 'SelfIndex_i3d1c2o0')
    """
    return f"SelfIndex_i{x}d{y}c{z}o{optim}"
