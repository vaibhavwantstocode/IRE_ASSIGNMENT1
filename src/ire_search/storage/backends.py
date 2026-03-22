"""
Storage Backends for Information Retrieval Index Persistence

Implements the Strategy Pattern for different storage approaches:
- JSONStorage: File-based JSON storage with optional compression
- SQLiteStorage: SQLite database storage with optional compression

Each backend handles save/load/delete/list operations for index data.
Compression is applied transparently during save and reversed during load.
"""

import json
import os
import sqlite3
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Any, Optional, List

from ..compression.elias import EliasCompressor
from ..compression.zlib_compressor import ZlibCompressor


# =============================================================================
# Compression Helpers
# =============================================================================

COMPRESSORS = {
    'NONE': None,
    'CODE': EliasCompressor,   # Elias Gamma/Delta (z=2)
    'CLIB': ZlibCompressor,    # zlib (z=3)
}


def get_compressor(compression_type: str):
    """Get a compressor by type code."""
    if compression_type not in COMPRESSORS:
        raise ValueError(f"Unknown compression: {compression_type}. Valid: {list(COMPRESSORS.keys())}")
    return COMPRESSORS[compression_type]


# =============================================================================
# Abstract Base Class
# =============================================================================

class StorageBackend(ABC):
    """
    Abstract base class for index storage backends.
    
    A storage backend defines how index data is persisted to and loaded from disk.
    Compression is handled within the storage layer, transparent to the indexer.
    """
    
    @abstractmethod
    def save(self, identifier: str, index_data: Dict[str, Any]) -> str:
        """
        Save index data to persistent storage.
        
        Args:
            identifier: Short identifier (e.g., 'SelfIndex_i1d1c1o0')
            index_data: Dict containing inverted_index, documents, metadata
            
        Returns:
            Path to the saved index file
        """
        ...
    
    @abstractmethod
    def load(self, identifier: str) -> Dict[str, Any]:
        """
        Load index data from persistent storage.
        
        Args:
            identifier: Short identifier to load
            
        Returns:
            Dict containing inverted_index, documents, metadata
        """
        ...
    
    @abstractmethod
    def delete(self, identifier: str) -> bool:
        """Delete an index. Returns True if deleted, False if not found."""
        ...
    
    @abstractmethod
    def list_indices(self) -> List[str]:
        """List all available index identifiers."""
        ...
    
    @abstractmethod
    def exists(self, identifier: str) -> bool:
        """Check if an index exists."""
        ...


# =============================================================================
# JSON Storage — y=1
# =============================================================================

class JSONStorage(StorageBackend):
    """
    File-based JSON storage backend.
    
    Saves indices as JSON files in the `indices/` directory.
    Supports optional compression of the inverted index during save.
    
    File format:
        indices/{identifier}.json
    """
    
    def __init__(self, compression_type: str = 'NONE', indices_dir: str = 'indices'):
        self.compression_type = compression_type
        self.compressor = get_compressor(compression_type)
        self.indices_dir = indices_dir
    
    def save(self, identifier: str, index_data: Dict[str, Any]) -> str:
        """Save index as JSON file with optional compression."""
        os.makedirs(self.indices_dir, exist_ok=True)
        filepath = os.path.join(self.indices_dir, f"{identifier}.json")
        
        # Prepare data for saving
        save_data = dict(index_data)
        save_data['identifier'] = identifier
        save_data['compression'] = self.compression_type
        
        # Compress inverted index if needed
        if self.compressor and 'inverted_index' in save_data:
            raw_index = save_data['inverted_index']
            if isinstance(raw_index, defaultdict):
                raw_index = dict(raw_index)
            
            print(f"  Compressing inverted index with {self.compression_type}...")
            compressed = self.compressor.compress_inverted_index(raw_index)
            
            num_terms = len(raw_index)
            total_postings = sum(len(v) for v in raw_index.values())
            save_data['inverted_index'] = compressed
            save_data['compression_stats'] = {
                'num_terms': num_terms,
                'total_postings': total_postings,
                'compressed_terms': len(compressed),
            }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f)
        
        file_size = os.path.getsize(filepath)
        print(f"  Index saved to {filepath} ({file_size / (1024*1024):.2f} MB)")
        return filepath
    
    def load(self, identifier: str) -> Dict[str, Any]:
        """Load index from JSON file with automatic decompression."""
        filepath = os.path.join(self.indices_dir, f"{identifier}.json")
        if not os.path.exists(filepath):
            filepath = f"{identifier}.json"
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Index not found: {filepath}")
        
        print(f"  Loading index from {filepath}...")
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Detect and apply decompression
        compression = data.get('compression', 'NONE')
        if compression != 'NONE' and compression in COMPRESSORS:
            compressor = get_compressor(compression)
            if compressor and 'inverted_index' in data:
                print(f"  Decompressing with {compression}...")
                data['inverted_index'] = defaultdict(
                    list,
                    compressor.decompress_inverted_index(data['inverted_index'])
                )
        else:
            if 'inverted_index' in data:
                data['inverted_index'] = defaultdict(list, data['inverted_index'])
        
        return data
    
    def delete(self, identifier: str) -> bool:
        """Delete a JSON index file."""
        filepath = os.path.join(self.indices_dir, f"{identifier}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"  Deleted index: {filepath}")
            return True
        return False
    
    def list_indices(self) -> List[str]:
        """List all JSON index files."""
        if not os.path.exists(self.indices_dir):
            return []
        return [
            f[:-5] for f in os.listdir(self.indices_dir)
            if f.startswith('SelfIndex_') and f.endswith('.json')
        ]
    
    def exists(self, identifier: str) -> bool:
        """Check if a JSON index file exists."""
        filepath = os.path.join(self.indices_dir, f"{identifier}.json")
        return os.path.exists(filepath)


# =============================================================================
# SQLite Storage — y=2
# =============================================================================

class SQLiteStorage(StorageBackend):
    """
    SQLite database storage backend.
    
    Stores index data in SQLite databases with tables for:
    - terms: inverted index (term -> compressed postings blob)
    - documents: document metadata
    - metadata: global index metadata (idf_scores, doc_norms, etc.)
    
    File format:
        indices/{identifier}.db
    """
    
    def __init__(self, compression_type: str = 'NONE', indices_dir: str = 'indices'):
        self.compression_type = compression_type
        self.compressor = get_compressor(compression_type)
        self.indices_dir = indices_dir
    
    def save(self, identifier: str, index_data: Dict[str, Any]) -> str:
        """Save index to SQLite database."""
        os.makedirs(self.indices_dir, exist_ok=True)
        db_path = os.path.join(self.indices_dir, f"{identifier}.db")
        
        # Remove existing DB to rebuild cleanly
        if os.path.exists(db_path):
            os.remove(db_path)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            # Create tables
            cursor.execute('''CREATE TABLE terms (
                term TEXT PRIMARY KEY,
                postings TEXT NOT NULL
            )''')
            cursor.execute('''CREATE TABLE documents (
                doc_id TEXT PRIMARY KEY,
                metadata TEXT NOT NULL
            )''')
            cursor.execute('''CREATE TABLE index_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )''')
            
            # Save inverted index
            inverted_index = index_data.get('inverted_index', {})
            if isinstance(inverted_index, defaultdict):
                inverted_index = dict(inverted_index)
            
            if self.compressor:
                print(f"  Compressing index with {self.compression_type}...")
                compressed = self.compressor.compress_inverted_index(inverted_index)
                for term, posting_data in compressed.items():
                    cursor.execute(
                        'INSERT INTO terms (term, postings) VALUES (?, ?)',
                        (term, json.dumps(posting_data))
                    )
            else:
                for term, postings in inverted_index.items():
                    cursor.execute(
                        'INSERT INTO terms (term, postings) VALUES (?, ?)',
                        (term, json.dumps(postings))
                    )
            
            # Save documents
            documents = index_data.get('documents', {})
            for doc_id, doc_meta in documents.items():
                cursor.execute(
                    'INSERT INTO documents (doc_id, metadata) VALUES (?, ?)',
                    (str(doc_id), json.dumps(doc_meta))
                )
            
            # Save metadata
            metadata_keys = ['idf_scores', 'doc_norms', 'doc_lengths',
                             'avg_doc_length', 'num_documents', 'k1', 'b']
            for key in metadata_keys:
                if key in index_data:
                    cursor.execute(
                        'INSERT INTO index_metadata (key, value) VALUES (?, ?)',
                        (key, json.dumps(index_data[key]))
                    )
            
            cursor.execute(
                'INSERT INTO index_metadata (key, value) VALUES (?, ?)',
                ('compression', json.dumps(self.compression_type))
            )
            cursor.execute(
                'INSERT INTO index_metadata (key, value) VALUES (?, ?)',
                ('identifier', json.dumps(identifier))
            )
            
            conn.commit()
            print(f"  Index saved to {db_path}")
            
        finally:
            conn.close()
        
        file_size = os.path.getsize(db_path)
        print(f"  Database size: {file_size / (1024*1024):.2f} MB")
        return db_path
    
    def load(self, identifier: str) -> Dict[str, Any]:
        """Load index from SQLite database."""
        db_path = os.path.join(self.indices_dir, f"{identifier}.db")
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Index not found: {db_path}")
        
        print(f"  Loading index from {db_path}...")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            # Load metadata first to determine compression
            metadata = {}
            cursor.execute('SELECT key, value FROM index_metadata')
            for key, value in cursor.fetchall():
                metadata[key] = json.loads(value)
            
            compression = metadata.get('compression', 'NONE')
            
            # Load inverted index
            cursor.execute('SELECT term, postings FROM terms')
            raw_index = {term: json.loads(postings) for term, postings in cursor.fetchall()}
            
            if compression != 'NONE' and compression in COMPRESSORS:
                compressor = get_compressor(compression)
                if compressor:
                    print(f"  Decompressing with {compression}...")
                    raw_index = compressor.decompress_inverted_index(raw_index)
            
            metadata['inverted_index'] = defaultdict(list, raw_index)
            
            # Load documents
            cursor.execute('SELECT doc_id, metadata FROM documents')
            metadata['documents'] = {
                doc_id: json.loads(meta) for doc_id, meta in cursor.fetchall()
            }
            
            return metadata
            
        finally:
            conn.close()
    
    def delete(self, identifier: str) -> bool:
        """Delete a SQLite index database."""
        db_path = os.path.join(self.indices_dir, f"{identifier}.db")
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"  Deleted index: {db_path}")
            return True
        return False
    
    def list_indices(self) -> List[str]:
        """List all SQLite index databases."""
        if not os.path.exists(self.indices_dir):
            return []
        return [
            f[:-3] for f in os.listdir(self.indices_dir)
            if f.startswith('SelfIndex_') and f.endswith('.db')
        ]
    
    def exists(self, identifier: str) -> bool:
        """Check if a SQLite index database exists."""
        db_path = os.path.join(self.indices_dir, f"{identifier}.db")
        return os.path.exists(db_path)


# =============================================================================
# Factory
# =============================================================================

STORAGE_BACKENDS = {
    1: JSONStorage,
    2: SQLiteStorage,
}


def get_storage(y: int, compression_type: str = 'NONE', **kwargs) -> StorageBackend:
    """
    Factory function to get a storage backend by datastore number.
    
    Args:
        y: Datastore type (1=JSON, 2=SQLite)
        compression_type: 'NONE', 'CODE', or 'CLIB'
        **kwargs: Additional backend args (e.g., indices_dir)
    
    Returns:
        StorageBackend instance
    """
    if y not in STORAGE_BACKENDS:
        raise ValueError(f"Unknown datastore: {y}. Valid: {list(STORAGE_BACKENDS.keys())}")
    return STORAGE_BACKENDS[y](compression_type=compression_type, **kwargs)
