"""
SelfIndexer_x3: TF-IDF Ranking Indexer with Cosine Similarity

Implements x=3 from the assignment requirements:
- TF-IDF scoring for better relevance ranking
- Document Length Normalization (Cosine Similarity)
- "Virtual Term" Phrase Boosting with Information-Theoretic weighting
- Supports compression (z=1: none, z=2: Elias, z=3: Zlib)
- Supports both datastores (y=1: JSON, y=2: SQLite via compressed_indexer)
- Implements TAAT and DAAT query processing

Index structure:
{
  "term": [
    [doc_id, tf, [positions]],
    [doc_id, tf, [positions]],
    ...
  ],
  "idf_scores": {term: idf_value, ...},
  "doc_norms": {doc_id: norm_value, ...}
}
"""

import json
import os
import math
import re
from collections import defaultdict
from .index_base import IndexBase
from .preprocessor import preprocess_text
from typing import Iterable, Dict, List


class SelfIndexer_x3(IndexBase):
    """
    TF-IDF ranking indexer with Cosine Similarity
    
    Assignment requirement: x=3 - Evaluate gains from TF-IDF scores
    Implements Standard Vector Space Model
    """
    
    def __init__(self, dstore='CUSTOM', compr='NONE', optim='Null'):
        super().__init__(core='SelfIndex', info='TFIDF', dstore=dstore, 
                         qproc='TERMatat', compr=compr, optim=optim)
        self.inverted_index = defaultdict(list)
        self.documents = {}
        self.idf_scores = {}
        self.doc_norms = {}
        self.num_documents = 0
        self.compression_type = compr
        self.optim = optim
        
    def create_index(self, index_id: str, documents: Iterable[Dict]):
        print(f"--- Building TF-IDF index '{self.identifier_short}' ---")
        
        # First pass: build TF index
        doc_count = 0
        for doc in documents:
            doc_id = doc['doc_id']
            self.documents[doc_id] = {'title': doc.get('title', 'No Title')}
            
            # Get preprocessed tokens
            if 'tokens' in doc:
                tokens = doc['tokens']
            else:
                content = doc.get('content', '')
                tokens = preprocess_text(content)
            
            # Count term frequencies and track positions
            term_data = defaultdict(lambda: {'count': 0, 'positions': []})
            for i, token in enumerate(tokens):
                term_data[token]['count'] += 1
                term_data[token]['positions'].append(i)
            
            # Build inverted index with TF scores
            for term, data in term_data.items():
                tf = data['count']
                positions = data['positions']
                self.inverted_index[term].append([doc_id, tf, positions])
            
            doc_count += 1
            if doc_count % 10000 == 0:
                print(f"  Processed {doc_count} documents...")

        self.num_documents = doc_count
        
        # Second pass: compute IDF scores
        print(f"Computing IDF scores for {len(self.inverted_index)} terms...")
        for term, postings in self.inverted_index.items():
            df = len(postings)  # Document frequency
            idf = math.log(self.num_documents / df) if df > 0 else 0
            self.idf_scores[term] = idf
        
        # Third pass: compute document norms (L2 norm of TF-IDF vector)
        print("Computing document norms...")
        self.doc_norms = defaultdict(float)
        for term, postings in self.inverted_index.items():
            idf = self.idf_scores[term]
            for doc_id, tf, _ in postings:
                self.doc_norms[doc_id] += (tf * idf) ** 2
        
        for doc_id in self.doc_norms:
            self.doc_norms[doc_id] = math.sqrt(self.doc_norms[doc_id])
        
        print(f"TF-IDF index complete. Processed {doc_count} documents.")
        print(f"Total unique terms: {len(self.inverted_index)}")
        
        if self.optim == 'Skipping':
            print("\n--- Building Skip Pointers (Build-Time Optimization) ---")
            print("Note: Skip pointers are primarily beneficial for Boolean queries.")
            print("For TF-IDF ranked retrieval, benefits are limited.\n")
            print("Skip pointer integration for ranked indexers: NOT YET IMPLEMENTED")
            print("Continuing with standard TF-IDF index...\n")
        
        self._save_index(index_id)

    def _save_index(self, index_id: str):
        """Save index to disk (uncompressed)"""
        os.makedirs('indices', exist_ok=True)
        
        filename = f"indices/{self.identifier_short}.json"
        index_data = {
            "identifier": self.identifier_short,
            "inverted_index": dict(self.inverted_index),
            "documents": self.documents,
            "idf_scores": self.idf_scores,
            "doc_norms": dict(self.doc_norms),
            "num_documents": self.num_documents,
            "compression": "NONE"
        }
        with open(filename, 'w') as f:
            json.dump(index_data, f)
        print(f"TF-IDF index saved to {filename}")

    def load_index(self, index_id: str):
        """Load index from disk with compression support"""
        filename = f"indices/{index_id}.json"
        if not os.path.exists(filename):
            filename = f"{index_id}.json"
        
        print(f"--- Loading TF-IDF index from {filename} ---")
        
        with open(filename, 'r') as f:
            index_data = json.load(f)
        
        compression_type = index_data.get("compression", "NONE")
        
        if compression_type != "NONE" and compression_type in ["CODE", "CLIB"]:
            print(f"Detected compression: {compression_type}")
            
            if compression_type == "CODE":
                from src.compression.elias import EliasCompressor
                compressor = EliasCompressor
            elif compression_type == "CLIB":
                from src.compression.zlib_compressor import ZlibCompressor
                compressor = ZlibCompressor
            
            compressed_index = index_data["inverted_index"]
            self.inverted_index = defaultdict(
                list,
                compressor.decompress_inverted_index(compressed_index)
            )
        else:
            self.inverted_index = defaultdict(list, index_data["inverted_index"])
        
        self.documents = index_data["documents"]
        self.idf_scores = index_data.get("idf_scores", {})
        self.doc_norms = index_data.get("doc_norms", {})
        self.num_documents = index_data.get("num_documents", len(self.documents))
        
        print(f"Loaded {len(self.inverted_index)} terms, {len(self.documents)} documents")
        print(f"IDF scores computed for {len(self.idf_scores)} terms")

    def query(self, query_str: str, mode: str = 'TAAT', top_k: int = 10) -> List[str]:
        """
        TF-IDF ranked retrieval with phrase support and Cosine Similarity
        
        Args:
            query_str: Query string (can include quoted phrases)
            mode: 'TAAT' (Term-at-a-Time) or 'DAAT' (Document-at-a-Time)
            top_k: Number of top results
        
        Returns:
            List of document IDs ranked by normalized TF-IDF scores
        """
        # Extract phrases
        phrases = re.findall(r'"([^"]*)"', query_str)
        # Remove phrases from query string to get remaining terms
        remaining_str = re.sub(r'"[^"]*"', '', query_str)
        
        query_terms = preprocess_text(remaining_str)
        processed_phrases = [preprocess_text(p) for p in phrases if p.strip()]
        
        if mode == 'DAAT':
            return self._ranked_query_daat(query_terms, processed_phrases, top_k)
        else:
            return self._ranked_query_taat(query_terms, processed_phrases, top_k)
    
    def query_daat(self, query_str: str, top_k: int = 10) -> List[str]:
        """DEPRECATED: Use query(query_str, mode='DAAT') instead"""
        phrases = re.findall(r'"([^"]*)"', query_str)
        remaining_str = re.sub(r'"[^"]*"', '', query_str)
        query_terms = preprocess_text(remaining_str)
        processed_phrases = [preprocess_text(p) for p in phrases if p.strip()]
        return self._ranked_query_daat(query_terms, processed_phrases, top_k)
    
    def _ranked_query_daat(self, query_terms: List[str], phrases: List[List[str]] = None, top_k: int = 10) -> List[str]:
        """DAAT (Document-at-a-Time) query processing with TF-IDF and Cosine Similarity"""
        if phrases is None:
            phrases = []
        
        # Get postings for all query terms
        term_postings = {}
        for term in query_terms:
            if term in self.inverted_index:
                term_postings[term] = self.inverted_index[term]
        
        # Build doc -> term -> tf mapping
        doc_term_tf = defaultdict(dict)
        for term, postings in term_postings.items():
            for posting in postings:
                doc_id = posting[0]
                tf = posting[1]
                doc_term_tf[doc_id][term] = tf
        
        # Collect candidate docs from terms and phrases
        candidate_docs = set(doc_term_tf.keys())
        for phrase in phrases:
            for term in phrase:
                if term in self.inverted_index:
                    for posting in self.inverted_index[term]:
                        candidate_docs.add(posting[0])

        # Pre-fetch phrase term positions for efficiency
        phrase_positions_cache = {}
        for phrase in phrases:
            for term in phrase:
                if term not in phrase_positions_cache:
                    if term in term_postings:
                        phrase_positions_cache[term] = {p[0]: p[2] for p in term_postings[term]}
                    else:
                        phrase_positions_cache[term] = self._get_postings_with_positions(term)

        # Score each document
        doc_scores = []
        for doc_id in candidate_docs:
            score = 0.0
            
            # Term contributions
            for term in query_terms:
                if term in doc_term_tf.get(doc_id, {}):
                    tf = doc_term_tf[doc_id][term]
                    idf = self.idf_scores.get(term, 0)
                    score += tf * idf
            
            # Phrase Boosting (Virtual Term with IDF-based weight)
            for phrase in phrases:
                count = self._count_phrase_matches(doc_id, phrase, phrase_positions_cache)
                if count > 0:
                    phrase_idf_sum = sum(self.idf_scores.get(t, 0) for t in phrase)
                    score += count * phrase_idf_sum
            
            # Normalize by document length (Cosine Similarity)
            norm = self.doc_norms.get(doc_id, 1.0)
            if norm == 0:
                norm = 1.0
            score /= norm
            
            doc_scores.append((doc_id, score))
        
        # Sort and return top-K
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, score in doc_scores[:top_k]]
    
    def _get_postings_with_positions(self, term: str) -> Dict[str, List[int]]:
        """Extract positions from posting list. Returns: {doc_id: [positions]}"""
        postings = self.inverted_index.get(term, [])
        return {posting[0]: posting[2] for posting in postings}
    
    def _count_phrase_matches(self, doc_id: str, terms: List[str], positions_cache: Dict[str, Dict[str, List[int]]] = None) -> int:
        """Count occurrences of the exact phrase in the document"""
        positions_lists = []
        for term in terms:
            if positions_cache and term in positions_cache:
                term_positions = positions_cache[term]
            else:
                term_positions = self._get_postings_with_positions(term)
            
            if doc_id not in term_positions:
                return 0
            positions_lists.append(term_positions[doc_id])
        
        count = 0
        first_term_positions = positions_lists[0]
        for start_pos in first_term_positions:
            match = True
            for i in range(1, len(terms)):
                expected_pos = start_pos + i
                if expected_pos not in positions_lists[i]:
                    match = False
                    break
            if match:
                count += 1
        return count
    
    def _ranked_query_taat(self, query_terms: List[str], phrases: List[List[str]] = None, top_k: int = 10) -> List[str]:
        """TAAT (Term-at-a-Time) query processing with TF-IDF and Cosine Similarity"""
        if phrases is None:
            phrases = []
            
        doc_scores = defaultdict(float)
        
        # Process each term
        for term in query_terms:
            if term not in self.inverted_index:
                continue
            
            postings = self.inverted_index[term]
            idf = self.idf_scores.get(term, 0)
            
            for posting in postings:
                doc_id = posting[0]
                tf = posting[1]
                tf_idf = tf * idf
                doc_scores[doc_id] += tf_idf
        
        # Process phrases
        for phrase in phrases:
            # Pre-fetch positions for this phrase
            phrase_positions_cache = {}
            for term in phrase:
                phrase_positions_cache[term] = self._get_postings_with_positions(term)

            # Find docs containing all terms of phrase
            candidate_docs = None
            for term in phrase:
                if term in self.inverted_index:
                    term_docs = set(phrase_positions_cache[term].keys())
                    if candidate_docs is None:
                        candidate_docs = term_docs
                    else:
                        candidate_docs &= term_docs
                else:
                    candidate_docs = set()
                    break
            
            if candidate_docs:
                phrase_idf_sum = sum(self.idf_scores.get(t, 0) for t in phrase)
                for doc_id in candidate_docs:
                    count = self._count_phrase_matches(doc_id, phrase, phrase_positions_cache)
                    if count > 0:
                        doc_scores[doc_id] += count * phrase_idf_sum

        # Normalize by document length
        final_scores = []
        for doc_id, score in doc_scores.items():
            norm = self.doc_norms.get(doc_id, 1.0)
            if norm == 0:
                norm = 1.0
            final_scores.append((doc_id, score / norm))
        
        # Sort and return top-K
        final_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, score in final_scores[:top_k]]

    def update_index(self, index_id: str, remove_docs: Iterable[Dict], add_docs: Iterable[Dict]):
        """Update index by removing and adding documents"""
        # Remove documents
        for doc in remove_docs:
            doc_id = doc['doc_id']
            if doc_id in self.documents:
                del self.documents[doc_id]
                if doc_id in self.doc_norms:
                    del self.doc_norms[doc_id]
                self.num_documents -= 1
                for term in self.inverted_index:
                    self.inverted_index[term] = [
                        posting for posting in self.inverted_index[term] 
                        if posting[0] != doc_id
                    ]
        
        # Add new documents
        for doc in add_docs:
            doc_id = doc['doc_id']
            self.documents[doc_id] = {'title': doc.get('title', 'No Title')}
            self.num_documents += 1
            
            if 'tokens' in doc:
                tokens = doc['tokens']
            else:
                tokens = preprocess_text(doc.get('content', ''))
            
            term_data = defaultdict(lambda: {'count': 0, 'positions': []})
            for i, token in enumerate(tokens):
                term_data[token]['count'] += 1
                term_data[token]['positions'].append(i)
            
            for term, data in term_data.items():
                tf = data['count']
                positions = data['positions']
                self.inverted_index[term].append([doc_id, tf, positions])
        
        # Recompute IDF scores for all terms
        print("Recomputing IDF scores...")
        for term, postings in self.inverted_index.items():
            df = len(postings)
            idf = math.log(self.num_documents / df) if df > 0 else 0
            self.idf_scores[term] = idf
        
        # Recompute norms
        print("Recomputing document norms...")
        self.doc_norms = defaultdict(float)
        for term, postings in self.inverted_index.items():
            idf = self.idf_scores[term]
            for doc_id, tf, _ in postings:
                self.doc_norms[doc_id] += (tf * idf) ** 2
        
        for doc_id in self.doc_norms:
            self.doc_norms[doc_id] = math.sqrt(self.doc_norms[doc_id])
        
        self._save_index(index_id)

    def delete_index(self, index_id: str):
        """Delete index from disk"""
        filename = f"{self.identifier_short}.json"
        if os.path.exists(filename):
            os.remove(filename)
            print(f"Deleted index: {filename}")

    def list_indices(self) -> Iterable[str]:
        """List all available indices"""
        indices = []
        for file in os.listdir('.'):
            if file.startswith('SelfIndex_i3') and file.endswith('.json'):
                indices.append(file[:-5])
        return indices

    def list_indexed_files(self, index_id: str) -> Iterable[str]:
        """List all documents in the index"""
        return list(self.documents.keys())
