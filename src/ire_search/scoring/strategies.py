"""
Scoring Strategies for Information Retrieval

Implements the Strategy Pattern for different ranking/scoring approaches:
- BooleanScorer: Set-based Boolean retrieval (AND/OR/NOT) with Shunting-yard parser
- TFScorer: Term Frequency with L2 document length normalization
- TFIDFScorer: TF-IDF with cosine similarity scoring
- BM25Scorer: Okapi BM25 — industry-standard probabilistic ranking

Each scorer implements a common interface so the unified SelfIndexer
can swap scoring strategies without changing any other code.
"""

import math
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Any, Set, Tuple, Optional


# =============================================================================
# Abstract Base Class
# =============================================================================

class ScoringStrategy(ABC):
    """
    Abstract base class for all scoring strategies.
    
    A scoring strategy defines:
    1. How documents are indexed (build_postings)
    2. What metadata is needed (compute_metadata)
    3. How queries are scored (score_query)
    """
    
    @property
    @abstractmethod
    def index_type(self) -> str:
        """Return the index type name (e.g., 'BOOLEAN', 'WORDCOUNT', 'TFIDF', 'BM25')"""
        ...
    
    @abstractmethod
    def build_postings(self, doc_id: str, tokens: List[str]) -> Dict[str, list]:
        """
        Build postings for a single document.
        
        Args:
            doc_id: Unique document identifier
            tokens: Preprocessed tokens from the document
            
        Returns:
            Dict mapping term -> posting entry for this document.
            The exact format depends on the scorer type.
        """
        ...
    
    @abstractmethod
    def compute_metadata(self, inverted_index: Dict[str, list], 
                         num_documents: int) -> Dict[str, Any]:
        """
        Compute any global metadata after all documents are indexed.
        Called once after all documents have been processed.
        
        Args:
            inverted_index: The complete inverted index
            num_documents: Total number of documents
            
        Returns:
            Dict of metadata to store (e.g., idf_scores, doc_norms)
        """
        ...
    
    @abstractmethod
    def score_query(self, query_str: str, inverted_index: Dict[str, list],
                    metadata: Dict[str, Any], documents: Dict,
                    mode: str = 'TAAT', top_k: int = 10) -> List[str]:
        """
        Score documents against a query.
        
        Args:
            query_str: Raw query string
            inverted_index: The complete inverted index
            metadata: Global metadata from compute_metadata()
            documents: Document metadata dict
            mode: 'TAAT' or 'DAAT' processing mode
            top_k: Number of top results to return
            
        Returns:
            List of document IDs, ranked by relevance (or sorted for Boolean)
        """
        ...


# =============================================================================
# Boolean Scorer — x=1
# =============================================================================

class BooleanScorer(ScoringStrategy):
    """
    Boolean retrieval with Shunting-yard algorithm for query parsing.
    
    Supports: AND, OR, NOT operators, parentheses, phrase queries.
    Returns unranked set of matching document IDs (sorted).
    
    Query examples:
        "machine" AND "learning"
        "neural" OR ("deep" AND "network")
        "machine" AND NOT "learning"
    """
    
    @property
    def index_type(self) -> str:
        return 'BOOLEAN'
    
    def build_postings(self, doc_id: str, tokens: List[str]) -> Dict[str, list]:
        """Boolean postings: term -> [doc_id, [positions]]"""
        term_positions = defaultdict(list)
        for i, token in enumerate(tokens):
            term_positions[token].append(i)
        return {term: [doc_id, positions] for term, positions in term_positions.items()}
    
    def compute_metadata(self, inverted_index: Dict[str, list],
                         num_documents: int) -> Dict[str, Any]:
        """Boolean needs no global metadata."""
        return {}
    
    def score_query(self, query_str: str, inverted_index: Dict[str, list],
                    metadata: Dict[str, Any], documents: Dict,
                    mode: str = 'TAAT', top_k: int = 10) -> List[str]:
        """Boolean query using Shunting-yard algorithm."""
        from ..core.preprocessor import preprocess_text
        
        tokens = self._tokenize_query(query_str)
        if not tokens:
            return []
        
        tokens = self._preprocess_phrase_queries(tokens)
        
        # Shunting-yard algorithm
        output_queue = []
        operator_stack = []
        precedence = {'OR': 1, 'AND': 2, 'NOT': 3}
        
        for token in tokens:
            if token.startswith('"') or token.startswith('PHRASE("'):
                output_queue.append(token)
            elif token == '(':
                operator_stack.append(token)
            elif token == ')':
                while operator_stack and operator_stack[-1] != '(':
                    output_queue.append(operator_stack.pop())
                if not operator_stack or operator_stack[-1] != '(':
                    return []
                operator_stack.pop()
            elif token in precedence:
                while (operator_stack and operator_stack[-1] != '(' and
                       precedence.get(operator_stack[-1], 0) >= precedence[token]):
                    output_queue.append(operator_stack.pop())
                operator_stack.append(token)
            else:
                return []
        
        while operator_stack:
            if operator_stack[-1] == '(':
                return []
            output_queue.append(operator_stack.pop())
        
        # RPN evaluation
        eval_stack = []
        all_doc_ids = set(documents.keys())
        
        for token in output_queue:
            if token.startswith('PHRASE("'):
                phrase_content = token[8:-2]
                phrase_terms = [f'"{term}"' for term in phrase_content.split()]
                result = self._check_phrase(phrase_terms, inverted_index)
                eval_stack.append(result)
            elif token.startswith('"'):
                eval_stack.append(self._get_postings(token, inverted_index))
            elif token == 'AND':
                if len(eval_stack) < 2:
                    return []
                right = eval_stack.pop()
                left = eval_stack.pop()
                eval_stack.append(left.intersection(right))
            elif token == 'OR':
                if len(eval_stack) < 2:
                    return []
                right = eval_stack.pop()
                left = eval_stack.pop()
                eval_stack.append(left.union(right))
            elif token == 'NOT':
                if len(eval_stack) < 1:
                    return []
                operand = eval_stack.pop()
                eval_stack.append(all_doc_ids.difference(operand))
            else:
                return []
        
        if len(eval_stack) == 1:
            return sorted(list(eval_stack[0]))
        return []
    
    def _tokenize_query(self, query_str: str) -> List[str]:
        """Tokenize a Boolean query into terms and operators."""
        from ..core.preprocessor import preprocess_text
        
        tokens = []
        i = 0
        chars = query_str.strip()
        
        while i < len(chars):
            if chars[i] == ' ':
                i += 1
                continue
            elif chars[i] == '(':
                tokens.append('(')
                i += 1
            elif chars[i] == ')':
                tokens.append(')')
                i += 1
            elif chars[i:i+3] == 'AND':
                if i + 3 >= len(chars) or not chars[i+3].isalnum():
                    tokens.append('AND')
                    i += 3
                    continue
                # Part of a word
                j = i
                while j < len(chars) and chars[j] not in ' ()':
                    j += 1
                word = chars[i:j]
                processed = preprocess_text(word)
                for p in processed:
                    tokens.append(f'"{p}"')
                i = j
            elif chars[i:i+2] == 'OR':
                if i + 2 >= len(chars) or not chars[i+2].isalnum():
                    tokens.append('OR')
                    i += 2
                    continue
                j = i
                while j < len(chars) and chars[j] not in ' ()':
                    j += 1
                word = chars[i:j]
                processed = preprocess_text(word)
                for p in processed:
                    tokens.append(f'"{p}"')
                i = j
            elif chars[i:i+3] == 'NOT':
                if i + 3 >= len(chars) or not chars[i+3].isalnum():
                    tokens.append('NOT')
                    i += 3
                    continue
                j = i
                while j < len(chars) and chars[j] not in ' ()':
                    j += 1
                word = chars[i:j]
                processed = preprocess_text(word)
                for p in processed:
                    tokens.append(f'"{p}"')
                i = j
            elif chars[i] == '"':
                j = chars.index('"', i + 1) + 1 if '"' in chars[i+1:] else len(chars)
                phrase = chars[i:j]
                tokens.append(f'PHRASE({phrase})')
                i = j
            else:
                j = i
                while j < len(chars) and chars[j] not in ' ()':
                    j += 1
                word = chars[i:j]
                processed = preprocess_text(word)
                for p in processed:
                    tokens.append(f'"{p}"')
                i = j
        
        return tokens
    
    def _preprocess_phrase_queries(self, tokens: List[str]) -> List[str]:
        """Convert PHRASE tokens into special tokens for Shunting-yard."""
        result = []
        for token in tokens:
            if token.startswith('PHRASE('):
                from ..core.preprocessor import preprocess_text
                phrase_content = token[8:-2] if token.startswith('PHRASE("') else token[7:-1]
                processed_terms = preprocess_text(phrase_content)
                if processed_terms:
                    result.append(f'PHRASE("{" ".join(processed_terms)}")')
            else:
                result.append(token)
        return result
    
    def _get_postings(self, token: str, inverted_index: Dict) -> Set:
        """Get set of doc IDs for a preprocessed term."""
        from ..core.preprocessor import preprocess_text
        
        term = token.strip('"')
        processed = preprocess_text(term)
        if not processed:
            return set()
        term = processed[0]
        
        if term in inverted_index:
            return set(posting[0] for posting in inverted_index[term])
        return set()
    
    def _check_phrase(self, phrase_terms: List[str], inverted_index: Dict) -> Set:
        """Check which documents contain the exact phrase."""
        from ..core.preprocessor import preprocess_text
        
        terms = []
        for pt in phrase_terms:
            t = pt.strip('"')
            processed = preprocess_text(t)
            if processed:
                terms.append(processed[0])
        
        if not terms:
            return set()
        
        # Get position data for each term
        term_positions = {}
        for term in terms:
            if term not in inverted_index:
                return set()
            term_positions[term] = {
                posting[0]: posting[1] if len(posting) > 1 and isinstance(posting[1], list) else []
                for posting in inverted_index[term]
            }
        
        # Find candidate docs (appear in all terms)
        candidate_docs = set(term_positions[terms[0]].keys())
        for term in terms[1:]:
            candidate_docs &= set(term_positions[term].keys())
        
        # Check positional adjacency
        result = set()
        for doc_id in candidate_docs:
            first_positions = term_positions[terms[0]].get(doc_id, [])
            for start_pos in first_positions:
                match = True
                for offset, term in enumerate(terms[1:], 1):
                    expected_pos = start_pos + offset
                    if expected_pos not in set(term_positions[term].get(doc_id, [])):
                        match = False
                        break
                if match:
                    result.add(doc_id)
                    break
        
        return result


# =============================================================================
# TF Scorer — x=2
# =============================================================================

class TFScorer(ScoringStrategy):
    """
    Term Frequency scorer with L2 document length normalization.
    
    Scoring: score(q,d) = Σ tf(t,d) / ||d||
    where ||d|| = sqrt(Σ tf(t,d)²) is the L2 norm.
    
    Also supports phrase boosting via position-based matching.
    """
    
    @property
    def index_type(self) -> str:
        return 'WORDCOUNT'
    
    def build_postings(self, doc_id: str, tokens: List[str]) -> Dict[str, list]:
        """TF postings: term -> [doc_id, tf, [positions]]"""
        term_data = defaultdict(lambda: {'count': 0, 'positions': []})
        for i, token in enumerate(tokens):
            term_data[token]['count'] += 1
            term_data[token]['positions'].append(i)
        
        return {
            term: [doc_id, data['count'], data['positions']]
            for term, data in term_data.items()
        }
    
    def compute_metadata(self, inverted_index: Dict[str, list],
                         num_documents: int) -> Dict[str, Any]:
        """Compute L2 document norms."""
        doc_norms = defaultdict(float)
        for term, postings in inverted_index.items():
            for posting in postings:
                doc_id = posting[0]
                tf = posting[1]
                doc_norms[doc_id] += tf * tf
        
        for doc_id in doc_norms:
            doc_norms[doc_id] = math.sqrt(doc_norms[doc_id]) if doc_norms[doc_id] > 0 else 1.0
        
        return {'doc_norms': dict(doc_norms)}
    
    def score_query(self, query_str: str, inverted_index: Dict[str, list],
                    metadata: Dict[str, Any], documents: Dict,
                    mode: str = 'TAAT', top_k: int = 10) -> List[str]:
        """TF-based ranked retrieval with phrase support."""
        from ..core.preprocessor import preprocess_text
        
        doc_norms = metadata.get('doc_norms', {})
        
        # Extract phrases
        phrases = re.findall(r'"([^"]*)"', query_str)
        remaining_str = re.sub(r'"[^"]*"', '', query_str)
        query_terms = preprocess_text(remaining_str)
        processed_phrases = [preprocess_text(p) for p in phrases if p.strip()]
        
        if mode == 'DAAT':
            return self._ranked_query_daat(query_terms, processed_phrases,
                                           inverted_index, doc_norms, top_k)
        else:
            return self._ranked_query_taat(query_terms, processed_phrases,
                                           inverted_index, doc_norms, top_k)
    
    def _ranked_query_taat(self, query_terms, phrases, inverted_index, doc_norms, top_k):
        """TAAT processing for TF scoring."""
        doc_scores = defaultdict(float)
        
        for term in query_terms:
            if term not in inverted_index:
                continue
            for posting in inverted_index[term]:
                doc_scores[posting[0]] += posting[1]  # tf
        
        # Process phrases
        for phrase in phrases:
            positions_cache = {
                term: self._get_postings_with_positions(term, inverted_index)
                for term in phrase
            }
            candidate_docs = None
            for term in phrase:
                if term in inverted_index:
                    term_docs = set(positions_cache[term].keys())
                    candidate_docs = term_docs if candidate_docs is None else candidate_docs & term_docs
                else:
                    candidate_docs = set()
                    break
            if candidate_docs:
                for doc_id in candidate_docs:
                    count = self._count_phrase_matches(doc_id, phrase, positions_cache)
                    if count > 0:
                        doc_scores[doc_id] += count * len(phrase)
        
        # Normalize and rank
        final_scores = []
        for doc_id, score in doc_scores.items():
            norm = doc_norms.get(doc_id, 1.0) or 1.0
            final_scores.append((doc_id, score / norm))
        
        final_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in final_scores[:top_k]]
    
    def _ranked_query_daat(self, query_terms, phrases, inverted_index, doc_norms, top_k):
        """DAAT processing for TF scoring."""
        # Build doc -> term -> tf mapping
        doc_term_tf = defaultdict(dict)
        for term in query_terms:
            if term in inverted_index:
                for posting in inverted_index[term]:
                    doc_term_tf[posting[0]][term] = posting[1]
        
        candidate_docs = set(doc_term_tf.keys())
        for phrase in phrases:
            for term in phrase:
                if term in inverted_index:
                    for posting in inverted_index[term]:
                        candidate_docs.add(posting[0])
        
        # Pre-cache phrase positions
        phrase_positions_cache = {}
        for phrase in phrases:
            for term in phrase:
                if term not in phrase_positions_cache:
                    phrase_positions_cache[term] = self._get_postings_with_positions(
                        term, inverted_index)
        
        doc_scores = []
        for doc_id in candidate_docs:
            score = sum(
                doc_term_tf.get(doc_id, {}).get(term, 0)
                for term in query_terms
            )
            for phrase in phrases:
                count = self._count_phrase_matches(doc_id, phrase, phrase_positions_cache)
                if count > 0:
                    score += count * len(phrase)
            
            norm = doc_norms.get(doc_id, 1.0) or 1.0
            doc_scores.append((doc_id, score / norm))
        
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in doc_scores[:top_k]]
    
    @staticmethod
    def _get_postings_with_positions(term: str, inverted_index: Dict) -> Dict:
        """Extract positions: {doc_id: [positions]}"""
        postings = inverted_index.get(term, [])
        return {p[0]: p[2] for p in postings if len(p) > 2}
    
    @staticmethod
    def _count_phrase_matches(doc_id, terms, positions_cache) -> int:
        """Count exact phrase matches using position sets."""
        positions_lists = []
        for term in terms:
            term_positions = positions_cache.get(term, {})
            if doc_id not in term_positions:
                return 0
            positions_lists.append(term_positions[doc_id])
        
        position_sets = [set(pl) for pl in positions_lists[1:]]
        count = 0
        for start_pos in positions_lists[0]:
            match = True
            for i, pos_set in enumerate(position_sets):
                if start_pos + i + 1 not in pos_set:
                    match = False
                    break
            if match:
                count += 1
        return count


# =============================================================================
# TF-IDF Scorer — x=3
# =============================================================================

class TFIDFScorer(ScoringStrategy):
    """
    TF-IDF scorer with cosine similarity.
    
    Scoring: score(q,d) = Σ tf(t,d) × idf(t) / ||d||
    where idf(t) = log(N / df(t))
    and ||d|| = sqrt(Σ (tf(t,d) × idf(t))²)
    
    Supports phrase boosting and both TAAT/DAAT processing.
    """
    
    @property
    def index_type(self) -> str:
        return 'TFIDF'
    
    def build_postings(self, doc_id: str, tokens: List[str]) -> Dict[str, list]:
        """Same as TF: term -> [doc_id, tf, [positions]]"""
        term_data = defaultdict(lambda: {'count': 0, 'positions': []})
        for i, token in enumerate(tokens):
            term_data[token]['count'] += 1
            term_data[token]['positions'].append(i)
        
        return {
            term: [doc_id, data['count'], data['positions']]
            for term, data in term_data.items()
        }
    
    def compute_metadata(self, inverted_index: Dict[str, list],
                         num_documents: int) -> Dict[str, Any]:
        """Compute IDF scores and TF-IDF document norms."""
        # IDF scores
        idf_scores = {}
        for term, postings in inverted_index.items():
            df = len(postings)
            idf_scores[term] = math.log(num_documents / df) if df > 0 else 0
        
        # Document norms (L2 norm of TF-IDF vector)
        doc_norms = defaultdict(float)
        for term, postings in inverted_index.items():
            idf = idf_scores[term]
            for posting in postings:
                doc_id = posting[0]
                tf = posting[1]
                doc_norms[doc_id] += (tf * idf) ** 2
        
        for doc_id in doc_norms:
            doc_norms[doc_id] = math.sqrt(doc_norms[doc_id]) if doc_norms[doc_id] > 0 else 1.0
        
        return {
            'idf_scores': idf_scores,
            'doc_norms': dict(doc_norms),
            'num_documents': num_documents,
        }
    
    def score_query(self, query_str: str, inverted_index: Dict[str, list],
                    metadata: Dict[str, Any], documents: Dict,
                    mode: str = 'TAAT', top_k: int = 10) -> List[str]:
        """TF-IDF ranked retrieval with phrase support."""
        from ..core.preprocessor import preprocess_text
        
        idf_scores = metadata.get('idf_scores', {})
        doc_norms = metadata.get('doc_norms', {})
        
        phrases = re.findall(r'"([^"]*)"', query_str)
        remaining_str = re.sub(r'"[^"]*"', '', query_str)
        query_terms = preprocess_text(remaining_str)
        processed_phrases = [preprocess_text(p) for p in phrases if p.strip()]
        
        # TAAT processing (primary for TF-IDF)
        doc_scores = defaultdict(float)
        
        for term in query_terms:
            if term not in inverted_index:
                continue
            idf = idf_scores.get(term, 0)
            for posting in inverted_index[term]:
                doc_scores[posting[0]] += posting[1] * idf  # tf * idf
        
        # Phrase boosting
        for phrase in processed_phrases:
            positions_cache = {
                term: TFScorer._get_postings_with_positions(term, inverted_index)
                for term in phrase
            }
            candidate_docs = None
            for term in phrase:
                if term in inverted_index:
                    term_docs = set(positions_cache[term].keys())
                    candidate_docs = term_docs if candidate_docs is None else candidate_docs & term_docs
                else:
                    candidate_docs = set()
                    break
            if candidate_docs:
                for doc_id in candidate_docs:
                    count = TFScorer._count_phrase_matches(doc_id, phrase, positions_cache)
                    if count > 0:
                        avg_idf = sum(idf_scores.get(t, 0) for t in phrase) / len(phrase)
                        doc_scores[doc_id] += count * len(phrase) * avg_idf
        
        # Normalize and rank
        final_scores = []
        for doc_id, score in doc_scores.items():
            norm = doc_norms.get(doc_id, 1.0) or 1.0
            final_scores.append((doc_id, score / norm))
        
        final_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in final_scores[:top_k]]


# =============================================================================
# BM25 Scorer — x=4 (NEW)
# =============================================================================

class BM25Scorer(ScoringStrategy):
    """
    Okapi BM25 — industry-standard probabilistic ranking function.
    
    score(q,d) = Σ IDF(t) × (tf(t,d) × (k1 + 1)) / (tf(t,d) + k1 × (1 - b + b × |d|/avgdl))
    
    where:
        IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
        k1 = 1.2 (term frequency saturation)
        b = 0.75 (document length normalization)
        avgdl = average document length
    
    Parameters:
        k1: Controls term frequency saturation. Higher = more credit for repeated terms.
        b: Controls document length normalization. 0 = no normalization, 1 = full normalization.
    """
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
    
    @property
    def index_type(self) -> str:
        return 'BM25'
    
    def build_postings(self, doc_id: str, tokens: List[str]) -> Dict[str, list]:
        """BM25 postings: term -> [doc_id, tf, doc_length, [positions]]"""
        term_data = defaultdict(lambda: {'count': 0, 'positions': []})
        for i, token in enumerate(tokens):
            term_data[token]['count'] += 1
            term_data[token]['positions'].append(i)
        
        doc_length = len(tokens)
        return {
            term: [doc_id, data['count'], doc_length, data['positions']]
            for term, data in term_data.items()
        }
    
    def compute_metadata(self, inverted_index: Dict[str, list],
                         num_documents: int) -> Dict[str, Any]:
        """Compute BM25 IDF scores and average document length."""
        # IDF using BM25 formula (Robertson-Sparck Jones IDF)
        idf_scores = {}
        for term, postings in inverted_index.items():
            df = len(postings)
            idf_scores[term] = math.log(
                (num_documents - df + 0.5) / (df + 0.5) + 1
            )
        
        # Average document length
        doc_lengths = {}
        for term, postings in inverted_index.items():
            for posting in postings:
                doc_id = posting[0]
                if doc_id not in doc_lengths:
                    doc_lengths[doc_id] = posting[2] if len(posting) > 2 else 0
        
        avg_doc_length = (
            sum(doc_lengths.values()) / len(doc_lengths)
            if doc_lengths else 1.0
        )
        
        return {
            'idf_scores': idf_scores,
            'doc_lengths': doc_lengths,
            'avg_doc_length': avg_doc_length,
            'num_documents': num_documents,
            'k1': self.k1,
            'b': self.b,
        }
    
    def score_query(self, query_str: str, inverted_index: Dict[str, list],
                    metadata: Dict[str, Any], documents: Dict,
                    mode: str = 'TAAT', top_k: int = 10) -> List[str]:
        """BM25 ranked retrieval."""
        from ..core.preprocessor import preprocess_text
        
        idf_scores = metadata.get('idf_scores', {})
        doc_lengths = metadata.get('doc_lengths', {})
        avgdl = metadata.get('avg_doc_length', 1.0)
        k1 = metadata.get('k1', self.k1)
        b = metadata.get('b', self.b)
        
        query_terms = preprocess_text(query_str)
        
        doc_scores = defaultdict(float)
        
        for term in query_terms:
            if term not in inverted_index:
                continue
            
            idf = idf_scores.get(term, 0)
            
            for posting in inverted_index[term]:
                doc_id = posting[0]
                tf = posting[1]
                dl = doc_lengths.get(doc_id, avgdl)
                
                # BM25 scoring formula
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * dl / avgdl)
                doc_scores[doc_id] += idf * numerator / denominator
        
        # Rank and return top-K
        scored_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in scored_docs[:top_k]]


# =============================================================================
# Factory
# =============================================================================

SCORERS = {
    1: BooleanScorer,
    2: TFScorer,
    3: TFIDFScorer,
    4: BM25Scorer,
}


def get_scorer(x: int, **kwargs) -> ScoringStrategy:
    """
    Factory function to get a scoring strategy by index type number.
    
    Args:
        x: Index type (1=Boolean, 2=TF, 3=TF-IDF, 4=BM25)
        **kwargs: Additional args (e.g., k1, b for BM25)
    
    Returns:
        ScoringStrategy instance
    """
    if x not in SCORERS:
        raise ValueError(f"Unknown index type: {x}. Valid: {list(SCORERS.keys())}")
    return SCORERS[x](**kwargs) if kwargs and x == 4 else SCORERS[x]()
