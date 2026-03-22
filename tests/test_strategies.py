"""
Unit tests for scoring strategies.

Tests each scorer independently: postings format, metadata computation, query results.
"""

import pytest
from collections import defaultdict
from ire_search.scoring import BooleanScorer, TFScorer, TFIDFScorer, BM25Scorer, get_scorer


class TestBooleanScorer:
    """Tests for BooleanScorer."""

    def test_build_postings_format(self):
        scorer = BooleanScorer()
        postings = scorer.build_postings('doc1', ['hello', 'world', 'hello'])
        assert 'hello' in postings
        assert 'world' in postings
        assert postings['hello'][0] == 'doc1'
        assert postings['hello'][1] == [0, 2]  # positions

    def test_compute_metadata_empty(self):
        scorer = BooleanScorer()
        meta = scorer.compute_metadata({}, 0)
        assert meta == {}

    def test_simple_and_query(self, boolean_indexer):
        results = boolean_indexer.query('"machine" AND "learning"')
        # Should find docs containing both 'machine' and 'learning'
        assert len(results) > 0
        assert all(isinstance(r, str) for r in results)

    def test_or_query(self, boolean_indexer):
        results = boolean_indexer.query('"python" OR "machine"')
        assert len(results) >= 2

    def test_not_query(self, boolean_indexer):
        all_results = boolean_indexer.query('"machine"')
        not_results = boolean_indexer.query('"machine" AND NOT "python"')
        assert len(not_results) <= len(all_results)

    def test_empty_query_returns_empty(self, boolean_indexer):
        results = boolean_indexer.query('')
        assert results == []


class TestTFScorer:
    """Tests for TFScorer."""

    def test_build_postings_format(self):
        scorer = TFScorer()
        postings = scorer.build_postings('d1', ['a', 'b', 'a'])
        assert postings['a'] == ['d1', 2, [0, 2]]  # doc_id, tf=2, positions
        assert postings['b'] == ['d1', 1, [1]]

    def test_metadata_has_doc_norms(self):
        scorer = TFScorer()
        index = defaultdict(list)
        index['cat'].append(['d1', 3, [0, 1, 2]])
        index['dog'].append(['d1', 1, [3]])
        meta = scorer.compute_metadata(index, 1)
        assert 'doc_norms' in meta
        assert 'd1' in meta['doc_norms']
        assert meta['doc_norms']['d1'] > 0

    def test_ranked_query(self, tf_indexer):
        results = tf_indexer.query('machine learning')
        assert len(results) > 0
        # Results should be strings (doc IDs)
        assert isinstance(results[0], str)


class TestTFIDFScorer:
    """Tests for TFIDFScorer."""

    def test_metadata_has_idf_and_norms(self):
        scorer = TFIDFScorer()
        index = defaultdict(list)
        index['rare'].append(['d1', 1, [0]])
        index['common'].append(['d1', 1, [1]])
        index['common'].append(['d2', 2, [0, 1]])
        meta = scorer.compute_metadata(index, 2)
        assert 'idf_scores' in meta
        assert 'doc_norms' in meta
        # Rare term should have higher IDF
        assert meta['idf_scores']['rare'] > meta['idf_scores']['common']

    def test_ranked_query(self, tfidf_indexer):
        results = tfidf_indexer.query('machine learning')
        assert len(results) > 0

    def test_top_k_limit(self, tfidf_indexer):
        results = tfidf_indexer.query('machine learning data', top_k=2)
        assert len(results) <= 2


class TestBM25Scorer:
    """Tests for BM25Scorer."""

    def test_build_postings_includes_doc_length(self):
        scorer = BM25Scorer()
        postings = scorer.build_postings('d1', ['a', 'b', 'c'])
        assert len(postings['a']) == 4  # [doc_id, tf, doc_length, positions]
        assert postings['a'][2] == 3  # doc_length = 3

    def test_metadata_has_avg_doc_length(self):
        scorer = BM25Scorer()
        index = defaultdict(list)
        index['hello'].append(['d1', 1, 5, [0]])
        index['world'].append(['d2', 1, 10, [0]])
        meta = scorer.compute_metadata(index, 2)
        assert 'avg_doc_length' in meta
        assert meta['avg_doc_length'] == 7.5

    def test_custom_parameters(self):
        scorer = BM25Scorer(k1=2.0, b=0.5)
        assert scorer.k1 == 2.0
        assert scorer.b == 0.5

    def test_ranked_query(self, bm25_indexer):
        results = bm25_indexer.query('machine learning')
        assert len(results) > 0


class TestScorerFactory:
    """Tests for the get_scorer() factory."""

    def test_all_valid_types(self):
        assert isinstance(get_scorer(1), BooleanScorer)
        assert isinstance(get_scorer(2), TFScorer)
        assert isinstance(get_scorer(3), TFIDFScorer)
        assert isinstance(get_scorer(4), BM25Scorer)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown index type: 99"):
            get_scorer(99)
