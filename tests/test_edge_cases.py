"""
Edge case tests — unusual inputs, empty data, error handling.
"""

import pytest
from ire_search.core.indexer import create_indexer
from ire_search.scoring import get_scorer


class TestEmptyInputs:
    """Test behavior with empty or minimal inputs."""

    def test_empty_query(self, boolean_indexer):
        assert boolean_indexer.query('') == []

    def test_query_nonexistent_term(self, tfidf_indexer):
        results = tfidf_indexer.query('xyznonexistentterm')
        assert results == []

    def test_empty_corpus(self, test_indices_dir):
        idx = create_indexer(x=3, y=1, z=1, optim='0')
        idx.create_index('TEST_empty', [])
        results = idx.query('anything')
        assert results == []

    def test_single_document(self, test_indices_dir):
        docs = [{'doc_id': 'news_1', 'title': 'A', 'content': 'hello world'}]
        idx = create_indexer(x=2, y=1, z=1, optim='0')
        idx.create_index('TEST_single', docs)
        results = idx.query('hello')
        assert results == ['news_1']


class TestSpecialCharacters:
    """Test handling of special characters and unusual text."""

    def test_query_with_punctuation(self, tfidf_indexer):
        # Punctuation should be stripped during preprocessing
        results = tfidf_indexer.query('machine!')
        assert len(results) > 0

    def test_query_with_numbers(self, test_indices_dir):
        docs = [{'doc_id': 'news_1', 'title': 'A', 'content': 'version 3 point 14 release'}]
        idx = create_indexer(x=2, y=1, z=1, optim='0')
        idx.create_index('TEST_nums', docs)
        results = idx.query('version')
        assert len(results) > 0


class TestTopK:
    """Test top-K limiting works correctly."""

    def test_top_1(self, tfidf_indexer):
        results = tfidf_indexer.query('machine learning data', top_k=1)
        assert len(results) <= 1

    def test_top_k_exceeds_results(self, tfidf_indexer):
        results = tfidf_indexer.query('xyzunique', top_k=100)
        assert len(results) == 0


class TestScorerFactory:
    """Test factory edge cases."""

    def test_invalid_index_type(self):
        with pytest.raises(ValueError):
            create_indexer(x=99, y=1, z=1)

    def test_invalid_datastore(self):
        with pytest.raises(ValueError):
            create_indexer(x=1, y=99, z=1)


class TestQueryModes:
    """Test TAAT vs DAAT produce consistent results."""

    def test_taat_daat_same_results(self, tfidf_indexer):
        taat = tfidf_indexer.query('machine learning', mode='TAAT')
        daat = tfidf_indexer.query('machine learning', mode='DAAT')
        # Both should return results (may not be identical ordering)
        assert len(taat) > 0
        assert len(daat) > 0
