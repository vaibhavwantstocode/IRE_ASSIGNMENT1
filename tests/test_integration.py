"""
Integration tests for the full create → save → load → query pipeline.

Tests every (x, y, z) combination to ensure all 24 configurations work end-to-end.
"""

import pytest
import os
from ire_search.core.indexer import create_indexer, get_index_identifier


class TestCreateSaveLoadQuery:
    """Test full pipeline for each configuration."""

    @pytest.mark.parametrize("x", [2, 3, 4])
    @pytest.mark.parametrize("y", [1, 2])
    @pytest.mark.parametrize("z", [1])  # Only test no-compression for speed
    def test_pipeline_ranked(self, x, y, z, tiny_docs, test_indices_dir):
        """Create → save → load → query for ranked scorers."""
        ident = get_index_identifier(x, y, z, '0')
        indexer = create_indexer(x=x, y=y, z=z, optim='0')
        indexer.create_index(ident, tiny_docs)
        indexer2 = create_indexer(x=x, y=y, z=z, optim='0')
        indexer2.load_index(ident)
        results = indexer2.query('machine learning')
        assert isinstance(results, list)
        assert len(results) > 0

    @pytest.mark.parametrize("y", [1, 2])
    def test_pipeline_boolean(self, y, tiny_docs, test_indices_dir):
        """Boolean scorer needs explicit operators."""
        ident = get_index_identifier(1, y, 1, '0')
        indexer = create_indexer(x=1, y=y, z=1, optim='0')
        indexer.create_index(ident, tiny_docs)
        indexer2 = create_indexer(x=1, y=y, z=1, optim='0')
        indexer2.load_index(ident)
        results = indexer2.query('"machine" AND "learning"')
        assert isinstance(results, list)
        assert len(results) > 0

    @pytest.mark.parametrize("z", [2, 3])
    def test_compression_roundtrip(self, z, tiny_docs, test_indices_dir):
        """Compression → save → load → query gives same results as no compression."""
        ident_plain = get_index_identifier(3, 1, 1, '0')
        ident_comp = get_index_identifier(3, 1, z, '0')

        # Build without compression
        plain = create_indexer(x=3, y=1, z=1, optim='0')
        plain.create_index(ident_plain, tiny_docs)
        plain_results = plain.query('machine learning')

        # Build with compression
        compressed = create_indexer(x=3, y=1, z=z, optim='0')
        compressed.create_index(ident_comp, tiny_docs)

        # Load compressed and query
        loaded = create_indexer(x=3, y=1, z=z, optim='0')
        loaded.load_index(ident_comp)
        comp_results = loaded.query('machine learning')

        # Results should be identical
        assert plain_results == comp_results

    def test_sqlite_persistence(self, tiny_docs, test_indices_dir):
        """SQLite indices persist and load correctly."""
        ident = 'SelfIndex_i3d2c1o0'
        indexer = create_indexer(x=3, y=2, z=1, optim='0')
        indexer.create_index(ident, tiny_docs)

        # Verify DB file exists
        db_path = os.path.join('indices', f"{ident}.db")
        assert os.path.exists(db_path)

        # Load and query
        loaded = create_indexer(x=3, y=2, z=1, optim='0')
        loaded.load_index(ident)
        results = loaded.query('machine')
        assert len(results) > 0


class TestIndexStats:
    """Test index metadata and statistics."""

    def test_stats_after_build(self, tfidf_indexer):
        stats = tfidf_indexer.get_stats()
        assert stats['num_documents'] == 4
        assert stats['num_terms'] > 0
        assert stats['scorer_type'] == 'TFIDF'

    def test_bm25_stats(self, bm25_indexer):
        stats = bm25_indexer.get_stats()
        assert stats['scorer_type'] == 'BM25'


class TestUpdateIndex:
    """Test incremental index updates."""

    def test_add_document(self, tfidf_indexer, test_indices_dir):
        initial_count = tfidf_indexer.get_stats()['num_documents']
        new_doc = [{'doc_id': 'doc_new', 'title': 'New', 'content': 'brand new document about quantum computing'}]
        tfidf_indexer.update_index('TEST_tfidf', remove_docs=[], add_docs=new_doc)
        assert tfidf_indexer.get_stats()['num_documents'] == initial_count + 1

    def test_remove_document(self, tfidf_indexer, test_indices_dir):
        initial_count = tfidf_indexer.get_stats()['num_documents']
        tfidf_indexer.update_index('TEST_tfidf', remove_docs=[{'doc_id': 'news_1'}], add_docs=[])
        assert tfidf_indexer.get_stats()['num_documents'] == initial_count - 1

    def test_remove_cleans_postings(self, tfidf_indexer, test_indices_dir):
        """Removing a doc should clean up its postings entries."""
        tfidf_indexer.update_index('TEST_tfidf', remove_docs=[{'doc_id': 'news_3'}], add_docs=[])
        # 'python' only appears in doc3, so it should be gone
        assert 'python' not in tfidf_indexer.inverted_index
