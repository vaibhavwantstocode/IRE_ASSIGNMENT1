"""
Shared pytest fixtures for IRE Search Engine tests.

Provides reusable document collections and pre-built indexer instances.
"""

import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_root, "src"))

import pytest
import shutil

from ire_search.core.indexer import create_indexer, SelfIndexer
from ire_search.scoring import BooleanScorer, TFScorer, TFIDFScorer, BM25Scorer
from ire_search.storage import JSONStorage, SQLiteStorage


# ===========================================================================
# Test Documents
# ===========================================================================

@pytest.fixture
def tiny_docs():
    """Minimal document set for unit tests.
    Uses news_X format for Elias compressor compatibility."""
    return [
        {
            'doc_id': 'news_1',
            'title': 'Machine Learning Basics',
            'content': 'machine learning is a branch of artificial intelligence'
        },
        {
            'doc_id': 'news_2',
            'title': 'Deep Learning',
            'content': 'deep learning uses neural networks for machine vision'
        },
        {
            'doc_id': 'news_3',
            'title': 'Python Programming',
            'content': 'python is a popular programming language for data science'
        },
        {
            'doc_id': 'news_4',
            'title': 'Data Analysis',
            'content': 'data analysis and machine learning go hand in hand'
        },
    ]


@pytest.fixture
def phrase_docs():
    """Documents specifically for phrase query testing."""
    return [
        {
            'doc_id': 'news_10',
            'title': 'NLP Paper',
            'content': 'natural language processing uses machine learning methods'
        },
        {
            'doc_id': 'news_11',
            'title': 'CV Paper',
            'content': 'computer vision and natural language understanding'
        },
        {
            'doc_id': 'news_12',
            'title': 'ML Survey',
            'content': 'machine learning and natural language processing survey'
        },
    ]


# ===========================================================================
# Test Index Directory
# ===========================================================================

@pytest.fixture(autouse=True)
def test_indices_dir(tmp_path):
    """Use a temp directory for all test indices to avoid polluting the real indices/."""
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    os.makedirs('indices', exist_ok=True)
    yield tmp_path
    os.chdir(original_cwd)


# ===========================================================================
# Pre-built Indexers
# ===========================================================================

@pytest.fixture
def boolean_indexer(tiny_docs, test_indices_dir):
    """Pre-built Boolean indexer."""
    idx = create_indexer(x=1, y=1, z=1, optim='0')
    idx.create_index('TEST_bool', tiny_docs)
    return idx


@pytest.fixture
def tf_indexer(tiny_docs, test_indices_dir):
    """Pre-built TF indexer."""
    idx = create_indexer(x=2, y=1, z=1, optim='0')
    idx.create_index('TEST_tf', tiny_docs)
    return idx


@pytest.fixture
def tfidf_indexer(tiny_docs, test_indices_dir):
    """Pre-built TF-IDF indexer."""
    idx = create_indexer(x=3, y=1, z=1, optim='0')
    idx.create_index('TEST_tfidf', tiny_docs)
    return idx


@pytest.fixture
def bm25_indexer(tiny_docs, test_indices_dir):
    """Pre-built BM25 indexer."""
    idx = create_indexer(x=4, y=1, z=1, optim='0')
    idx.create_index('TEST_bm25', tiny_docs)
    return idx
