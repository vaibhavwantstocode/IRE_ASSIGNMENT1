"""Optional semantic / Chroma / file-crawler smoke tests."""

import pytest


def test_local_indexer_reset(tmp_path):
    pytest.importorskip("chromadb")
    pytest.importorskip("sentence_transformers")
    from ire_search.integrations.chroma_local import LocalIndexer

    p = tmp_path / "chroma"
    idx = LocalIndexer(persist_directory=p)
    n = idx.add_documents(
        [
            {"doc_id": "a", "text": "hello world", "metadata": {"filename": "a.txt"}},
            {"doc_id": "b", "text": "goodbye moon", "metadata": {"filename": "b.txt"}},
        ]
    )
    assert n == 2
    hits = idx.search("hello", top_k=5)
    assert len(hits) >= 1
    idx.reset()
    assert idx.search("hello", top_k=1) == []


def test_file_crawler_md(tmp_path):
    pytest.importorskip("markitdown")

    root = tmp_path / "docs"
    root.mkdir()
    f = root / "hello.md"
    f.write_text("# Title\n\nHello from markdown.\n", encoding="utf-8")

    from ire_search.core.file_crawler import crawl_to_list

    docs = crawl_to_list(root)
    assert len(docs) == 1
    assert "Hello" in docs[0]["text"] or "hello" in docs[0]["text"].lower()
    assert docs[0]["metadata"]["suffix"] == ".md"
