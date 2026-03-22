"""
Recursively extract text from supported file types under a root directory.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES: Set[str] = {".pdf", ".docx", ".pptx", ".md"}


def _doc_id_for_path(path: Path, root: Path) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        rel = path.name
    raw = f"{rel}:{path.stat().st_mtime_ns}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _extract_pdf(path: Path) -> str:
    import pymupdf4llm

    return pymupdf4llm.to_markdown(str(path)) or ""


def _extract_markitdown(path: Path) -> str:
    from markitdown import MarkItDown

    md = MarkItDown()
    result = md.convert(str(path))
    text = getattr(result, "text_content", None) or ""
    return text or ""


def extract_text(path: Path) -> str:
    """Dispatch by suffix; raises on unsupported type."""
    suf = path.suffix.lower()
    if suf == ".pdf":
        return _extract_pdf(path)
    if suf in {".docx", ".pptx", ".md"}:
        return _extract_markitdown(path)
    raise ValueError(f"Unsupported file type: {suf}")


def iter_documents(
    root: Path,
    *,
    suffixes: Optional[Set[str]] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Walk `root` recursively and yield document dicts for LocalIndexer / indexing.

    Each dict: doc_id, text, metadata { source_path, suffix, filename }.
    Skips unreadable files with a log warning.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    active = suffixes or SUPPORTED_SUFFIXES

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in active:
            continue
        try:
            text = extract_text(path)
        except Exception as e:
            logger.warning("Skipping %s: %s", path, e)
            continue
        text = (text or "").strip()
        if not text:
            logger.warning("Empty content, skipping: %s", path)
            continue
        rel = path
        try:
            rel = path.relative_to(root)
        except ValueError:
            pass
        did = _doc_id_for_path(path, root)
        yield {
            "doc_id": did,
            "text": text,
            "metadata": {
                "source_path": str(path),
                "rel_path": str(rel),
                "filename": path.name,
                "suffix": path.suffix.lower(),
            },
        }


def crawl_to_list(root: Path, **kwargs) -> List[Dict[str, Any]]:
    """Convenience: materialize all documents (use large dirs with care)."""
    return list(iter_documents(root, **kwargs))
