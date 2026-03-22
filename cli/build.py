#!/usr/bin/env python
"""
Build Script for SelfIndex — Unified Architecture

ALL 24 COMBINATIONS SUPPORTED:
- 4 index types (x): Boolean (1), TF (2), TF-IDF (3), BM25 (4)
- 2 datastores (y): JSON (1), SQLite (2)
- 3 compression methods (z): None (1), Elias (2), zlib (3)

Format: SelfIndex_i{x}d{y}c{z}o{optim}

Usage:
    python build.py -x 1 -y 1 -z 1 -optim 0           # Boolean + JSON + No compression
    python build.py -x 3 -y 2 -z 2 -optim 0           # TF-IDF + SQLite + Elias
    python build.py -x 4 -y 1 -z 1 -optim 0           # BM25 + JSON
    python build.py -x 2 -y 1 -z 3 -optim 0 --limit 100  # TF + zlib (test subset)
"""

import sys
import os
import argparse

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from ire_search.core.indexer import create_indexer, get_index_identifier
from ire_search.io.data_loader import load_documents, load_documents_preprocessed


# Display maps
INDEX_MAP = {1: 'Boolean', 2: 'TF (Term Frequency)', 3: 'TF-IDF', 4: 'BM25'}
DATASTORE_MAP = {1: 'JSON', 2: 'SQLite'}
COMPRESS_MAP = {1: 'None', 2: 'Elias Gamma/Delta', 3: 'zlib'}
OPTIM_MAP = {'0': 'None', 'sp': 'Skip Pointers', 'th': 'Thresholding', 'es': 'Early Stopping'}


def build_index(x, y, z, optim, limit=None, force=False):
    """
    Build index with specified parameters.

    Args:
        x: Index type (1=Boolean, 2=TF, 3=TF-IDF, 4=BM25)
        y: Datastore (1=JSON, 2=SQLite)
        z: Compression (1=None, 2=Elias, 3=zlib)
        optim: Optimization ('0'=None, 'sp'=Skip, 'th'=Threshold, 'es'=EarlyStop)
        limit: Limit documents per source for testing
        force: Skip rebuild confirmation prompts
    """
    print(f"\n{'='*70}")
    print(f"Building SelfIndex")
    print(f"  Index Type:    {INDEX_MAP.get(x, '?')} (x={x})")
    print(f"  Datastore:     {DATASTORE_MAP.get(y, '?')} (y={y})")
    print(f"  Compression:   {COMPRESS_MAP.get(z, '?')} (z={z})")
    print(f"  Optimization:  {OPTIM_MAP.get(optim, '?')} (o={optim})")
    print(f"{'='*70}")

    identifier = get_index_identifier(x, y, z, optim)
    print(f"\nIdentifier: {identifier}")

    # Check if already exists
    ext = '.db' if y == 2 else '.json'
    index_path = os.path.join('indices', f"{identifier}{ext}")
    if os.path.exists(index_path):
        if not force:
            print(f"\n⚠ Index already exists: {index_path}")
            response = input("Rebuild? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
        else:
            print(f"⚠ Index exists: {index_path} — rebuilding (--force)")

    # Create indexer using the unified factory
    indexer = create_indexer(x=x, y=y, z=z, optim=optim)

    # Load documents
    print("\nLoading documents...")
    if limit:
        print(f"⚠ TEST MODE: Limited to {limit:,} documents per source")

    try:
        doc_limit = limit if limit else 50000
        documents = load_documents_preprocessed(limit=doc_limit)
    except FileNotFoundError:
        print("⚠ Preprocessed cache not found, using raw documents...")
        if limit:
            documents = load_documents(limit_per_source=limit)
        else:
            documents = load_documents()

    # Build
    indexer.create_index(index_id=identifier, documents=documents, limit=limit)

    print(f"\n✓ Index built: {identifier}")
    print(f"  Location: {index_path}")

    stats = indexer.get_stats()
    print(f"\n  Documents:  {stats['num_documents']:,}")
    print(f"  Terms:      {stats['num_terms']:,}")
    print(f"  Scorer:     {stats['scorer_type']}")


def main():
    parser = argparse.ArgumentParser(
        description="Build SelfIndex with pluggable scoring, storage, and compression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build.py -x 1 -y 1 -z 1 -optim 0     # Boolean + JSON
  python build.py -x 3 -y 1 -z 1 -optim 0     # TF-IDF + JSON
  python build.py -x 4 -y 1 -z 1 -optim 0     # BM25 + JSON
  python build.py -x 3 -y 2 -z 2 -optim 0     # TF-IDF + SQLite + Elias
  python build.py -x 2 -y 1 -z 3 -optim 0 --limit 100  # Quick test

Format: SelfIndex_i{x}d{y}c{z}o{optim}
  x: 1=Boolean, 2=TF, 3=TF-IDF, 4=BM25
  y: 1=JSON, 2=SQLite
  z: 1=None, 2=Elias, 3=zlib
  o: 0=None, sp=SkipPointers, th=Threshold, es=EarlyStop
        """
    )

    parser.add_argument('-x', '--index-type', type=int, choices=[1, 2, 3, 4], required=True,
                        help='Index type: 1=Boolean, 2=TF, 3=TF-IDF, 4=BM25')
    parser.add_argument('-y', '--datastore', type=int, choices=[1, 2], required=True,
                        help='Datastore: 1=JSON, 2=SQLite')
    parser.add_argument('-z', '--compression', type=int, choices=[1, 2, 3], required=True,
                        help='Compression: 1=None, 2=Elias, 3=zlib')
    parser.add_argument('-optim', '--optimization', choices=['0', 'sp', 'th', 'es'], required=True,
                        help='Optimization: 0=None, sp=Skip, th=Threshold, es=EarlyStop')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit documents per source (for testing)')
    parser.add_argument('--force', action='store_true',
                        help='Skip rebuild confirmation')

    args = parser.parse_args()

    try:
        build_index(
            x=args.index_type, y=args.datastore, z=args.compression,
            optim=args.optimization, limit=args.limit, force=args.force
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
