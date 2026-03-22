#!/usr/bin/env python
"""
Query Interface for SelfIndex — Unified Architecture

Usage:
    python query.py -x 1 -y 1 -z 1 -q T -optim 0 --query "machine AND learning"
    python query.py -x 3 -y 1 -z 1 -q T -optim 0 --interactive
    python query.py -x 4 -y 1 -z 1 -q T -optim 0 --query "neural networks" --top-k 20
"""

import sys
import os
import argparse
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from ire_search.core.indexer import create_indexer, get_index_identifier


def load_indexer(x, y, z, optim):
    """Load the built index using the unified factory."""
    identifier = get_index_identifier(x, y, z, optim)
    indexer = create_indexer(x=x, y=y, z=z, optim=optim)
    indexer.load_index(index_id=identifier)
    return indexer, identifier


def run_query(indexer, query, x, q_mode, top_k=10):
    """Run a single query."""
    print(f"\nQuery: {query}")
    mode = 'TAAT' if q_mode == 'T' else 'DAAT'
    print(f"Mode: {mode}")
    print("-" * 70)

    start_time = time.time()
    results = indexer.query(query, mode=mode, top_k=top_k)
    elapsed_ms = (time.time() - start_time) * 1000

    if not results:
        print("No results found.")
        print(f"Query time: {elapsed_ms:.2f} ms")
        return

    print(f"Found {len(results)} results in {elapsed_ms:.2f} ms\n")

    for i, doc_id in enumerate(results[:top_k], 1):
        print(f"  {i}. {doc_id}")

    print("-" * 70)


def interactive_mode(indexer, x, q_mode, top_k=10):
    """Interactive query mode."""
    print(f"\n{'='*70}")
    print("Interactive Query Mode")
    mode = 'TAAT' if q_mode == 'T' else 'DAAT'
    print(f"Algorithm: {mode}")
    print("Type 'quit' to exit")
    print(f"{'='*70}\n")

    while True:
        try:
            query = input("Query> ").strip()
            if not query:
                continue
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            run_query(indexer, query, x, q_mode, top_k)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Query SelfIndex",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python query.py -x 1 -y 1 -z 1 -q T -optim 0 --query "machine AND learning"
  python query.py -x 3 -y 1 -z 1 -q T -optim 0 --interactive
  python query.py -x 4 -y 1 -z 1 -q T -optim 0 --query "neural networks"

Index types:  1=Boolean  2=TF  3=TF-IDF  4=BM25
Datastores:   1=JSON  2=SQLite
Compression:  1=None  2=Elias  3=zlib
        """
    )

    parser.add_argument('-x', '--index-type', type=int, choices=[1, 2, 3, 4], required=True,
                        help='Index type: 1=Boolean, 2=TF, 3=TF-IDF, 4=BM25')
    parser.add_argument('-y', '--datastore', type=int, choices=[1, 2], required=True,
                        help='Datastore: 1=JSON, 2=SQLite')
    parser.add_argument('-z', '--compression', type=int, choices=[1, 2, 3], required=True,
                        help='Compression: 1=None, 2=Elias, 3=zlib')
    parser.add_argument('-q', '--query-mode', choices=['T', 'D'], required=True,
                        help='Query mode: T=TAAT, D=DAAT')
    parser.add_argument('-optim', '--optimization', choices=['0', 'sp', 'th', 'es'], required=True,
                        help='Optimization: 0=None, sp=Skip, th=Threshold, es=EarlyStop')
    parser.add_argument('--query', help='Query string')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--top-k', type=int, default=10, help='Number of results')

    args = parser.parse_args()

    try:
        print("Loading index...")
        indexer, identifier = load_indexer(
            args.index_type, args.datastore, args.compression,
            args.optimization
        )
        print(f"✓ Loaded: {identifier}\n")

        if args.interactive:
            interactive_mode(indexer, args.index_type, args.query_mode, args.top_k)
        elif args.query:
            run_query(indexer, args.query, args.index_type, args.query_mode, args.top_k)
        else:
            print("Error: Provide --query or --interactive")
            sys.exit(1)

    except FileNotFoundError:
        print(f"Error: Index not found. Build it first:")
        print(f"  python build.py -x {args.index_type} -y {args.datastore} "
              f"-z {args.compression} -optim {args.optimization}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
