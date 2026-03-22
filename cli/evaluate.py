#!/usr/bin/env python
"""
Evaluation Script for SelfIndex — Unified Architecture

Measures Artifacts A (Latency), B (Throughput), C (Memory).

Usage:
    python evaluate.py -x 3 -y 1 -z 1 -optim 0
    python evaluate.py -x 4 -y 1 -z 1 -optim 0 -q D --queries custom.txt
"""

import sys
import os
import argparse
import json
import time
import psutil
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from ire_search.core.indexer import create_indexer, get_index_identifier
from ire_search.evaluation.metrics_collector import MetricsCollector


def load_indexer(x, y, z, optim):
    """Load index using the unified factory."""
    identifier = get_index_identifier(x, y, z, optim)
    indexer = create_indexer(x=x, y=y, z=z, optim=optim)
    indexer.load_index(index_id=identifier)
    return indexer, identifier


def evaluate_index(x, y, z, optim, query_mode='TAAT', query_file='queries/test_queries.txt', top_k=10):
    """
    Evaluate index performance.

    Artifact A: Latency (p50/p90/p95/p99)
    Artifact B: Throughput (QPS)
    Artifact C: Memory (disk + RAM)
    """
    identifier = get_index_identifier(x, y, z, optim)

    print(f"\n{'='*70}")
    print(f"Evaluating: {identifier}")
    print(f"Query Mode: {query_mode}")
    print(f"{'='*70}\n")

    # Load index
    indexer, identifier = load_indexer(x, y, z, optim)
    print(f"✓ Loaded: {identifier}")

    # Load queries
    print(f"\nLoading queries from: {query_file}")
    with open(query_file, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip()]
    print(f"✓ {len(queries)} queries loaded")

    # Warmup
    print("\nWarming up...")
    for query in queries[:5]:
        try:
            indexer.query(query, mode=query_mode, top_k=top_k)
        except Exception:
            pass
    print("✓ Warmup complete")

    # ===== ARTIFACT A: LATENCY (via MetricsCollector) =====
    print(f"\n{'='*70}")
    print("ARTIFACT A: Latency")
    print(f"{'='*70}")

    collector = MetricsCollector()
    latencies = []
    for i, query in enumerate(queries, 1):
        collector.start_query()
        try:
            results = indexer.query(query, mode=query_mode, top_k=top_k)
            collector.end_query(results)
            latencies.append(collector.query_times[-1])
        except Exception as e:
            collector.cancel_query()
            print(f"  Warning: Query {i} failed: {e}")

        if i % 10 == 0:
            print(f"  Processed {i}/{len(queries)} queries...")

    if not latencies:
        raise RuntimeError("No successful queries — cannot compute latency stats.")

    artifact_A = {
        'average_ms': float(np.mean(latencies)),
        'median_ms': float(np.median(latencies)),
        'p50_ms': float(np.percentile(latencies, 50)),
        'p90_ms': float(np.percentile(latencies, 90)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies)),
        'std_ms': float(np.std(latencies)),
    }

    print(f"\n✓ Latency:")
    print(f"  Average: {artifact_A['average_ms']:.4f} ms")
    print(f"  P50:     {artifact_A['p50_ms']:.4f} ms")
    print(f"  P95:     {artifact_A['p95_ms']:.4f} ms")
    print(f"  P99:     {artifact_A['p99_ms']:.4f} ms")

    # ===== ARTIFACT B: THROUGHPUT =====
    print(f"\n{'='*70}")
    print("ARTIFACT B: Throughput")
    print(f"{'='*70}")

    total_time = sum(latencies) / 1000
    n_ok = len(latencies)
    qps = n_ok / total_time if total_time > 0 else 0

    artifact_B = {
        'queries_per_second': float(qps),
        'total_queries': n_ok,
        'total_time_seconds': float(total_time),
    }

    print(f"\n✓ Throughput: {artifact_B['queries_per_second']:.2f} QPS ({n_ok} successful queries)")

    # ===== ARTIFACT C: MEMORY =====
    print(f"\n{'='*70}")
    print("ARTIFACT C: Memory Footprint")
    print(f"{'='*70}")

    ext = '.db' if y == 2 else '.json'
    index_path = os.path.join('indices', f"{identifier}{ext}")
    disk_bytes = os.path.getsize(index_path) if os.path.exists(index_path) else 0
    ram_bytes = psutil.Process().memory_info().rss

    collector_stats = collector.get_statistics()
    artifact_C = {
        'disk_mb': disk_bytes / (1024 * 1024),
        'ram_gb': ram_bytes / (1024 ** 3),
        'disk_bytes': disk_bytes,
        'process_rss_gb': ram_bytes / (1024 ** 3),
        'collector_memory': collector_stats.get('artefact_c_memory', {}),
    }

    print(f"\n✓ Disk:  {artifact_C['disk_mb']:.2f} MB")
    print(f"  RAM:   {artifact_C['ram_gb']:.4f} GB (process RSS)")
    if collector_stats.get('artefact_c_memory'):
        cm = collector_stats['artefact_c_memory']
        print(f"  Query ΔRAM (collector): mean {cm.get('mean_delta_mb', 0):.4f} MB")

    # ===== SAVE RESULTS =====
    os.makedirs('results', exist_ok=True)
    result_file = f"results/eval_{identifier}_q{query_mode}.json"

    results = {
        'identifier': identifier,
        'configuration': {
            'index_type': x, 'datastore': y, 'compression': z,
            'query_mode': query_mode, 'optimization': optim,
        },
        'artifact_A_latency': artifact_A,
        'artifact_B_throughput': artifact_B,
        'artifact_C_memory': artifact_C,
        'collector_aggregates': collector_stats,
        'evaluation_params': {
            'query_file': query_file,
            'num_queries': len(queries),
            'successful_queries': n_ok,
            'top_k': top_k,
        },
    }

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"✓ Results saved to: {result_file}")
    print(f"{'='*70}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SelfIndex performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py -x 3 -y 1 -z 1 -optim 0
  python evaluate.py -x 4 -y 1 -z 1 -optim 0 -q D
  python evaluate.py -x 3 -y 2 -z 2 -optim 0
        """
    )

    parser.add_argument('-x', '--index-type', type=int, choices=[1, 2, 3, 4], required=True,
                        help='1=Boolean, 2=TF, 3=TF-IDF, 4=BM25')
    parser.add_argument('-y', '--datastore', type=int, choices=[1, 2], required=True,
                        help='1=JSON, 2=SQLite')
    parser.add_argument('-z', '--compression', type=int, choices=[1, 2, 3], required=True,
                        help='1=None, 2=Elias, 3=zlib')
    parser.add_argument('-q', '--query-mode', choices=['T', 'D'], default='T',
                        help='T=TAAT, D=DAAT')
    parser.add_argument('-optim', '--optimization', choices=['0', 'sp', 'th', 'es'], required=True,
                        help='0=None, sp=Skip, th=Threshold, es=EarlyStop')
    parser.add_argument('--queries', default='queries/test_queries.txt',
                        help='Query file')
    parser.add_argument('--top-k', type=int, default=10, help='Results count')

    args = parser.parse_args()
    query_mode = 'TAAT' if args.query_mode == 'T' else 'DAAT'

    try:
        evaluate_index(
            x=args.index_type, y=args.datastore, z=args.compression,
            optim=args.optimization, query_mode=query_mode,
            query_file=args.queries, top_k=args.top_k
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
