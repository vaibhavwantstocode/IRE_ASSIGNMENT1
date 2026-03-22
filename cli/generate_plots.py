#!/usr/bin/env python
"""
Generate evaluation plots from stored results.

Reads JSON result files from results/ and produces comparison charts
for latency, throughput, and memory across index configurations.

Usage:
    python generate_plots.py                     # All results
    python generate_plots.py --filter "i3"       # Only TF-IDF
    python generate_plots.py --output plots/     # Custom output dir
"""

import sys
import os
import json
import glob
import argparse
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")


# Display names for configurations
SCORER_NAMES = {1: 'Boolean', 2: 'TF', 3: 'TF-IDF', 4: 'BM25'}
STORAGE_NAMES = {1: 'JSON', 2: 'SQLite'}
COMPRESS_NAMES = {1: 'None', 2: 'Elias', 3: 'zlib'}
COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4']


def load_results(results_dir='results', filter_str=None):
    """Load all evaluation result JSON files."""
    pattern = os.path.join(results_dir, 'eval_*.json')
    files = glob.glob(pattern)
    
    if not files:
        print(f"No result files found in {results_dir}/")
        return []
    
    results = []
    for f in sorted(files):
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                if filter_str and filter_str not in data.get('identifier', ''):
                    continue
                data['_file'] = os.path.basename(f)
                results.append(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Skipping {f}: {e}")
    
    print(f"Loaded {len(results)} result files")
    return results


def plot_latency_comparison(results, output_dir):
    """Bar chart comparing latency percentiles across configurations."""
    if not results:
        return
    
    labels = [r['identifier'] for r in results]
    p50 = [r['artifact_A_latency']['p50_ms'] for r in results]
    p95 = [r['artifact_A_latency']['p95_ms'] for r in results]
    p99 = [r['artifact_A_latency']['p99_ms'] for r in results]
    
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 2), 6))
    ax.bar(x - width, p50, width, label='P50', color=COLORS[0], alpha=0.8)
    ax.bar(x, p95, width, label='P95', color=COLORS[1], alpha=0.8)
    ax.bar(x + width, p99, width, label='P99', color=COLORS[2], alpha=0.8)
    
    ax.set_xlabel('Index Configuration')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Query Latency by Configuration (P50/P95/P99)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'latency_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_throughput_comparison(results, output_dir):
    """Bar chart comparing QPS across configurations."""
    if not results:
        return
    
    labels = [r['identifier'] for r in results]
    qps = [r['artifact_B_throughput']['queries_per_second'] for r in results]
    
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 2), 6))
    bars = ax.bar(labels, qps, color=COLORS[:len(labels)], alpha=0.8)
    
    for bar, val in zip(bars, qps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Index Configuration')
    ax.set_ylabel('Queries per Second')
    ax.set_title('Throughput Comparison (QPS)')
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'throughput_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_memory_comparison(results, output_dir):
    """Bar chart comparing disk and RAM usage."""
    if not results:
        return
    
    labels = [r['identifier'] for r in results]
    disk = [r['artifact_C_memory']['disk_mb'] for r in results]
    ram = [r['artifact_C_memory']['ram_gb'] * 1024 for r in results]  # Convert to MB

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 2), 6))
    ax.bar(x - width/2, disk, width, label='Disk (MB)', color=COLORS[0], alpha=0.8)
    ax.bar(x + width/2, ram, width, label='RAM (MB)', color=COLORS[3], alpha=0.8)

    ax.set_xlabel('Index Configuration')
    ax.set_ylabel('Size (MB)')
    ax.set_title('Memory Footprint: Disk vs RAM')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'memory_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_summary_table(results, output_dir):
    """Generate a text summary table."""
    if not results:
        return
    
    lines = [
        f"{'Config':<30} {'P50 (ms)':<10} {'P95 (ms)':<10} {'QPS':<10} {'Disk (MB)':<10} {'RAM (GB)':<10}",
        '-' * 80,
    ]
    
    for r in results:
        lines.append(
            f"{r['identifier']:<30} "
            f"{r['artifact_A_latency']['p50_ms']:<10.3f} "
            f"{r['artifact_A_latency']['p95_ms']:<10.3f} "
            f"{r['artifact_B_throughput']['queries_per_second']:<10.1f} "
            f"{r['artifact_C_memory']['disk_mb']:<10.2f} "
            f"{r['artifact_C_memory']['ram_gb']:<10.4f}"
        )
    
    table = '\n'.join(lines)
    print(f"\n{table}")
    
    path = os.path.join(output_dir, 'summary.txt')
    with open(path, 'w') as f:
        f.write(table + '\n')
    print(f"\n  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation plots")
    parser.add_argument('--results-dir', default='results', help='Results directory')
    parser.add_argument('--output', default='plots', help='Output directory for plots')
    parser.add_argument('--filter', default=None, help='Filter by identifier substring')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"\nLoading results from: {args.results_dir}")
    results = load_results(args.results_dir, args.filter)

    if not results:
        print("No results to plot. Run evaluate.py first.")
        sys.exit(1)

    print(f"\nGenerating plots...")
    if HAS_MATPLOTLIB:
        plot_latency_comparison(results, args.output)
        plot_throughput_comparison(results, args.output)
        plot_memory_comparison(results, args.output)
    else:
        print("  Skipping charts (matplotlib not installed)")

    plot_summary_table(results, args.output)
    print(f"\n✓ All plots saved to: {args.output}/")


if __name__ == '__main__':
    main()
