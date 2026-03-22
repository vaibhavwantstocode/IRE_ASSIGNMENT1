# Evaluation Guide

## Running Evaluations

```bash
# Evaluate TF-IDF with JSON
python cli/evaluate.py -x 3 -y 1 -z 1 -optim 0

# Or: python main.py evaluate -x 3 -y 1 -z 1 -optim 0

# Evaluate BM25 with DAAT
python cli/evaluate.py -x 4 -y 1 -z 1 -optim 0 -q D

# Custom query file
python cli/evaluate.py -x 3 -y 1 -z 1 -optim 0 --queries my_queries.txt
```

`cli/evaluate.py` uses `MetricsCollector` ([evaluation/metrics_collector.py](../src/ire_search/evaluation/metrics_collector.py)) per query and writes **`collector_aggregates`** plus disk/RAM stats into `results/eval_*.json`.

## Artifacts

### Artifact A: Latency
Measures per-query response time with percentiles.

| Metric | Description |
|--------|-------------|
| P50 | Median latency |
| P95 | 95th percentile (tail latency) |
| P99 | 99th percentile (worst case) |
| Average | Mean over all queries |

### Artifact B: Throughput
Queries per second (QPS) = `total_queries / total_time`.

### Artifact C: Memory
- **Disk**: Size of index file on disk
- **RAM**: Process RSS after loading index
- **Collector**: Per-query RAM deltas / peaks from `MetricsCollector` (see JSON `collector_aggregates`)

## Generating Plots

```bash
python cli/generate_plots.py                    # All results
python cli/generate_plots.py --filter "i3"      # TF-IDF only
python cli/generate_plots.py --output my_plots/ # Custom output dir
```

Produces:
- `latency_comparison.png` — P50/P95/P99 bar chart
- `throughput_comparison.png` — QPS comparison
- `memory_comparison.png` — Disk vs RAM
- `summary.txt` — Text table

## Interpreting Results

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| P99 >> P95 | Outlier queries (long terms) | Check query file |
| Low QPS | Large index in RAM | Use SQLite (`y=2`) |
| High disk usage | No compression | Use zlib (`z=3`) |
| Boolean slow | Complex nested queries | Simplify expressions |
