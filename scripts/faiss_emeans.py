#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "faiss-cpu==1.16.0",
# ]
# ///
"""e-means with the authors' implementation (faiss.contrib.e_means).

Usage: uv run scripts/faiss_emeans.py <data_dir> <openai5m|cohere> <n_clusters> [epochs=2]
"""

import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from faiss.contrib.e_means import NumpyEmeans

DATASETS = {
    "openai5m": (5_000_000, 1536),
    "cohere": (10_000_000, 1024),
}
N_QUERIES = 1000
SEED = 42
ABSOLUTE_EXPLORE_COUNTS = [1, 2]
EXPLORE_FRACTIONS = [
    0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
    0.0100, 0.0125, 0.0150, 0.0175, 0.0200, 0.0225, 0.0250, 0.0275,
    0.0300, 0.0325, 0.0350, 0.0375, 0.0400, 0.0425, 0.0450, 0.0475, 0.0500,
    0.1,
]


def compute_recall(ground_truth, assignments, queries, centroids, n_clusters, knn):
    """Probe the centroids closest to each query; recall = fraction of the query's true top-knn
    assigned to them, vectors = points held by them. Same as SuperKMeans' bench_utils.py."""
    cluster_sizes = np.bincount(assignments, minlength=n_clusters)
    distances = (
        np.sum(queries**2, axis=1, keepdims=True)
        + np.sum(centroids**2, axis=1, keepdims=True).T
        - 2 * queries @ centroids.T
    )
    order = np.argsort(distances, axis=1)
    configs = [(min(c, n_clusters), min(c, n_clusters) / n_clusters) for c in ABSOLUTE_EXPLORE_COUNTS]
    configs += [(max(1, int(n_clusters * frac)), frac) for frac in EXPLORE_FRACTIONS]
    results = []
    for n_probe, fraction in configs:
        recalls, total_vectors = [], 0
        for query in range(len(queries)):
            neighbors = ground_truth.get(str(query))
            if neighbors is None:
                continue
            probed = order[query, :n_probe]
            total_vectors += np.sum(cluster_sizes[probed])
            probed_set = set(probed)
            neighbors = neighbors[:knn]
            found = sum(assignments[v] in probed_set for v in neighbors)
            recalls.append(found / len(neighbors))
        std = np.std(recalls, ddof=1) if len(recalls) > 1 else 0.0
        results.append((n_probe, fraction, np.mean(recalls), std, total_vectors / len(queries)))
    return results


def print_recall(results, knn):
    print(f"\n--- Recall@{knn} ---")
    for n_probe, fraction, recall, std, vectors in results:
        print(
            f"Recall@{n_probe:4d} ({fraction * 100:5.2f}% centroids, {vectors:8.0f} avg vectors): "
            f"{recall:.4f} ± {std:.4f}"
        )


def at_fraction(results, fraction):
    return next(r for r in results if abs(r[1] - fraction) < 1e-9)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    data_dir, dataset, n_clusters = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    n, d = DATASETS[dataset]
    print(f"=== faiss_emeans | {dataset} (n={n}, d={d}) | k={n_clusters} | {epochs} epochs ===")
    print(f"faiss {faiss.__version__}, {faiss.omp_get_max_threads()} threads")

    data = np.fromfile(data_dir / f"data_{dataset}.bin", dtype=np.float32).reshape(n, d)
    queries = np.fromfile(data_dir / f"data_{dataset}_test.bin", dtype=np.float32)
    queries = queries[: N_QUERIES * d].reshape(N_QUERIES, d)
    with open(data_dir / f"gt_{dataset}.json") as f:
        ground_truth = json.load(f)

    emeans = NumpyEmeans(d, n_clusters, nepochs=epochs, seed=SEED, spherical=False)
    start = time.perf_counter()
    centroids, _ = emeans.train(data)
    train_seconds = time.perf_counter() - start
    print(f"\nTraining completed in {train_seconds:.1f} s")

    _, assignments = emeans.assign(data)
    recall_at_10 = compute_recall(ground_truth, assignments, queries, centroids, n_clusters, 10)
    recall_at_100 = compute_recall(ground_truth, assignments, queries, centroids, n_clusters, 100)
    print_recall(recall_at_10, 10)
    print_recall(recall_at_100, 100)

    p100, p090 = at_fraction(recall_at_100, 0.01), at_fraction(recall_at_100, 0.009)
    print(
        f"RESULT algorithm=faiss_emeans dataset={dataset} n_clusters={n_clusters} "
        f"iterations={epochs} labels=assign train_s={train_seconds:.1f} wcss=nan "
        f"recall100_1pct={p100[2]:.4f} vectors_1pct={p100[4]:.0f} "
        f"recall100_090pct={p090[2]:.4f} vectors_090pct={p090[4]:.0f}"
    )


if __name__ == "__main__":
    main()
