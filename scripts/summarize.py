#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Print the blog post's tables from the RESULT lines in the logs.

Usage: uv run scripts/summarize.py [logs]
"""

import sys
from pathlib import Path

# e-means rows use the labels of the last training pass, as in the blog post
LABELS = {"lloyd": "assign", "superkmeans": "assign", "emeans": "last_pass",
          "superemeans": "last_pass", "faiss_emeans": "assign"}


def name(algorithm, iterations):
    return {
        "lloyd": f"Lloyd's k-means ({iterations} iters)",
        "superkmeans": f"SuperKMeans ({iterations} iters)",
        "emeans": f"e-means ({iterations} epochs)",
        "superemeans": f"**SuperEMeans ({iterations} epochs)**",
        "faiss_emeans": f"e-means ({iterations} epochs, author's implementation)",
    }[algorithm]


def read_results(log_dir):
    results = {}
    for log in sorted(Path(log_dir).glob("*.log")):
        for line in log.read_text().splitlines():
            if not line.startswith("RESULT "):
                continue
            fields = dict(token.split("=", 1) for token in line.split()[1:])
            if fields["labels"] != LABELS[fields["algorithm"]]:
                continue
            key = (fields["dataset"], fields["algorithm"], int(fields["iterations"]))
            results[key] = fields
    return results


def table(results, dataset, rows, fraction="1pct", label="1%"):
    lines = [
        f"| Algorithm | Runtime (s) | Avg. Recall@100 at {label} of probed centroids "
        f"| Avg. Vectors scanned at {label} of probed centroids |",
        "| --- | --- | --- | --- |",
    ]
    for algorithm, iterations in rows:
        r = results.get((dataset, algorithm, iterations))
        if r is None:
            continue
        vectors = f"{int(r['vectors_' + fraction]):,}".replace(",", " ")
        lines.append(
            f"| {name(algorithm, iterations)} | {r['train_s']} | "
            f"{float(r['recall100_' + fraction]):.3f} | {vectors} |"
        )
    return "\n".join(lines) if len(lines) > 2 else None


def main():
    results = read_results(sys.argv[1] if len(sys.argv) > 1 else "logs")
    main_rows = [("lloyd", 10), ("superkmeans", 10), ("emeans", 2), ("superemeans", 2),
                 ("faiss_emeans", 2)]
    for dataset in ("openai5m", "cohere"):
        sections = [
            (f"{dataset}: 10 iterations / 2 epochs", table(results, dataset, main_rows)),
            (f"{dataset}: e-means at 0.90% of probed centroids (matching vectors scanned)",
             table(results, dataset, [("emeans", 2), ("superemeans", 2), ("faiss_emeans", 2)],
                   "090pct", "0.90%")),
            (f"{dataset}: 5 epochs",
             table(results, dataset, [("emeans", 5), ("superemeans", 5)])),
        ]
        for title, body in sections:
            if body:
                print(f"\n### {title}\n\n{body}")


if __name__ == "__main__":
    main()
