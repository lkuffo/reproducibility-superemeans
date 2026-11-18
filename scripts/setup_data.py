#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "pyarrow",
#   "datasets",
#   "huggingface_hub",
#   "faiss-cpu==1.16.0",
# ]
# ///
"""Download a dataset of the blog post and compute its ground truth.

Usage: uv run scripts/setup_data.py <openai5m|cohere> [--data-dir data]

Writes, inside --data-dir:
  data_<dataset>.bin       vectors: raw little-endian float32, row-major, no header
  data_<dataset>_test.bin  queries: same format
  gt_<dataset>.json        {"query index": [ids of its 100 nearest neighbors]}, first 1,000 queries
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

DATASETS = {
    "openai5m": (5_000_000, 1536),
    "cohere": (10_000_000, 1024),
}
N_QUERIES = 1000
KNN = 100


def has_size(path, n_floats):
    return path.exists() and path.stat().st_size == n_floats * 4


def download_openai5m(data_dir, cache_dir):
    """5M OpenAI text-embedding-3-large vectors (1536 dims) from VectorDBBench, in fbin format:
    a uint32 count and a uint32 dimensionality, followed by the float32 rows."""
    from huggingface_hub import hf_hub_download

    files = (("fbin/base.fbin", "data_openai5m.bin"), ("fbin/queries.fbin", "data_openai5m_test.bin"))
    for remote, name in files:
        print(f"Downloading makneeee/openai_large_5m/{remote} ...")
        local = hf_hub_download(
            repo_id="makneeee/openai_large_5m",
            filename=remote,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        with open(local, "rb") as f_in, open(data_dir / name, "wb") as f_out:
            n, d = np.frombuffer(f_in.read(8), dtype=np.uint32)
            shutil.copyfileobj(f_in, f_out, length=64 << 20)
        print(f"  {name}: {n:,} x {d}")
        shutil.rmtree(cache_dir, ignore_errors=True)


def download_cohere(data_dir, cache_dir):
    """The first 10M MS MARCO v2.1 passages embedded with Cohere Embed v3 (1024 dims), read from the
    passage shards in order, and the first 1,000 queries of the test split."""
    import pyarrow.parquet as pq
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    repo_id = "Cohere/msmarco-v2.1-embed-english-v3"
    n_target, d = DATASETS["cohere"]
    written = 0
    with open(data_dir / "data_cohere.bin", "wb") as f_out:
        for shard in range(32):
            remote = f"passages_parquet/msmarco_v2.1_doc_segmented_{shard:02d}.parquet"
            print(f"Downloading {repo_id}/{remote} ...")
            local = hf_hub_download(
                repo_id=repo_id, filename=remote, repo_type="dataset", cache_dir=cache_dir
            )
            for batch in pq.ParquetFile(local).iter_batches(batch_size=100_000, columns=["emb"]):
                values = batch.column("emb").flatten().to_numpy()
                vectors = values.astype(np.float32, copy=False).reshape(-1, d)
                take = min(len(vectors), n_target - written)
                vectors[:take].tofile(f_out)
                written += take
                if written == n_target:
                    break
            print(f"  {written:,} / {n_target:,} passages")
            shutil.rmtree(cache_dir, ignore_errors=True)
            if written == n_target:
                break

    print(f"Downloading {repo_id} queries ...")
    queries = load_dataset(repo_id, "queries", split="test", cache_dir=str(cache_dir))
    embeddings = np.asarray(queries[:N_QUERIES]["emb"], dtype=np.float32)
    embeddings.tofile(data_dir / "data_cohere_test.bin")
    print(f"  data_cohere_test.bin: {len(embeddings):,} x {embeddings.shape[1]}")
    shutil.rmtree(cache_dir, ignore_errors=True)


def compute_ground_truth(data_dir, dataset):
    """Exact (brute-force L2) top-100 neighbors of the first 1,000 queries."""
    import faiss

    n, d = DATASETS[dataset]
    print(f"Computing ground truth for {dataset} ...")
    base = np.fromfile(data_dir / f"data_{dataset}.bin", dtype=np.float32).reshape(n, d)
    queries = np.fromfile(data_dir / f"data_{dataset}_test.bin", dtype=np.float32).reshape(-1, d)
    queries = queries[:N_QUERIES]
    _, ids = faiss.knn(queries, base, KNN)
    with open(data_dir / f"gt_{dataset}.json", "w") as f:
        json.dump({str(i): ids[i].tolist() for i in range(len(queries))}, f)
    print(f"  gt_{dataset}.json: {len(queries):,} queries x top-{KNN}")


def main():
    parser = argparse.ArgumentParser(description="Download a dataset and compute its ground truth")
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--data-dir", default="data", type=Path)
    args = parser.parse_args()

    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = data_dir / ".hf_cache"
    n, d = DATASETS[args.dataset]

    if has_size(data_dir / f"data_{args.dataset}.bin", n * d):
        print(f"data_{args.dataset}.bin already present, skipping download")
    elif args.dataset == "openai5m":
        download_openai5m(data_dir, cache_dir)
    else:
        download_cohere(data_dir, cache_dir)

    if not (data_dir / f"gt_{args.dataset}.json").exists():
        compute_ground_truth(data_dir, args.dataset)


if __name__ == "__main__":
    main()
