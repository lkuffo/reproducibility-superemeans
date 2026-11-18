# reproducibility-superemeans
Code to reproduce the experiments of the blog post [SuperEMeans: Making e-means 3× faster](https://www.lkuffo.com/superemeans/).


- **Data**: **OpenAI** (5M vectors, 1536 dims, 20,000 centroids) and **Cohere** (10M vectors, 1024 dims, 40,000 centroids). 
- **Metrics recorded**: runtime, recall@100 and vectors scanned at 1% of probed centroids for
  - Lloyd's k-means (10 iterations)
  - SuperKMeans (10 iterations)
  - e-means (2 epochs)
  - SuperEMeans (2 epochs)
  - e-means, authors' implementation (2 epochs)
- **Matching vectors scanned**: recall@100 of the e-means variants at 0.90% of probed centroids.
- **More epochs**: e-means and SuperEMeans with 5 epochs.

## Quick start
```sh
./run_all.sh
```
It downloads both datasets, computes their ground truth, builds the benchmarks, runs every experiment and prints the tables of the post. Each run writes a log to `logs/`. If interrupted, run it again: finished experiments are skipped.

Run a single dataset with `DATASETS=openai5m ./run_all.sh`.

## Requirements
- Linux (x86 or ARM) with Clang 17+ or GCC 13+, CMake 3.26+, git and [uv](https://docs.astral.sh/uv/).
- **128 GB of RAM**: training on Cohere holds the dataset twice (~82 GB).
- **~75 GB of disk** for the data, plus up to ~30 GB more while downloading.

The post used an AWS `r8a.4xlarge` (16 Zen 5 cores, 128 GB). All experiments take ~3 hours there, plus the downloads.

## Step by step

### 1. Downloading Data
```sh
uv run scripts/setup_data.py openai5m
uv run scripts/setup_data.py cohere
```
Each call writes three files to `data/` (change it with `--data-dir`):
- `data_<dataset>.bin`: the vectors, as raw float32 (row-major, no header)
- `data_<dataset>_test.bin`: the queries, same format
- `gt_<dataset>.json`: the exact top-100 neighbors of the first 1,000 queries

Sources: [makneeee/openai_large_5m](https://huggingface.co/datasets/makneeee/openai_large_5m) and the first 10M passages of [Cohere/msmarco-v2.1-embed-english-v3](https://huggingface.co/datasets/Cohere/msmarco-v2.1-embed-english-v3).

### 2. Building
```sh
cmake -B build
cmake --build build -j
```


### 3. Running
```sh
./build/kmeans data openai5m 20000 false 10   # Lloyd's k-means, 10 iterations
./build/kmeans data openai5m 20000 true 10    # SuperKMeans, 10 iterations
./build/emeans data openai5m 20000 false 2    # e-means, 2 epochs
./build/emeans data openai5m 20000 true 2     # SuperEMeans, 2 epochs
uv run scripts/faiss_emeans.py data openai5m 20000 2   # e-means, authors' implementation
```
Arguments: `<data_dir> <dataset> <n_clusters> <pruning> [iterations or epochs]`. For Cohere, use `cohere 40000`. For the 5-epoch experiments, set the last argument to `5`.

### 4. Generating Tables
```sh
uv run scripts/summarize.py logs
```

<details>

<summary>How we measure</summary>

- **Runtime**: clustering only; loading the data and the final assignment are excluded. All cores are used.
- **Recall@100**: for each of the 1,000 queries, we probe the 1% (or 0.90%) of centroids closest to it and count how many of its true top-100 neighbors are assigned to those centroids. We report the average over queries.
- **Vectors scanned**: the number of points in the probed clusters, averaged over queries.
- **Assignments**: the k-means rows use SuperKMeans' `AssignTrainingPoints`; the e-means rows use the labels of the last training pass (the e-means runs also print the `AssignTrainingPoints` numbers).
- **Authors' implementation**: `faiss.contrib.e_means` from the `faiss-cpu==1.16.0` wheel. The post used FAISS built from source with BLIS, so its runtime here may differ.

</details>
