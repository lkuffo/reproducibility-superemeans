#!/usr/bin/env bash
# Reproduces the experiments of https://www.lkuffo.com/superemeans/
# Usage: ./run_all.sh            (both datasets)
#        DATASETS=openai5m ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

DATASETS="${DATASETS:-openai5m cohere}"
DATA_DIR="${DATA_DIR:-data}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$DATA_DIR" "$LOG_DIR"

n_clusters() {
    case "$1" in
        openai5m) echo 20000 ;;
        cohere) echo 40000 ;;
    esac
}

# Runs one experiment unless its log already holds a result (so an interrupted run resumes)
run() {
    local log="$LOG_DIR/$1.log"
    shift
    if [[ -f "$log" ]] && grep -q '^RESULT' "$log"; then
        echo "[skip] $log"
        return
    fi
    echo "[run]  $log"
    "$@" > "$log" 2>&1
}

for dataset in $DATASETS; do
    uv run scripts/setup_data.py "$dataset" --data-dir "$DATA_DIR"
done

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

for dataset in $DATASETS; do
    k="$(n_clusters "$dataset")"
    run "${dataset}_lloyd"                ./build/kmeans "$DATA_DIR" "$dataset" "$k" false 10
    run "${dataset}_superkmeans"          ./build/kmeans "$DATA_DIR" "$dataset" "$k" true 10
    run "${dataset}_emeans"               ./build/emeans "$DATA_DIR" "$dataset" "$k" false 2
    run "${dataset}_superemeans"          ./build/emeans "$DATA_DIR" "$dataset" "$k" true 2
    run "${dataset}_faiss_emeans"         uv run scripts/faiss_emeans.py "$DATA_DIR" "$dataset" "$k" 2
    run "${dataset}_emeans_5epochs"       ./build/emeans "$DATA_DIR" "$dataset" "$k" false 5
    run "${dataset}_superemeans_5epochs"  ./build/emeans "$DATA_DIR" "$dataset" "$k" true 5
done

uv run scripts/summarize.py "$LOG_DIR"
