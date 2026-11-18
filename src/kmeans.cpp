#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME = true
#endif

#include <cstdio>
#include <string>
#include <vector>

#include "bench_utils.h"
#include "superkmeans/superkmeans.h"

// Lloyd's k-means (pruning=false) and SuperKMeans (pruning=true).
// Usage: kmeans <data_dir> <dataset> <n_clusters> <pruning: true|false> [iterations=10]
int main(int argc, char* argv[]) {
    if (argc < 5) {
        fprintf(
            stderr,
            "Usage: %s <data_dir> <dataset> <n_clusters> <pruning: true|false> [iterations=10]\n",
            argv[0]
        );
        return 1;
    }
    const std::string data_dir = argv[1], dataset = argv[2];
    const size_t n_clusters = std::stoul(argv[3]);
    const bool pruning = std::string(argv[4]) == "true";
    const uint32_t iterations = argc > 5 ? static_cast<uint32_t>(std::stoul(argv[5])) : 10;
    const auto [n, d] = repro::GetDataset(dataset);
    const std::string algorithm = pruning ? "superkmeans" : "lloyd";

    printf("=== %s | %s (n=%zu, d=%zu) | k=%zu | %u iterations ===\n", algorithm.c_str(),
           dataset.c_str(), n, d, n_clusters, iterations);
    auto data = repro::ReadFloats(repro::DataPath(data_dir, dataset), n * d);
    auto queries = repro::ReadFloats(repro::QueriesPath(data_dir, dataset), repro::N_QUERIES * d);
    const auto ground_truth = repro::ReadGroundTruth(repro::GroundTruthPath(data_dir, dataset));

    skmeans::SuperKMeansConfig config;
    config.iters = iterations;
    config.verbose = true;
    config.n_threads = skmeans::ResolveNumThreads(0);
    config.unrotate_centroids = true;
    config.early_termination = false;
    config.sampling_fraction = 1.0f;
    config.use_blas_only = !pruning;
    config.tol = 1e-3f;

    auto kmeans = skmeans::SuperKMeans(n_clusters, d, config);
    repro::Timer timer;
    timer.Tic();
    std::vector<float> centroids = kmeans.Train(data.get(), n);
    timer.Toc();
    printf("\nTraining completed in %.1f s\n", timer.Seconds());

    auto assignments = kmeans.AssignTrainingPoints(data.get(), centroids.data(), n, n_clusters);
    const double wcss =
        skmeans::SuperKMeans<>::ComputeWCSS(data.get(), centroids.data(), assignments.data(), n, d);
    printf("WCSS: %.2f\n", wcss);

    repro::RecallEvaluator evaluator(
        ground_truth, queries.get(), repro::N_QUERIES, centroids.data(), n_clusters, d
    );
    const auto recall_at_10 = evaluator.Evaluate(assignments, 10);
    const auto recall_at_100 = evaluator.Evaluate(assignments, 100);
    repro::PrintRecall(recall_at_10, 10);
    repro::PrintRecall(recall_at_100, 100);
    repro::PrintResult(
        algorithm, dataset, n_clusters, iterations, "assign", timer.Seconds(), wcss, recall_at_100
    );
    return 0;
}
