#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME = true
#endif

#include <cstdio>
#include <string>
#include <vector>

#include "bench_utils.h"
#include "superkmeans/superemeans.h"

// e-means (pruning=false) and SuperEMeans (pruning=true).
// Usage: emeans <data_dir> <dataset> <n_clusters> <pruning: true|false> [epochs=2]
int main(int argc, char* argv[]) {
    if (argc < 5) {
        fprintf(
            stderr,
            "Usage: %s <data_dir> <dataset> <n_clusters> <pruning: true|false> [epochs=2]\n",
            argv[0]
        );
        return 1;
    }
    const std::string data_dir = argv[1], dataset = argv[2];
    const size_t n_clusters = std::stoul(argv[3]);
    const bool pruning = std::string(argv[4]) == "true";
    const uint32_t epochs = argc > 5 ? static_cast<uint32_t>(std::stoul(argv[5])) : 2;
    const auto [n, d] = repro::GetDataset(dataset);
    const std::string algorithm = pruning ? "superemeans" : "emeans";

    printf("=== %s | %s (n=%zu, d=%zu) | k=%zu | %u epochs ===\n", algorithm.c_str(),
           dataset.c_str(), n, d, n_clusters, epochs);
    auto data = repro::ReadFloats(repro::DataPath(data_dir, dataset), n * d);
    auto queries = repro::ReadFloats(repro::QueriesPath(data_dir, dataset), repro::N_QUERIES * d);
    const auto ground_truth = repro::ReadGroundTruth(repro::GroundTruthPath(data_dir, dataset));

    skmeans::SuperEMeansConfig config;
    config.iters = epochs;
    config.verbose = true;
    config.n_threads = skmeans::ResolveNumThreads(0);
    config.unrotate_centroids = true;
    config.use_blas_only = !pruning;

    auto emeans = skmeans::SuperEMeans<skmeans::Quantization::f32>(n_clusters, d, config);
    repro::Timer timer;
    timer.Tic();
    std::vector<float> centroids = emeans.Train(data.get(), n);
    timer.Toc();
    printf("\nTraining completed in %.1f s\n", timer.Seconds());
    printf("Batch size: %zu | Initial step: %f\n", emeans.GetBatchSize(), emeans.GetInitialStep());

    std::vector<uint32_t> last_pass(emeans.assignments.get(), emeans.assignments.get() + n);
    auto assignments = emeans.AssignTrainingPoints(data.get(), centroids.data(), n, n_clusters);

    repro::RecallEvaluator evaluator(
        ground_truth, queries.get(), repro::N_QUERIES, centroids.data(), n_clusters, d
    );
    for (const auto& [labels_name, labels] :
         {std::pair{"last_pass", &last_pass}, std::pair{"assign", &assignments}}) {
        const double wcss =
            skmeans::SuperKMeans<>::ComputeWCSS(data.get(), centroids.data(), labels->data(), n, d);
        printf("\n  [%s labels] WCSS: %.2f\n", labels_name, wcss);
        const auto recall_at_10 = evaluator.Evaluate(*labels, 10);
        const auto recall_at_100 = evaluator.Evaluate(*labels, 100);
        repro::PrintRecall(recall_at_10, 10);
        repro::PrintRecall(recall_at_100, 100);
        repro::PrintResult(
            algorithm, dataset, n_clusters, epochs, labels_name, timer.Seconds(), wcss, recall_at_100
        );
    }
    return 0;
}
