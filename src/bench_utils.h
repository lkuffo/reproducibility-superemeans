#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace repro {

struct Dataset {
    size_t n;
    size_t d;
};

inline const std::unordered_map<std::string, Dataset> DATASETS = {
    {"openai5m", {5'000'000, 1536}},
    {"cohere", {10'000'000, 1024}},
};

inline constexpr size_t N_QUERIES = 1000;

// Same probe sweep as SuperKMeans' benchmarks: nprobe = 1 and 2, then fractions of the centroids
inline const std::vector<int> ABSOLUTE_EXPLORE_COUNTS = {1, 2};
inline const std::vector<float> EXPLORE_FRACTIONS = {
    0.001f,  0.002f,  0.003f,  0.004f,  0.005f,  0.006f,  0.007f,  0.008f,  0.009f,
    0.0100f, 0.0125f, 0.0150f, 0.0175f, 0.0200f, 0.0225f, 0.0250f, 0.0275f, 0.0300f,
    0.0325f, 0.0350f, 0.0375f, 0.0400f, 0.0425f, 0.0450f, 0.0475f, 0.0500f, 0.1f
};

inline Dataset GetDataset(const std::string& name) {
    auto it = DATASETS.find(name);
    if (it == DATASETS.end()) {
        throw std::invalid_argument("Unknown dataset '" + name + "' (use openai5m or cohere)");
    }
    return it->second;
}

inline std::string DataPath(const std::string& data_dir, const std::string& dataset) {
    return data_dir + "/data_" + dataset + ".bin";
}

inline std::string QueriesPath(const std::string& data_dir, const std::string& dataset) {
    return data_dir + "/data_" + dataset + "_test.bin";
}

inline std::string GroundTruthPath(const std::string& data_dir, const std::string& dataset) {
    return data_dir + "/gt_" + dataset + ".json";
}

inline std::unique_ptr<float[]> ReadFloats(const std::string& path, size_t count) {
    std::unique_ptr<float[]> out(new float[count]);
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open " + path);
    }
    const auto n_bytes = static_cast<std::streamsize>(count * sizeof(float));
    file.read(reinterpret_cast<char*>(out.get()), n_bytes);
    if (file.gcount() != n_bytes) {
        throw std::runtime_error(path + " holds fewer values than expected");
    }
    return out;
}

// Parses {"query index": [ids of the true nearest neighbors], ...} as written by setup_data.py
inline std::unordered_map<int, std::vector<int>> ReadGroundTruth(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open " + path);
    }
    const std::string content(
        (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>()
    );
    std::unordered_map<int, std::vector<int>> ground_truth;
    size_t pos = 0;
    while ((pos = content.find('"', pos)) != std::string::npos) {
        const size_t key_end = content.find('"', pos + 1);
        const size_t list_start = content.find('[', key_end);
        const size_t list_end = content.find(']', list_start);
        if (key_end == std::string::npos || list_start == std::string::npos ||
            list_end == std::string::npos) {
            break;
        }
        const int query = std::stoi(content.substr(pos + 1, key_end - pos - 1));
        std::vector<int> ids;
        std::istringstream list(content.substr(list_start + 1, list_end - list_start - 1));
        std::string token;
        while (std::getline(list, token, ',')) {
            ids.push_back(std::stoi(token));
        }
        ground_truth[query] = std::move(ids);
        pos = list_end + 1;
    }
    return ground_truth;
}

class Timer {
  public:
    void Tic() { start = std::chrono::high_resolution_clock::now(); }
    void Toc() { elapsed += std::chrono::high_resolution_clock::now() - start; }
    double Seconds() const { return std::chrono::duration<double>(elapsed).count(); }

  private:
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::duration elapsed{0};
};

struct RecallPoint {
    int n_probe;
    float fraction;
    float recall;
    float recall_std;
    float avg_vectors;
};

// IVF recall: probe the n_probe centroids closest to each query, count how many of the query's
// true nearest neighbors are assigned to them (recall), and how many vectors they hold (vectors
// scanned). The centroid order of each query is computed once and reused for every n_probe.
class RecallEvaluator {
  public:
    RecallEvaluator(
        const std::unordered_map<int, std::vector<int>>& ground_truth,
        const float* queries,
        const size_t n_queries,
        const float* centroids,
        const size_t n_clusters,
        const size_t d
    )
        : ground_truth(ground_truth), n_queries(n_queries), n_clusters(n_clusters),
          order(n_queries * n_clusters), rank(n_queries * n_clusters) {
        for (int count : ABSOLUTE_EXPLORE_COUNTS) {
            const int c = std::min(count, static_cast<int>(n_clusters));
            configs.push_back({c, static_cast<float>(c) / static_cast<float>(n_clusters)});
        }
        for (float frac : EXPLORE_FRACTIONS) {
            configs.push_back({std::max(1, static_cast<int>(n_clusters * frac)), frac});
        }

        std::vector<float> centroid_norms(n_clusters);
        for (size_t j = 0; j < n_clusters; ++j) {
            float norm = 0.0f;
            for (size_t t = 0; t < d; ++t) {
                norm += centroids[j * d + t] * centroids[j * d + t];
            }
            centroid_norms[j] = norm;
        }

        const size_t n_threads = std::max<size_t>(1, std::thread::hardware_concurrency());
        std::vector<std::thread> threads;
        for (size_t w = 0; w < n_threads; ++w) {
            threads.emplace_back([&, w]() {
                std::vector<std::pair<float, int>> distances(n_clusters);
                for (size_t q = w; q < n_queries; q += n_threads) {
                    const float* query = queries + q * d;
                    float query_norm = 0.0f;
                    for (size_t t = 0; t < d; ++t) {
                        query_norm += query[t] * query[t];
                    }
                    for (size_t j = 0; j < n_clusters; ++j) {
                        float dot = 0.0f;
                        for (size_t t = 0; t < d; ++t) {
                            dot += query[t] * centroids[j * d + t];
                        }
                        distances[j] = {
                            query_norm + centroid_norms[j] - 2.0f * dot, static_cast<int>(j)
                        };
                    }
                    std::sort(distances.begin(), distances.end());
                    for (size_t r = 0; r < n_clusters; ++r) {
                        const auto c = static_cast<uint32_t>(distances[r].second);
                        order[q * n_clusters + r] = c;
                        rank[q * n_clusters + c] = static_cast<uint32_t>(r);
                    }
                }
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
    }

    std::vector<RecallPoint> Evaluate(const std::vector<uint32_t>& assignments, int knn) const {
        std::vector<size_t> cluster_sizes(n_clusters, 0);
        for (uint32_t label : assignments) {
            cluster_sizes[label]++;
        }
        std::vector<std::vector<float>> recalls(configs.size());
        std::vector<size_t> total_vectors(configs.size(), 0);
        std::vector<size_t> prefix_vectors(n_clusters);
        for (size_t q = 0; q < n_queries; ++q) {
            auto it = ground_truth.find(static_cast<int>(q));
            if (it == ground_truth.end()) {
                continue;
            }
            const uint32_t* query_order = order.data() + q * n_clusters;
            const uint32_t* query_rank = rank.data() + q * n_clusters;
            size_t running = 0;
            for (size_t r = 0; r < n_clusters; ++r) {
                running += cluster_sizes[query_order[r]];
                prefix_vectors[r] = running;
            }
            const auto& neighbors = it->second;
            const int gt_count = std::min(knn, static_cast<int>(neighbors.size()));
            for (size_t c = 0; c < configs.size(); ++c) {
                const auto n_probe = static_cast<uint32_t>(configs[c].first);
                total_vectors[c] += prefix_vectors[n_probe - 1];
                int found = 0;
                for (int i = 0; i < gt_count; ++i) {
                    found += query_rank[assignments[neighbors[i]]] < n_probe;
                }
                recalls[c].push_back(static_cast<float>(found) / static_cast<float>(gt_count));
            }
        }

        std::vector<RecallPoint> points;
        for (size_t c = 0; c < configs.size(); ++c) {
            float mean = 0.0f;
            for (float recall : recalls[c]) {
                mean += recall;
            }
            mean /= static_cast<float>(n_queries);
            float std = 0.0f;
            if (recalls[c].size() > 1) {
                float variance = 0.0f;
                for (float recall : recalls[c]) {
                    variance += (recall - mean) * (recall - mean);
                }
                std = std::sqrt(variance / static_cast<float>(recalls[c].size() - 1));
            }
            points.push_back(
                {configs[c].first,
                 configs[c].second,
                 mean,
                 std,
                 static_cast<float>(total_vectors[c]) / static_cast<float>(n_queries)}
            );
        }
        return points;
    }

  private:
    const std::unordered_map<int, std::vector<int>>& ground_truth;
    const size_t n_queries;
    const size_t n_clusters;
    std::vector<std::pair<int, float>> configs;
    std::vector<uint32_t> order;
    std::vector<uint32_t> rank;
};

inline void PrintRecall(const std::vector<RecallPoint>& points, int knn) {
    printf("\n--- Recall@%d ---\n", knn);
    for (const auto& p : points) {
        printf(
            "Recall@%4d (%5.2f%% centroids, %8.0f avg vectors): %.4f ± %.4f\n",
            p.n_probe,
            p.fraction * 100.0f,
            p.avg_vectors,
            p.recall,
            p.recall_std
        );
    }
}

inline const RecallPoint& AtFraction(const std::vector<RecallPoint>& points, float fraction) {
    for (const auto& p : points) {
        if (std::abs(p.fraction - fraction) < 1e-6f) {
            return p;
        }
    }
    throw std::runtime_error("Probe fraction not evaluated");
}

// One machine-readable line per run, parsed by scripts/summarize.py
inline void PrintResult(
    const std::string& algorithm,
    const std::string& dataset,
    size_t n_clusters,
    uint32_t iterations,
    const std::string& labels,
    double train_seconds,
    double wcss,
    const std::vector<RecallPoint>& recall_at_100
) {
    const auto& p100 = AtFraction(recall_at_100, 0.01f);
    const auto& p090 = AtFraction(recall_at_100, 0.009f);
    printf(
        "RESULT algorithm=%s dataset=%s n_clusters=%zu iterations=%u labels=%s train_s=%.1f "
        "wcss=%.2f recall100_1pct=%.4f vectors_1pct=%.0f recall100_090pct=%.4f "
        "vectors_090pct=%.0f\n",
        algorithm.c_str(),
        dataset.c_str(),
        n_clusters,
        iterations,
        labels.c_str(),
        train_seconds,
        wcss,
        p100.recall,
        p100.avg_vectors,
        p090.recall,
        p090.avg_vectors
    );
}

} // namespace repro
