#include <algorithm>
#include <spdlog/spdlog.h>
#include <stdexcept>

#include "homa/patching/lloyd_patcher.h"
#include "clusterAPI.h"

namespace homa {

void LloydPatcher::compute(int               n,
                           const int*        Gp,
                           const int*        Gi,
                           std::vector<int>& node_to_patch)
{
    LloydOptions opt;
    opt.lloyd_iters_to_add_seed = lloyd_iters;
    opt.random_seed             = random_seed;

    switch (seed_method) {
        case SeedMethod::MORTON:
            opt.seed_selection_method =
                LloydOptions::SeedSelectionMethod::MORTON_CODE;
            break;
        case SeedMethod::FPS:
            opt.seed_selection_method = LloydOptions::SeedSelectionMethod::FPS;
            break;
        default:
            opt.seed_selection_method =
                LloydOptions::SeedSelectionMethod::RANDOM;
            break;
    }

    spdlog::info("LloydPatcher: n={} patch_size={} lloyd_iters={}",
                 n,
                 patch_size,
                 lloyd_iters);

    using Clock     = std::chrono::high_resolution_clock;
    auto elapsed_ms = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(b - a)
            .count();
    };
    auto t0               = Clock::now();
    opt.execution_backend = LloydOptions::ExecutionBackend::GPU;
    create_clusters(n, Gp, Gi, patch_size, &opt, node_to_patch);
    spdlog::info("LloydPatcher: create_clusters took {} (ms)",
                 elapsed_ms(t0, Clock::now()));
    if (opt.execution_backend == LloydOptions::ExecutionBackend::GPU) {
        spdlog::info("LloydPatcher: create_clusters uses GPU backend");
    }
    if (opt.execution_backend == LloydOptions::ExecutionBackend::CPU) {
        spdlog::info("LloydPatcher: create_clusters uses CPU backend");
    }

    if (static_cast<int>(node_to_patch.size()) != n) {
        throw std::runtime_error(
            "LloydPatcher: invalid cluster assignment size");
    }

    std::vector<int> labels = node_to_patch;
    std::sort(labels.begin(), labels.end());
    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
    if (!labels.empty() && labels.front() < 0) {
        throw std::runtime_error(
            "LloydPatcher: unassigned vertex in cluster assignment");
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n > 4096)
#endif
    for (int i = 0; i < n; ++i) {
        node_to_patch[i] = static_cast<int>(
            std::lower_bound(labels.begin(), labels.end(), node_to_patch[i]) -
            labels.begin());
    }
    spdlog::debug("LloydPatcher: done — {} nodes assigned", n);
}

}  // namespace homa
