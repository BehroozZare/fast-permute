#include "homa/ordering.h"
#include "homa/utils/remove_diagonal.h"
#include "ordering/cpu_ordering_with_patch.h"
#include "patching/greedy_patcher.h"
#ifdef HOMA_WITH_LLOYD_PATCHER
#include "patching/lloyd_patcher.h"
#endif
#include "patching/metis_patcher.h"
#include <cassert>
#include <chrono>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <unordered_set>

namespace homa {

namespace {

using Clock = std::chrono::high_resolution_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

struct OrderingRunTiming {
  double compute_permutation_ms = 0.0;
  double etree_reorder_ms = 0.0;
};

void resolve_separator_policy(int n, const int* Gp,
    const Options& opts,
    bool& use_quotient_base,
    int& max_level,
    int& min_nodes) {
    // Decides which decomposition levels should use a DIRECT METIS vertex
    // separator on the actual subgraph (`METIS_ComputeVertexSeparator`) vs. the
    // cheaper QUOTIENT-graph based separator (`two_way_Q_partition` +
    // `compute_vertex_separator` + lightweight refinement).
    //
    // Empirically the two methods trade ordering cost for factorization
    // quality:
    //   - Direct  : high ordering cost, slightly better separators
    //   - Quotient: very cheap ordering, separators ride on patch boundaries
    //
    // Our patches are produced by GreedyPatcher and already follow the graph's
    // natural connectivity, so the quality gap is usually small while the cost
    // gap is huge — especially for upper tree levels where direct METIS runs on
    // the full subgraph with only 1 / 2 / 4 OpenMP threads available.
    //
    // Previously the heuristic forced direct METIS at every level whenever
    // n < 50k, which made HOMA significantly slower than cuDSS's default
    // ordering for small dense matrices
    if (opts.direct_separator_min_nodes >= 0) {
        min_nodes = opts.direct_separator_min_nodes;
    }
    max_level = -1;

    // Mode 1: always the cheap quotient-graph separator (never direct METIS)
    if (opts.separator_method == Options::SeparatorMethod::QUOTIENT) {
        use_quotient_base = true;
        return;
    }
    // Mode 2: always METIS_ComputeVertexSeparator on the real subgraph.
    if (opts.separator_method == Options::SeparatorMethod::DIRECT_METIS) {
        use_quotient_base = false;
        return;
    }

    // Mode 3 (AUTO): quotient base plus the size/density heuristic below
    use_quotient_base = true;
    if (opts.direct_separator_max_level >= 0 || !opts.compute_etree) {
        max_level = opts.direct_separator_max_level;
        return;
    }

    if (n < min_nodes) {
        return;
    }

    const double avg_degree =
        n > 0 ? static_cast<double>(Gp[n]) / static_cast<double>(n) : 0.0;

    // Direct METIS dominates ordering cost roughly as O(nnz_subgraph * log n),
    // and METIS_ComputeVertexSeparator is essentially serial. The wall-clock
    // for the top L+1 levels is therefore ~ (nnz / threads_at_each_level) and
    // bounded below by the *level 0* call which runs on a single thread.
    //
    // The classic FEM / mesh regime (avg_degree ~ 6) really does benefit from
    // direct METIS at every level — so leave those configurations alone.
    if (n < 50000) {
        if (avg_degree < 8.0) {
            // Sparse small matrices (FEM 2D, planar meshes). Direct METIS is cheap
            // here because there are few edges; keep the original policy.
            max_level = opts.nd_levels;
        }
        else if (avg_degree < 20.0) {
            // Moderately dense small matrices. Use direct METIS only for the top
            // few levels where it pays off the most (gives the high-quality cut
            // at the root), then switch to quotient-based for the rest.
            max_level = 1;
        }
        else {
            // Dense / very dense small matrices. Direct METIS even at the top
            // level costs more than we ever save downstream. Skip direct METIS
            // entirely and rely on the quotient-based separator running on the
            // (much smaller) patch graph.
            max_level = -1;
        }
        return;
    }

    // Large matrices (>= 50k rows). Here factorization usually dominates, so
    // we keep more levels on direct METIS for better quality.
    if (avg_degree >= 30.0) {
        max_level = 3;
        return;
    }
    if (avg_degree >= 8.0) {
        max_level = n >= 150000 ? 3 : 2;
    }
}

// After CPUOrdering_PATCH::compute_permutation(), if compute_etree was
// requested, rebuild the permutation in level order and populate the etree.
void apply_etree_reorder(homa::CPUOrdering_PATCH &cpu, std::vector<int> &perm,
                         std::vector<int> &etree) {
  auto &nodes = cpu._decomposition_tree.decomposition_nodes;
  int binary_tree_size = static_cast<int>(nodes.size());

  // level_numbering[hmd_id] = position in the level-order sequence
  std::vector<int> level_numbering(binary_tree_size);
  for (int i = 0; i < binary_tree_size; ++i) {
    level_numbering[binary_tree_size - 1 - i] = i;
  }

  // etree[level_idx] = number of nodes owned by that tree node
  etree.assign(binary_tree_size, 0);
  for (int hmd_id = 0; hmd_id < binary_tree_size; ++hmd_id) {
    int level_idx = level_numbering[hmd_id];
    etree[level_idx] = static_cast<int>(nodes[hmd_id].assigned_g_nodes.size());
  }

  // Reassemble permutation in level order
  std::vector<int> etree_inverse(binary_tree_size);
  for (int hmd_id = 0; hmd_id < binary_tree_size; ++hmd_id) {
    etree_inverse[level_numbering[hmd_id]] = hmd_id;
  }

  perm.assign(cpu._G_n, -1);
  int offset = 0;
  for (int i = 0; i < static_cast<int>(etree_inverse.size()); ++i) {
    int hmd_id = etree_inverse[i];
    auto &node = nodes[hmd_id];
    if (node.assigned_g_nodes.empty())
      continue;
    for (int local = 0; local < static_cast<int>(node.assigned_g_nodes.size());
         ++local) {
      int global_node = node.assigned_g_nodes[local];
      int perm_index = node.local_new_labels[local] + offset;
      assert(perm_index >= 0 && perm_index < cpu._G_n);
      perm[perm_index] = global_node;
    }
    offset += static_cast<int>(node.assigned_g_nodes.size());
  }
}

OrderingResult run_ordering(int n, const int *Gp, const int *Gi,
                            const std::vector<int> &node_to_patch,
                            const Options &opts,
                            OrderingRunTiming *timing = nullptr) {
  homa::CPUOrdering_PATCH cpu;
  cpu.applyOptions(opts);
  
  resolve_separator_policy(n, Gp, opts, cpu.use_patch_separator,
                           cpu.direct_separator_max_level, cpu.direct_separator_min_nodes);
  
  if (cpu.direct_separator_max_level >= 0) {
    const double avg_degree =
        n > 0 ? static_cast<double>(Gp[n]) / static_cast<double>(n) : 0.0;
    spdlog::info(
        "HOMA separator: using direct graph separators through level {} "
        "for subgraphs with at least {} nodes (n={} avg_degree={:.3f})",
        cpu.direct_separator_max_level,
        cpu.direct_separator_min_nodes,
        n,
        avg_degree);
  }

  // setGraph takes non-const; the algorithm only reads the data
  cpu.setGraph(const_cast<int *>(Gp), const_cast<int *>(Gi), n, Gp[n]);

  // Count distinct patches
  int num_patches = 0;
  {
    std::unordered_set<int> ids(node_to_patch.begin(), node_to_patch.end());
    num_patches = static_cast<int>(ids.size());
  }
  if (num_patches == 0)
    num_patches = 1;

  std::vector<int> ntp = node_to_patch; // init_patches takes non-const
  cpu.init_patches(num_patches, ntp, opts.nd_levels);

  OrderingResult result;
  auto ordering_start = Clock::now();
  cpu.compute_permutation(result.perm);
  if (timing != nullptr) {
    timing->compute_permutation_ms = elapsed_ms(ordering_start, Clock::now());
  }

  if (opts.compute_etree) {
    auto etree_start = Clock::now();
    apply_etree_reorder(cpu, result.perm, result.etree);
    if (timing != nullptr) {
      timing->etree_reorder_ms = elapsed_ms(etree_start, Clock::now());
    }
  }

  return result;
}

void compute_node_to_patch(int n, const int* Gp, const int* Gi,
    const Options& opts,
    std::vector<int>& node_to_patch,
    double& patcher_ms)
{

    auto patcher_start = Clock::now();
    if (opts.patch_method == Options::PatchMethod::GREEDY) {
        spdlog::info("HOMA patcher: using GreedyPatcher");
        GreedyPatcher patcher;
        patcher.patch_size = opts.patch_size;
        patcher.compute(n, Gp, Gi, node_to_patch);
        patcher_ms = elapsed_ms(patcher_start, Clock::now());
        return;
    }

    if (opts.patch_method == Options::PatchMethod::METIS) {
        spdlog::info("HOMA patcher: using MetisPatcher");
        MetisPatcher patcher;
        patcher.patch_size = opts.patch_size;
        patcher.compute(n, Gp, Gi, node_to_patch);
        patcher_ms = elapsed_ms(patcher_start, Clock::now());
        return;
    }

    if (opts.patch_method == Options::PatchMethod::LLOYD) {
#ifdef HOMA_WITH_LLOYD_PATCHER
        spdlog::info("HOMA patcher: using LloydPatcher");
        LloydPatcher patcher;
        patcher.patch_size = opts.patch_size;
        patcher.lloyd_iters = opts.lloyd_iters;
        patcher.random_seed = opts.lloyd_seed;
        using SeedMethod = Options::LloydSeedMethod;
        if (opts.lloyd_seed_method == SeedMethod::MORTON) {
            patcher.seed_method = LloydPatcher::SeedMethod::MORTON;
        }
        else if (opts.lloyd_seed_method == SeedMethod::FPS) {
            patcher.seed_method = LloydPatcher::SeedMethod::FPS;
        }
        patcher.compute(n, Gp, Gi, node_to_patch);
#else
        throw std::runtime_error(
            "homa::compute_ordering: Lloyd patcher was requested but "
            "HOMA_WITH_LLOYD_PATCHER is disabled");
#endif

        patcher_ms = elapsed_ms(patcher_start, Clock::now());
        return;
    }
}

} // anonymous namespace

OrderingResult compute_ordering(int n, const int *Gp, const int *Gi,
                                const Options &opts) {
    if (n <= 0) {
        throw std::invalid_argument("homa::compute_ordering: n must be positive");
    }
    
  auto total_start = Clock::now();
  std::vector<int> node_to_patch;
  double patcher_ms = 0.0;
  compute_node_to_patch(n, Gp, Gi, opts, node_to_patch, patcher_ms);

  if (static_cast<int>(node_to_patch.size()) != n) {
      throw std::invalid_argument(
          "homa::compute_ordering: node_to_patch.size() != n");
  }

  OrderingRunTiming timing;
  OrderingResult result = run_ordering(n, Gp, Gi, node_to_patch, opts, &timing);
  spdlog::info(
      "HOMA CPU ordering timing: compute_ordering total={:.3f} ms "
      "patcher={:.3f} ms cpu_ordering={:.3f} ms etree_reorder={:.3f} ms",
      elapsed_ms(total_start, Clock::now()),
      patcher_ms,
      timing.compute_permutation_ms,
      timing.etree_reorder_ms);
  return result;
}

template <class Scalar>
OrderingResult compute_ordering(const Eigen::SparseMatrix<Scalar>& A,
                                const Options& opts) {
  if (A.rows() != A.cols()) {
    throw std::invalid_argument(
        "homa::compute_ordering: sparse matrix must be square");
  }

  Eigen::SparseMatrix<Scalar> compressed = A;
  compressed.makeCompressed();

  std::vector<int> Gp, Gi;
  remove_diagonal(static_cast<int>(compressed.rows()),
                  compressed.outerIndexPtr(),
                  compressed.innerIndexPtr(),
                  Gp,
                  Gi);
  return compute_ordering(static_cast<int>(compressed.rows()),
                          Gp.data(),
                          Gi.data(),
                          opts);
}

template <class Scalar>
OrderingResult compute_ordering(const SparseMatrixView<Scalar>& A,
                                const Options& opts) {
  if (A.location != MemoryLocation::Host) {
    throw std::invalid_argument(
        "homa::compute_ordering: device matrix views are not supported");
  }
  if (A.rows <= 0 || A.cols <= 0 || A.rows != A.cols || A.nnz <= 0) {
    throw std::invalid_argument(
        "homa::compute_ordering: invalid sparse matrix dimensions");
  }
  if (A.outer == nullptr || A.inner == nullptr) {
    throw std::invalid_argument(
        "homa::compute_ordering: sparse matrix view has null pattern pointers");
  }

  std::vector<int> Gp, Gi;
  remove_diagonal(A.rows, A.outer, A.inner, Gp, Gi);
  return compute_ordering(A.rows, Gp.data(), Gi.data(), opts);
}

OrderingResult compute_ordering(int n, const int *Gp, const int *Gi,
                                const std::vector<int> &node_to_patch,
                                const Options &opts) {
    if (n <= 0) {
        throw std::invalid_argument("homa::compute_ordering: n must be positive");
    }
    if (static_cast<int>(node_to_patch.size()) != n) {
        throw std::invalid_argument(
            "homa::compute_ordering: node_to_patch.size() != n");
    }

  auto total_start = Clock::now();
  OrderingRunTiming timing;
  OrderingResult result = run_ordering(n, Gp, Gi, node_to_patch, opts, &timing);
  spdlog::info(
      "HOMA CPU ordering timing: compute_ordering total={:.3f} ms "
      "patcher={:.3f} ms cpu_ordering={:.3f} ms etree_reorder={:.3f} ms",
      elapsed_ms(total_start, Clock::now()),
      0.0,
      timing.compute_permutation_ms,
      timing.etree_reorder_ms);
  return result;
}

template OrderingResult compute_ordering<float>(const Eigen::SparseMatrix<float>&,
                                                const Options&);
template OrderingResult compute_ordering<double>(const Eigen::SparseMatrix<double>&,
                                                 const Options&);
template OrderingResult compute_ordering<float>(const SparseMatrixView<float>&,
                                              const Options&);
template OrderingResult compute_ordering<double>(const SparseMatrixView<double>&,
                                                 const Options&);

} // namespace homa
