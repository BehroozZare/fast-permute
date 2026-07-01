#include "patching/greedy_patcher.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <stdexcept>

#include <spdlog/spdlog.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace homa {

namespace {

// Serial BFS patcher fallback. Each call seeds at vertex `seed_start` 
// (assumed unassigned) and grows a single patch up to patch_size vertices, 
// returning the number of vertices it actually claimed.
int bfs_one_patch(int n, const int *Gp, const int *Gi,
                  std::vector<int> &node_to_patch, std::vector<int> &queue,
                  int seed, int patch_id, int patch_size) {
  queue.clear();
  node_to_patch[seed] = patch_id;
  queue.push_back(seed);

  size_t head = 0;
  while (head < queue.size() &&
      static_cast<int>(queue.size()) < patch_size) {
    const int v = queue[head++];
    for (int e = Gp[v]; e < Gp[v + 1] &&
        static_cast<int>(queue.size()) < patch_size;
        ++e) {
      const int u = Gi[e];
      if (u < 0 || u >= n) {
          continue;
      }
      if (node_to_patch[u] != -1) {
          continue;
      }
      node_to_patch[u] = patch_id;
      queue.push_back(u);
    }
  }
  return static_cast<int>(queue.size());
}

} // namespace

void GreedyPatcher::compute(int n, const int *Gp, const int *Gi,
                             std::vector<int> &node_to_patch) {
  if (n < 0 || Gp == nullptr || Gi == nullptr) {
    throw std::invalid_argument("GreedyPatcher: invalid graph");
  }
  if (patch_size <= 0) {
    throw std::invalid_argument("GreedyPatcher: patch_size must be positive");
  }

  using Clock = std::chrono::high_resolution_clock;
  const auto start = Clock::now();

  node_to_patch.assign(n, -1);
    
#ifdef _OPENMP
  const int max_threads = std::max(1, omp_get_max_threads());
#else
  const int max_threads = 1;
#endif
  const int kParallelMinNodes = 16 * 1024;
  const bool use_parallel = max_threads > 1 && n >= kParallelMinNodes;
  
  int patch_id = 0;

  if (use_parallel) {    
    //1. Pick a set of well-spread seed vertices. The simplest choice that
    //   respects graph locality without expensive analysis is to slice
    //   the vertex array into equal chunks and pick the first vertex of
    //   every chunk. 
    //2. Each thread runs serial BFS from its seeds, claiming vertices
    //   via `std::atomic_compare_exchange`
    //3. After the parallel pass, a serial sweep grows leftover patches
    //   from any vertex that is still unassigned. This handles boundary
    //   regions and disconnected components that the parallel pass
    //   missed.
    const int target_patches = std::max(1, (n + patch_size - 1) / patch_size);
    const int per_thread_seeds = std::max(1, target_patches / max_threads);

    // Generate seed vertices spaced ~patch_size apart and assign them a
    // unique patch id up front so we can BFS in parallel.
    std::vector<int> seeds;
    seeds.reserve(target_patches);
    for (int v = 0; v < n; v += patch_size) {
      seeds.push_back(v);
    }
    const int num_seeds = static_cast<int>(seeds.size());

    // Per-thread scratch + patch id allocator.
    std::atomic<int> next_patch_id{0};
    std::atomic<int> total_assigned{0};
    
    // We use a `std::vector<std::atomic<int>>` for node_to_patch writes    
    std::vector<std::atomic<int>> atomic_owner(n);

#pragma omp parallel for schedule(static)
    for (int v = 0; v < n; ++v) { 
        atomic_owner[v].store(-1, std::memory_order_relaxed); 
    }

#pragma omp parallel
    {
      std::vector<int> queue;
      queue.reserve(static_cast<size_t>(patch_size));

#pragma omp for schedule(dynamic, std::max(1, per_thread_seeds / 4))
      for (int s = 0; s < num_seeds; ++s) {
        const int seed = seeds[s];
        int expected = -1;
        const int my_patch = next_patch_id.fetch_add(1,
                                                     std::memory_order_relaxed);
        if (!atomic_owner[seed].compare_exchange_strong(
                expected, my_patch, std::memory_order_relaxed)) {
          // Another thread already grew a patch into our seed, skip
          continue;
        }
        queue.clear();
        queue.push_back(seed);
        int local_assigned = 1;

        size_t head = 0;
        while (head < queue.size() &&
               static_cast<int>(queue.size()) < patch_size) {
          const int v = queue[head++];
          for (int e = Gp[v]; e < Gp[v + 1] &&
              static_cast<int>(queue.size()) < patch_size;
              ++e) {
            const int u = Gi[e];
            if (u < 0 || u >= n) { 
                continue; 
            }
            int free_slot = -1;
            if (atomic_owner[u].compare_exchange_strong(
                    free_slot, my_patch, std::memory_order_relaxed)) {
              queue.push_back(u);
              ++local_assigned;
            }
          }
        }
        total_assigned.fetch_add(local_assigned, std::memory_order_relaxed);
      }
    }
        
#pragma omp parallel for schedule(static)
    for (int v = 0; v < n; ++v) {
      node_to_patch[v] = atomic_owner[v].load(std::memory_order_relaxed);
    }

    patch_id = next_patch_id.load();

    // Compact the patch ids: parallel pass may have allocated ids for seeds
    // that were grabbed by another thread (resulting in a skipped patch).
    // We compact by remapping used ids to a dense 0..k-1 range.
    std::vector<int> id_map(patch_id, -1);
    for (int v = 0; v < n; ++v) {
      const int pid = node_to_patch[v];
      if (pid < 0) { 
          continue; 
      }
      if (id_map[pid] == -1) {
        // Will be assigned on second pass after we know the final count.
        id_map[pid] = 0;
      }
    }
    int compact_id = 0;
    for (int pid = 0; pid < patch_id; ++pid) {
      if (id_map[pid] == 0) {
        id_map[pid] = compact_id++;
      }
    }
#pragma omp parallel for schedule(static)
    for (int v = 0; v < n; ++v) {
      const int pid = node_to_patch[v];
      if (pid >= 0) { 
          node_to_patch[v] = id_map[pid]; 
      }
    }
    patch_id = compact_id;
  }

  // Serial cleanup pass where any vertex still tagged -1 (because it was outside
  // the parallel pass's reachable region) gets a new patch grown from it.
  // This loop is identical to the original serial BFS but bounded to the
  // remaining vertices, so it's typically very short (and runs only at all
  // when the parallel pass left holes).
  {
    std::vector<int> queue;
    queue.reserve(static_cast<size_t>(patch_size));
    int next_seed = 0;
    while (next_seed < n) {
      while (next_seed < n && node_to_patch[next_seed] != -1) ++next_seed;
      if (next_seed >= n) break;
      bfs_one_patch(n, Gp, Gi, node_to_patch, queue, next_seed, patch_id,
                    patch_size);
      ++patch_id;
    }
  }

  if (std::any_of(node_to_patch.begin(), node_to_patch.end(),
                  [](int patch) { return patch < 0; })) {
    throw std::runtime_error("GreedyPatcher: failed to assign all vertices");
  }

  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  spdlog::info("GreedyPatcher: use_parallel= {}, n={} patch_size={} patches={} time={:.3f} ms",
      (use_parallel ? "true" : "false"),
      n,
      patch_size,
      patch_id,
      elapsed_ms);
}

} // namespace homa
