#include "homa/ordering.h"
#include "ordering/cpu_ordering_with_patch.h"
#ifdef HOMA_WITH_LLOYD_PATCHER
#include "patching/lloyd_patcher.h"
#else
#include "patching/metis_patcher.h"
#endif
#include <cassert>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <unordered_set>

namespace homa {

namespace {

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
                            const Options &opts) {
  homa::CPUOrdering_PATCH cpu;
  cpu.applyOptions(opts);

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
  cpu.compute_permutation(result.perm);

  if (opts.compute_etree) {
    apply_etree_reorder(cpu, result.perm, result.etree);
  }

  return result;
}

} // anonymous namespace

OrderingResult compute_ordering(int n, const int *Gp, const int *Gi,
                                const Options &opts) {
  std::vector<int> node_to_patch;

#ifdef HOMA_WITH_LLOYD_PATCHER
  {
    LloydPatcher patcher;
    patcher.patch_size = opts.patch_size;
    patcher.lloyd_iters = opts.lloyd_iters;
    patcher.random_seed = opts.lloyd_seed;
    using S = Options::LloydSeedMethod;
    if (opts.lloyd_seed_method == S::MORTON)
      patcher.seed_method = LloydPatcher::SeedMethod::MORTON;
    else if (opts.lloyd_seed_method == S::FPS)
      patcher.seed_method = LloydPatcher::SeedMethod::FPS;
    patcher.compute(n, Gp, Gi, node_to_patch);
  }
#else
  {
    MetisPatcher patcher;
    patcher.patch_size = opts.patch_size;
    patcher.compute(n, Gp, Gi, node_to_patch);
  }
#endif

  return compute_ordering(n, Gp, Gi, node_to_patch, opts);
}

OrderingResult compute_ordering(int n, const int *Gp, const int *Gi,
                                const std::vector<int> &node_to_patch,
                                const Options &opts) {
  if (n <= 0)
    throw std::invalid_argument("homa::compute_ordering: n must be positive");
  if (static_cast<int>(node_to_patch.size()) != n)
    throw std::invalid_argument(
        "homa::compute_ordering: node_to_patch.size() != n");

  return run_ordering(n, Gp, Gi, node_to_patch, opts);
}

} // namespace homa
