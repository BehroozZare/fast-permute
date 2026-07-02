#pragma once
#include <cstdint>
#include <vector>

namespace homa {

struct Options {
  int nd_levels = 9;
  int patch_size = 512;

  // Selects how each internal decomposition node finds its vertex separator:
  //   AUTO         - pick per node/level via a size/density heuristic (default)
  //   QUOTIENT     - always the (cheap) quotient-graph based separator
  //   DIRECT_METIS - always METIS_ComputeVertexSeparator on the subgraphs
  enum class SeparatorMethod { AUTO, QUOTIENT, DIRECT_METIS };
  SeparatorMethod separator_method = SeparatorMethod::AUTO;
    
  // -1 is use heuristics; 0 means only the root uses direct graph separators.
  //only active with AUTO separator method 
  int direct_separator_max_level = -1;

  // -1 is use heuristics; direct separators are only used for subgraphs at or
  // above this size.
  int direct_separator_min_nodes = -1;

  bool compute_etree = false; // set true when using cuDSS
  bool use_gpu = true;       // GPU backend
  

  enum class LocalMethod { AMD, METIS, NONE };
  LocalMethod local_method = LocalMethod::AMD;

  enum class PatchMethod { LLOYD, METIS, GREEDY };
  PatchMethod patch_method = PatchMethod::GREEDY;

  // Lloyd patcher options — effective when HOMA_WITH_LLOYD_PATCHER is enabled.
  // lloyd_iters: BFS iterations between seed-addition rounds (default matches
  // the benchmark script default of 2).
  int lloyd_iters = 2;
  int lloyd_seed = 42;

  enum class LloydSeedMethod { RANDOM, MORTON, FPS };
  LloydSeedMethod lloyd_seed_method = LloydSeedMethod::RANDOM;
};

struct OrderingResult {
  std::vector<int> perm;
  std::vector<int> etree; // non-empty only when options.compute_etree == true
};

} // namespace homa
