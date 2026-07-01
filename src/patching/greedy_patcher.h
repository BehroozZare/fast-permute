#pragma once

#include "homa/patcher.h"

namespace homa {

// One pass greedy patching. It is a lot cheapter than Lloyd refinement since 
// each patch is filled by a bounded BFS from the next unassigned vertex.
class GreedyPatcher : public IPatcher {
public:
  int patch_size = 512;

  ~GreedyPatcher() override = default;

  void compute(int n, const int *Gp, const int *Gi,
               std::vector<int> &node_to_patch) override;
};

} // namespace homa
