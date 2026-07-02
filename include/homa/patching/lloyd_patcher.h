#pragma once
#include <vector>

#include "homa/patcher.h"


namespace homa {

/// Lloyd-clustering graph partitioner. Uses iterative BFS-based Lloyd
/// relaxation to produce bounded-size clusters. Requires the library to be
/// built with HOMA_WITH_LLOYD_PATCHER=ON.
class LloydPatcher : public IPatcher
{
   public:
    int patch_size  = 512;
    int lloyd_iters = 2;  ///< Lloyd BFS iterations between seed-addition rounds
    int random_seed = 42;

    enum class SeedMethod
    {
        RANDOM,
        MORTON,
        FPS
    };
    SeedMethod seed_method = SeedMethod::RANDOM;

    ~LloydPatcher() override = default;

    void compute(int               n,
                 const int*        Gp,
                 const int*        Gi,
                 std::vector<int>& node_to_patch) override;
};

}  // namespace homa
