#pragma once
#include "homa/patcher.h"
#include <vector>

namespace homa {

/// METIS k-way graph partitioner. Assigns nodes to patches using
/// METIS_PartGraphKway with the given target patch size.
class MetisPatcher : public IPatcher {
public:
    int patch_size = 512;

    ~MetisPatcher() override = default;

    void compute(int n, const int* Gp, const int* Gi,
                 std::vector<int>& node_to_patch) override;
};

} // namespace homa
