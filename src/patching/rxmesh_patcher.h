#pragma once
#include "homa/patcher.h"
#include <cstdint>
#include <vector>

namespace homa {

/// RXMesh-based patcher. Assigns nodes to patches using the RXMesh GPU
/// mesh partitioning. Call setMesh() before compute(); without mesh data
/// it assigns all nodes to a single patch.
class RxMeshPatcher : public IPatcher {
public:
    int patch_size = 512;

    std::vector<std::vector<uint32_t>> fv;
    std::vector<std::vector<float>> vertices;

    ~RxMeshPatcher() override = default;

    void setMesh(const double* V_data, int V_rows, int V_cols,
                 const int* F_data, int F_rows, int F_cols);

    void compute(int n, const int* Gp, const int* Gi,
                 std::vector<int>& node_to_patch) override;
};

} // namespace homa
