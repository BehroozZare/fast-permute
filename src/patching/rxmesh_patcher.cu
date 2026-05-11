#include "rxmesh_patcher.h"

#include <chrono>
#include <spdlog/spdlog.h>
#include <cuda_runtime.h>

#include "rxmesh/rxmesh_static.h"
#include "../utils/compute_inverse_perm.h"

namespace homa {

void RxMeshPatcher::setMesh(const double* V_data, int V_rows, int V_cols,
                             const int* F_data, int F_rows, int F_cols)
{
    fv.resize(F_rows);
    for (int i = 0; i < F_rows; ++i) {
        fv[i].resize(F_cols);
        for (int j = 0; j < F_cols; ++j) {
            fv[i][j] = static_cast<uint32_t>(F_data[i + j * F_rows]);
        }
    }
    vertices.resize(V_rows);
    for (int i = 0; i < V_rows; ++i) {
        vertices[i].resize(V_cols);
        for (int j = 0; j < V_cols; ++j) {
            vertices[i][j] = static_cast<float>(V_data[i + j * V_rows]);
        }
    }
}

void RxMeshPatcher::compute(int n, const int* /*Gp*/, const int* /*Gi*/,
                             std::vector<int>& node_to_patch)
{
    if (fv.empty()) {
        spdlog::warn("RxMeshPatcher: no mesh set, single patch fallback");
        node_to_patch.assign(n, 0);
        return;
    }

    rxmesh::rx_init(0);
    rxmesh::RXMeshStatic rx(fv, "", patch_size);
    spdlog::info("RxMeshPatcher: {} patches, patching time {}s",
                 rx.get_num_patches(), rx.get_patching_time());

    node_to_patch.resize(rx.get_num_vertices());
    rx.for_each_vertex(
        rxmesh::HOST,
        [&](const rxmesh::VertexHandle vh) {
            uint32_t node_id       = rx.map_to_global(vh);
            node_to_patch[node_id] = static_cast<int>(vh.patch_id());
        },
        NULL,
        false);
}

} // namespace homa