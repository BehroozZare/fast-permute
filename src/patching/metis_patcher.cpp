#include "metis_patcher.h"
#include <metis.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace homa {

void MetisPatcher::compute(int n, const int* Gp, const int* Gi,
                            std::vector<int>& node_to_patch)
{
    idx_t nvtxs  = static_cast<idx_t>(n);
    idx_t ncon   = 1;
    idx_t nparts = static_cast<idx_t>(
        std::max(1, (n + patch_size - 1) / patch_size));

    node_to_patch.assign(n, 0);

    if (nparts <= 1) {
        spdlog::info("MetisPatcher: single partition for n={}", n);
        return;
    }

    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_PTYPE]     = METIS_PTYPE_KWAY;
    options[METIS_OPTION_OBJTYPE]   = METIS_OBJTYPE_VOL;
    options[METIS_OPTION_NUMBERING] = 0;
    options[METIS_OPTION_CONTIG]    = 0;
    options[METIS_OPTION_COMPRESS]  = 0;
    options[METIS_OPTION_DBGLVL]    = 0;

    idx_t objval = 0;
    // METIS requires non-const pointers; the values are not modified
    int* Gp_nc = const_cast<int*>(Gp);
    int* Gi_nc = const_cast<int*>(Gi);

    int status = METIS_PartGraphKway(&nvtxs, &ncon, Gp_nc, Gi_nc,
        nullptr, nullptr, nullptr,
        &nparts, nullptr, nullptr, options,
        &objval, node_to_patch.data());
    
    if (status != METIS_OK) {
        spdlog::error("MetisPatcher: METIS_PartGraphKway failed (status={})", status);
        throw std::runtime_error("METIS_PartGraphKway failed");
    }

    spdlog::info("MetisPatcher: n={} -> {} patches, objval={}", n,
                 static_cast<int>(nparts), static_cast<long long>(objval));
}

} // namespace homa