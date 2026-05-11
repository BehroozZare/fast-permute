#pragma once
#include <cstdint>
#include <vector>

namespace homa {

struct Options {
    int  nd_levels           = 9;
    int  patch_size          = 512;
    bool use_patch_separator = true;
    bool compute_etree       = false; // set true when using cuDSS
    bool use_gpu             = false; // GPU backend (future)
    bool deterministic       = false;

    enum class LocalMethod { AMD, METIS, NONE };
    LocalMethod local_method = LocalMethod::AMD;
};

struct OrderingResult {
    std::vector<int> perm;
    std::vector<int> etree; // non-empty only when options.compute_etree == true
};

} // namespace homa
