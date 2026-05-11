#pragma once
#include "homa/types.h"
#include "homa/patcher.h"

namespace homa {

/// Compute a fill-reducing ordering using Homa's patch-based nested
/// dissection. The internal patcher (METIS-kway by default) is used.
/// @param n    number of rows/columns
/// @param Gp   CSR row pointers of the GRAPH (diagonal removed), size n+1
/// @param Gi   CSR column indices, size Gp[n]
OrderingResult compute_ordering(int n, const int* Gp, const int* Gi,
                                const Options& opts = {});

/// Same as above but uses a caller-supplied patch assignment, skipping
/// the internal patching step entirely.
/// @param node_to_patch  node_to_patch[i] = patch index for node i
OrderingResult compute_ordering(int n, const int* Gp, const int* Gi,
                                const std::vector<int>& node_to_patch,
                                const Options& opts = {});

} // namespace homa
