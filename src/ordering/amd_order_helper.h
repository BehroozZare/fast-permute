#pragma once

namespace homa::detail {

// Thin wrapper around SuiteSparse amd_order on 0-based CSR.
// Keeps <amd.h> as a private dependency of the homa library so that other
// translation units do not need SuiteSparse on their include path.
//
// Returns the AMD status code (AMD_OK == 0 on success). On success, `perm`
// is filled with a fill-reducing permutation of [0, n).
int amd_local_order(int n, int* Ap, int* Ai, int* perm);

}  // namespace homa::detail
