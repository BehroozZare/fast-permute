#pragma once
#include <Eigen/Sparse>
#include <stdexcept>
#include <vector>
#include "homa/matrix_view.h"
#include "homa/patcher.h"
#include "homa/types.h"
#include "homa/utils/remove_diagonal.h"

namespace homa {

/// Compute a fill-reducing ordering using Homa's patch-based nested
/// dissection. The default patcher is Lloyd-clustering (when built with
/// HOMA_WITH_LLOYD_PATCHER=ON, the default) or METIS-kway as fallback.
/// Tune patching via opts.patch_size, opts.lloyd_iters, opts.lloyd_seed_method.
/// @param n    number of rows/columns
/// @param Gp   CSR row pointers of the GRAPH (diagonal removed), size n+1
/// @param Gi   CSR column indices, size Gp[n]
OrderingResult compute_ordering(int            n,
                                const int*     Gp,
                                const int*     Gi,
                                const Options& opts = {});

/// Convenience overload for sparse matrices. The diagonal is removed and the
/// off-diagonal pattern is symmetrized before ordering. The Scalar argument is
/// only used for the sparsity pattern -- values are not read.
template <class Scalar>
OrderingResult compute_ordering(const Eigen::SparseMatrix<Scalar>& A,
                                const Options&                     opts = {});

/// Convenience overload for host sparse matrix views. Device views are rejected
/// because the current ordering implementation consumes host graph arrays.
template <class Scalar>
OrderingResult compute_ordering(const SparseMatrixView<Scalar>& A,
                                const Options&                  opts = {});

/// Same as above but uses a caller-supplied patch assignment, skipping
/// the internal patching step entirely.
/// @param node_to_patch  node_to_patch[i] = patch index for node i
OrderingResult compute_ordering(int                     n,
                                const int*              Gp,
                                const int*              Gi,
                                const std::vector<int>& node_to_patch,
                                const Options&          opts = {});

}  // namespace homa
