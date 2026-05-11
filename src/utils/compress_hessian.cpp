#include "compress_hessian.h"
#include <algorithm>
#include <cassert>
#include <tuple>
#include <vector>

namespace homa {

// Compress a CSC matrix by merging groups of `dim` consecutive DOFs into a
// single graph node. Builds the symmetric off-diagonal sparsity pattern.
void compress_hessian(int N,
                      const int* Ap, const int* Ai,
                      std::vector<int>& Gp, std::vector<int>& Gi, int dim)
{
    assert(N % dim == 0);
    const int Gn = N / dim;

    std::vector<std::pair<int, int>> coefficients;
    for (int c = 0; c < N; c += dim) {
        for (int r_ptr = Ap[c]; r_ptr < Ap[c + 1]; r_ptr += dim) {
            int mesh_c = c / dim;
            int mesh_r = Ai[r_ptr] / dim;
            if (mesh_c != mesh_r) {
                coefficients.emplace_back(mesh_c, mesh_r);
                coefficients.emplace_back(mesh_r, mesh_c);
            }
        }
    }

    std::sort(coefficients.begin(), coefficients.end());
    coefficients.erase(std::unique(coefficients.begin(), coefficients.end()),
                       coefficients.end());

    Gp.assign(Gn + 1, 0);
    for (const auto& e : coefficients)
        Gp[e.first + 1]++;
    for (int i = 1; i <= Gn; ++i)
        Gp[i] += Gp[i - 1];

    Gi.resize(Gp.back());
    std::vector<int> cnt(Gn, 0);
    for (const auto& e : coefficients) {
        Gi[Gp[e.first] + cnt[e.first]++] = e.second;
    }

#ifndef NDEBUG
    for (int r = 0; r < Gn; ++r)
        for (int i = Gp[r]; i < Gp[r + 1] - 1; ++i)
            assert(Gi[i] < Gi[i + 1]);
#endif
}

} // namespace homa
