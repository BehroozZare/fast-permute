#pragma once
#include <string>
#include <vector>

namespace homa {

// Build a CSC sparsity graph (Gp, Gi) from a CSC matrix (Ap, Ai) of size N x N
// by removing the diagonal entries and symmetrizing the off-diagonal pattern.
void remove_diagonal(int N,
                     const int* Ap, const int* Ai,
                     std::vector<int>& Gp, std::vector<int>& Gi);

// Compare the off-diagonal, symmetrized sparsity patterns of two CSC matrices.
bool compare_sparsity_no_diagonal(int N,
                                  const int* Ap, const int* Ai,
                                  const int* Bp, const int* Bi,
                                  const std::string& label_a = "A",
                                  const std::string& label_b = "B",
                                  int max_diffs = 20);

} // namespace homa
