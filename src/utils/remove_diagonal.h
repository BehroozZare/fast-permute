//
// Created by Behrooz on 2025-09-10.
//


#pragma once

#include <cholmod.h>
#include <string>
#include <vector>

namespace RXMESH_SOLVER {

// Build a CSC sparsity graph (Gp, Gi) from a CSC matrix (Ap, Ai) of size N x N
// by removing the diagonal entries and symmetrizing the off-diagonal pattern.
void remove_diagonal(int N,
                     int* Ap, int* Ai,
                     std::vector<int>& Gp, std::vector<int>& Gi);

// Compare the off-diagonal, symmetrized sparsity patterns of two CSC matrices
// A and B of size N x N. Returns true iff the patterns are identical after
// self-loops are removed and the pattern is symmetrized. On mismatch, logs
// summary counts and up to `max_diffs` differing (col, row) pairs from each
// side via spdlog. `label_a` / `label_b` are used only in log messages.
bool compare_sparsity_no_diagonal(int N,
                                  int* Ap, int* Ai,
                                  int* Bp, int* Bi,
                                  const std::string& label_a = "A",
                                  const std::string& label_b = "B",
                                  int max_diffs = 20);

}  // namespace RXMESH_SOLVER
