//
// Created by Behrooz on 2025-09-10.
//

#include "compress_hessian.h"
#include <cassert>
#include <iostream>
#include <tuple>
#include <vector>
#include <algorithm>

namespace RXMESH_SOLVER {
//Compress the hessian matrix by collapsing each Dim consecutive rows/columns into 1
void compress_hessian(int N,
    int* Ap, int* Ai,
    std::vector<int>& Gp, std::vector<int>& Gi, int dim)
{
  assert(N % dim == 0);
  this->Gp.clear();
  this->Gp.resize(N / dim + 1, 0);
  std::vector<std::tuple<int, int>> coefficients;
  for (int c = 0; c < N; c += dim) {
    assert((Ap[c + 1] - Ap[c]) % dim == 0);
    for (int r_ptr = Ap[c]; r_ptr < Ap[c + 1]; r_ptr += dim) {
      int r = Ai[r_ptr];
      int mesh_c = c / dim;
      int mesh_r = r / dim;
      if (mesh_c != mesh_r) {
        coefficients.emplace_back(mesh_c, mesh_r);
        coefficients.emplace_back(mesh_r, mesh_c);
      }
    }
  }

  //Remove duplicates
  std::ranges::sort(coefficients);
  coefficients.erase(std::ranges::unique(coefficients).begin(), coefficients.end());

  for (int i = 0; i < coefficients.size(); i++) {
    Gp[std::get<0>(coefficients[i]) + 1]++;
  }
  for (int i = 1; i < Gp.size(); i++) {
    Gp[i] += Gp[i - 1];
  }

  Gi.resize(Gp.back());
  std::vector<int> Gp_cnt(Gp.size(), 0);
  for (int i = 0; i < coefficients.size(); i++) {
    int row = std::get<0>(coefficients[i]);
    int col = std::get<1>(coefficients[i]);
    Gi[Gp[row] + Gp_cnt[row]] = col;
    Gp_cnt[row]++;
  }

  #ifndef NDEBUG
  //Make sure that each row is sorted
  for(int r = 0; r < Gp.size() - 1; r++) {
    for (int i = Gp[r]; i < Gp[r + 1] - 1; i++) {
      assert(Gi[i] < Gi[i + 1]);
    }
  }
  #endif

  int G_n = Gp.size() - 1;
  Gp = Gp.data();
  Gi = Gi.data();
  return;
}

}
