#pragma once
#include <vector>

namespace homa {
    void compress_hessian(int N,
                          const int* Ap, const int* Ai,
                          std::vector<int>& Gp, std::vector<int>& Gi, int dim);
}
