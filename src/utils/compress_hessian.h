//
// Created by Behrooz on 2025-09-10.
//


#pragma once

#include <cholmod.h>
#include <vector>
//Return the factor's nnz using CHOLMOD analysis
namespace RXMESH_SOLVER {
    void compress_hessian(int N,
            int* Ap, int* Ai,
            std::vector<int>& Gp, std::vector<int>& Gi, int dim);
}
