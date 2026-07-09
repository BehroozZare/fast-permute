//
// Created by Behrooz on 2025-09-10.
//

#include "homa/utils/check_valid_permutation.h"
#include <cassert>
#include <iostream>
#include <tuple>
#include <vector>
#include <algorithm>

#include <spdlog/spdlog.h>

namespace homa {
// Return the factor's nnz using CHOLMOD analysis. The input matrix should be
// CSC with only lower part represented.
bool check_valid_permutation(int* perm, int n)
{
    std::vector<bool> marker(n, false);
    for (int i = 0; i < n; i++) {
        if (perm[i] < 0 || perm[i] >= n) {
            spdlog::error(
                "ERROR: Invalid permutation. Element {} has value {} which is "
                "out of range [0, {}}",
                i,
                perm[i],
                n - 1);
            return false;
        }
        if (marker[perm[i]]) {
            spdlog::error(
                "ERROR: Invalid permutation. Element {} appears more than "
                "once.",
                perm[i]);
            return false;
        }
        marker[perm[i]] = true;
    }

    // Check to see there is no marker left
    for (int i = 0; i < n; i++) {
        if (marker[i] == false) {
            spdlog::error("ERROR: Invalid permutation. Element {} is missing.",
                          i);
            return false;
        }
    }
    return true;
}

}  // namespace homa
