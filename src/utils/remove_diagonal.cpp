//
// Created by Behrooz on 2025-09-10.
//

#include "remove_diagonal.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <tuple>
#include <vector>

namespace homa {

namespace {

// Build a sorted, deduped vector of (col, row) off-diagonal symmetric pairs
// from a CSC matrix (Ap, Ai) of size N x N.
std::vector<std::tuple<int, int>> build_offdiag_pairs(int N, const int* Ap, const int* Ai)
{
    const int dim = 1;
    std::vector<std::tuple<int, int>> coefficients;
    for (int c = 0; c < N; c += dim) {
        assert((Ap[c + 1] - Ap[c]) % dim == 0);
        for (int r_ptr = Ap[c]; r_ptr < Ap[c + 1]; r_ptr += dim) {
            int r      = Ai[r_ptr];
            int mesh_c = c / dim;
            int mesh_r = r / dim;
            if (mesh_c != mesh_r) {
                coefficients.emplace_back(mesh_c, mesh_r);
                coefficients.emplace_back(mesh_r, mesh_c);
            }
        }
    }

    std::sort(coefficients.begin(), coefficients.end());
    coefficients.erase(
        std::unique(coefficients.begin(), coefficients.end()),
        coefficients.end());
    return coefficients;
}

}  // namespace

void remove_diagonal(int N,
                     const int* Ap, const int* Ai,
                     std::vector<int>& Gp, std::vector<int>& Gi)
{
    std::vector<std::tuple<int, int>> coefficients =
        build_offdiag_pairs(N, Ap, Ai);

    Gp.assign(N + 1, 0);
    for (size_t i = 0; i < coefficients.size(); i++) {
        Gp[std::get<0>(coefficients[i]) + 1]++;
    }
    for (size_t i = 1; i < Gp.size(); i++) {
        Gp[i] += Gp[i - 1];
    }

    Gi.resize(Gp.back());
    std::vector<int> Mp_vec_cnt(Gp.size(), 0);
    for (size_t i = 0; i < coefficients.size(); i++) {
        int row = std::get<0>(coefficients[i]);
        int col = std::get<1>(coefficients[i]);
        Gi[Gp[row] + Mp_vec_cnt[row]] = col;
        Mp_vec_cnt[row]++;
    }

#ifndef NDEBUG
    for (size_t r = 0; r + 1 < Gp.size(); r++) {
        for (int i = Gp[r]; i < Gp[r + 1] - 1; i++) {
            assert(Gi[i] < Gi[i + 1]);
        }
    }
#endif
}

bool compare_sparsity_no_diagonal(int N,
                                  const int* Ap, const int* Ai,
                                  const int* Bp, const int* Bi,
                                  const std::string& label_a,
                                  const std::string& label_b,
                                  int max_diffs)
{
    std::vector<std::tuple<int, int>> pairs_a =
        build_offdiag_pairs(N, Ap, Ai);
    std::vector<std::tuple<int, int>> pairs_b =
        build_offdiag_pairs(N, Bp, Bi);

    size_t only_in_a = 0;
    size_t only_in_b = 0;
    size_t shared    = 0;

    std::vector<std::tuple<int, int>> diffs_a;
    std::vector<std::tuple<int, int>> diffs_b;
    diffs_a.reserve(std::min<size_t>(max_diffs, pairs_a.size()));
    diffs_b.reserve(std::min<size_t>(max_diffs, pairs_b.size()));

    size_t i = 0;
    size_t j = 0;
    while (i < pairs_a.size() && j < pairs_b.size()) {
        if (pairs_a[i] == pairs_b[j]) {
            ++shared;
            ++i;
            ++j;
        } else if (pairs_a[i] < pairs_b[j]) {
            ++only_in_a;
            if (static_cast<int>(diffs_a.size()) < max_diffs) {
                diffs_a.push_back(pairs_a[i]);
            }
            ++i;
        } else {
            ++only_in_b;
            if (static_cast<int>(diffs_b.size()) < max_diffs) {
                diffs_b.push_back(pairs_b[j]);
            }
            ++j;
        }
    }
    while (i < pairs_a.size()) {
        ++only_in_a;
        if (static_cast<int>(diffs_a.size()) < max_diffs) {
            diffs_a.push_back(pairs_a[i]);
        }
        ++i;
    }
    while (j < pairs_b.size()) {
        ++only_in_b;
        if (static_cast<int>(diffs_b.size()) < max_diffs) {
            diffs_b.push_back(pairs_b[j]);
        }
        ++j;
    }

    spdlog::info(
        "compare_sparsity_no_diagonal: N={}, |{}|={}, |{}|={}, shared={}, "
        "{}_only={}, {}_only={}",
        N, label_a, pairs_a.size(), label_b, pairs_b.size(), shared,
        label_a, only_in_a, label_b, only_in_b);

    if (only_in_a != 0 || only_in_b != 0) {
        for (const auto& p : diffs_a) {
            spdlog::warn("{} only: col={}, row={}",
                         label_a, std::get<0>(p), std::get<1>(p));
        }
        for (const auto& p : diffs_b) {
            spdlog::warn("{} only: col={}, row={}",
                         label_b, std::get<0>(p), std::get<1>(p));
        }
        return false;
    }
    return true;
}

}  // namespace homa
