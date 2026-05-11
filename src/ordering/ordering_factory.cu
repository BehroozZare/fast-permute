#include "ordering_factory.h"
#include <cassert>
#include <iostream>
#include <metis.h>
#include <spdlog/spdlog.h>

#include "neutral_ordering.h"
#include "patch_ordering.h"
#include "parth_ordering.h"
#include "../patching/metis_patcher.h"
#include "../patching/rxmesh_patcher.h"

namespace homa {

// Thin shim: METIS ND full-graph ordering used by benchmarks as a baseline.
// Uses METIS_NodeND directly; distinct from MetisPatcher (k-way partitioning).
class MetisOrderingShim : public Ordering {
public:
    DEMO_ORDERING_TYPE type() const override { return DEMO_ORDERING_TYPE::METIS; }
    std::string typeStr() const override { return "METIS"; }

    void setGraph(int* Gp_, int* Gi_, int G_N_, int NNZ_) override {
        Gp = Gp_; Gi = Gi_; G_N = G_N_; G_NNZ = NNZ_;
    }

    void compute_permutation(std::vector<int>& out_perm, std::vector<int>& etree,
                             bool /*compute_etree*/) override {
        idx_t N = G_N;
        out_perm.resize(G_N);
        if (Gp[G_N] == 0) {
            for (int i = 0; i < G_N; ++i) out_perm[i] = i;
            etree.clear();
            return;
        }
        std::vector<int> tmp(G_N);
        METIS_NodeND(&N, Gp, Gi, nullptr, nullptr, out_perm.data(), tmp.data());
        etree.clear();
    }
};

// Thin shim: wraps RxMeshPatcher to keep RXMESH_ND in the factory.
class RXMeshOrderingShim : public Ordering {
    homa::RxMeshPatcher _patcher;
    bool m_has_mesh = false;
public:
    DEMO_ORDERING_TYPE type() const override { return DEMO_ORDERING_TYPE::RXMESH_ND; }
    std::string typeStr() const override { return "RXMesh_ND"; }
    bool needsMesh() const override { return true; }

    void setGraph(int* Gp_, int* Gi_, int G_N_, int NNZ_) override {
        Gp = Gp_; Gi = Gi_; G_N = G_N_; G_NNZ = NNZ_;
    }

    void setMesh(const double* V_data, int V_rows, int V_cols,
                 const int* F_data, int F_rows, int F_cols) override {
        m_has_mesh = true;
        _patcher.setMesh(V_data, V_rows, V_cols, F_data, F_rows, F_cols);
    }

    void compute_permutation(std::vector<int>& out_perm, std::vector<int>& etree,
                             bool /*compute_etree*/) override {
        if (!m_has_mesh) {
            spdlog::error("RXMeshOrderingShim: no mesh supplied, returning identity");
            out_perm.resize(G_N);
            for (int i = 0; i < G_N; ++i) out_perm[i] = i;
            etree.clear();
            return;
        }
        _patcher.compute(G_N, Gp, Gi, out_perm);
        etree.clear();
    }
};

Ordering* Ordering::create(const DEMO_ORDERING_TYPE type) {
    switch (type) {
        case DEMO_ORDERING_TYPE::METIS:
            return new MetisOrderingShim();
        case DEMO_ORDERING_TYPE::RXMESH_ND:
            return new RXMeshOrderingShim();
        case DEMO_ORDERING_TYPE::PATCH_ORDERING:
            return new PatchOrdering();
        case DEMO_ORDERING_TYPE::NEUTRAL:
            return new NeutralOrdering();
#ifdef USE_PARTH
        case DEMO_ORDERING_TYPE::PARTH:
            return new ParthOrdering();
#endif
        default:
            std::cerr << "Unknown Ordering type" << std::endl;
            return nullptr;
    }
}

} // namespace homa