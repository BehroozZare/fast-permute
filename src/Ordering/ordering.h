//
//  LinSysSolver.hpp
//  IPC
//
//  Created by Minchen Li on 6/30/18.
//
#pragma once


#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cholmod.h>
#include <spdlog/spdlog.h>

namespace RXMESH_SOLVER {

enum class DEMO_ORDERING_TYPE
{
    METIS,
    AMD,
    NEUTRAL,
    PARTH,
    RXMESH_ND,
    PATCH_ORDERING
};

class Ordering
{
public:
    std::vector<int> perm;
    int* Gp = nullptr;
    int* Gi = nullptr;
    int  G_N = 0;
    int  G_NNZ = 0;

public:
    virtual ~Ordering(void) {};

    static Ordering* create(const DEMO_ORDERING_TYPE type);

    virtual DEMO_ORDERING_TYPE type() const = 0;
    virtual std::string typeStr() const = 0;

    virtual void setGraph(int*               Gp,
                           int*              Gi,
                           int               G_N,
                           int               NNZ) = 0;

    // Optional: for orderings that need the original mesh (like RXMesh ND)
    // Pass raw pointers to avoid ABI issues between C++ and CUDA compilation
    virtual void setMesh(const double* V_data, int V_rows, int V_cols,
                        const int* F_data, int F_rows, int F_cols) {}
    
    virtual bool needsMesh() const { return false; }

    virtual void setOptions(const std::map<std::string, std::string>& options) {}

    virtual void init(){
        return;
    }

    // The compute etree is solely for CUDSS ordering where the binary tree order is level order
    virtual void compute_permutation(std::vector<int>& perm, std::vector<int>& etree, bool compute_etree = false) = 0;

    virtual void add_record(std::string save_address, std::map<std::string, double> extra_info, std::string mesh_name) {};

    virtual void reset() {
        spdlog::error("Reset is not implemented for this ordering.");
        return;
    }

    virtual void getEtree(std::vector<int>& new_labels, std::vector<int>& sep_ptr) {};
    virtual void getStatistics(std::map<std::string, double>& stat) {};
    virtual void getPatch(std::vector<int>& patches) {};
};

}  // namespace PARTH_SOLVER
