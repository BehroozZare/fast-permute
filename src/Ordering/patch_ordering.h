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
#include "ordering.h"
#include "gpu_ordering_with_patch.h"
#include "cpu_ordering_with_patch.h"
#include <rxmesh/rxmesh_static.h>

namespace RXMESH_SOLVER {


class PatchOrdering: public Ordering
{
private:
    std::vector<std::vector<uint32_t>> _fv;
    std::vector<std::vector<float>> _vertices;
    int _patch_size = 512;
    bool m_has_mesh = false;

    enum class PatchOrderingType {
        RXMESH_PATCH,
        METIS_KWAY_PATCH,
        METIS_SEPARATOR_PATCH
    };

    bool _use_gpu = false;
    PatchOrderingType _patch_ordering_type = PatchOrderingType::RXMESH_PATCH;
    std::unique_ptr<rxmesh::RXMeshStatic> _rxmesh;
    GPUOrdering_PATCH _gpu_order;
    CPUOrdering_PATCH _cpu_order;
    
    int _num_patches = -1;
    std::vector<int> _g_node_to_patch;

    
public:
    virtual ~PatchOrdering(void);

    virtual DEMO_ORDERING_TYPE type() const override;
    virtual std::string typeStr() const override;

    virtual void setGraph(int*              Gp,
                           int*              Gi,
                           int               G_N,
                           int               NNZ) override;

    virtual void setMesh(const double* V_data, int V_rows, int V_cols,
                        const int* F_data, int F_rows, int F_cols) override;
    
    virtual void init() override;
    virtual bool needsMesh() const override;

    //Given the binary tree order, it assembles the permutation
    void assemble_perm(std::vector<int>& level_numbering, std::vector<int>& perm);
    //The level_numbering[hmd_id] is going to be the sequence number in final permutation
    //For example the root should be last so level_numbering[0] = last_node
    void get_level_numbering(int binary_tree_size, std::vector<int>& level_numbering);
    //Given the binary tree order, it computes the etree
    void compute_etree(std::vector<int>& level_numbering, std::vector<int>& etree);
    
    virtual void compute_permutation(std::vector<int>& perm, std::vector<int>& etree, bool compute_etree = false) override;

    virtual void setOptions(const std::map<std::string, std::string>& options) override;

    virtual void add_record(std::string save_address, std::map<std::string, double> extra_info, std::string mesh_name) override;

    double compute_separator_ratio();
};

}  // namespace RXMESH_SOLVER
