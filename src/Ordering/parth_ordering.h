//
//  LinSysSolver.hpp
//  IPC
//
//  Created by Minchen Li on 6/30/18.
//
#pragma once


#ifdef USE_PARTH
#include <Eigen/Core>
#include <parth//parth.h>
#include "ordering.h"

namespace RXMESH_SOLVER {


class ParthOrdering: public Ordering
{
public:
    PARTH::ParthAPI parth;
    virtual ~ParthOrdering(void);

    int patch_size = 512;
    int nd_levels = 9;
    virtual DEMO_ORDERING_TYPE type() const override;
    virtual std::string typeStr() const override;

    virtual void setGraph(int*              Gp,
                           int*              Gi,
                           int               G_N,
                           int               NNZ) override;

    //Given the binary tree order, it assembles the permutation
    void assemble_perm(std::vector<int>& level_numbering, std::vector<int>& perm);
    //The level_numbering[hmd_id] is going to be the sequence number in final permutation
    //For example the root should be last so level_numbering[0] = last_node
    void get_level_numbering(int binary_tree_size, std::vector<int>& level_numbering);
    //Given the binary tree order, it computes the etree
    void compute_etree(std::vector<int>& level_numbering, std::vector<int>& etree);

    virtual void compute_permutation(std::vector<int>& perm, std::vector<int>& etree, bool compute_etree = false) override;
    virtual void setOptions(const std::map<std::string, std::string>& options) override;
    void computeRatioOfBoundaryVertices();
    void computeTheStatisticsOfPatches();
};



}  // namespace PARTH_SOLVER
#endif
