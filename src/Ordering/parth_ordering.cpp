//
// Created by behrooz on 2025-09-29.
//

#include "parth_ordering.h"

#include "ordering.h"
#include "spdlog/spdlog.h"

namespace RXMESH_SOLVER {

ParthOrdering::~ParthOrdering()
{
}

void ParthOrdering::setGraph(int* Gp, int* Gi, int G_N, int NNZ)
{
    parth.setNDLevels(4);
    parth.setMatrix(G_N, Gp, Gi, 1);
}

void ParthOrdering::setOptions(const std::map<std::string, std::string>& options)
{
    if(options.find("binary_tree_order") != options.end()) {
        binary_tree_order = options.at("binary_tree_order");
    }
}


void ParthOrdering::get_level_numbering(int binary_tree_size, std::vector<int>& level_numbering) {
    level_numbering.clear();
    level_numbering.resize(binary_tree_size, 0);
    for(int i = 0; i < binary_tree_size; i++){
        int hmd_id = binary_tree_size - 1 - i;
        level_numbering[hmd_id] = i;
    }
}

void ParthOrdering::compute_etree(std::vector<int>& level_numbering, std::vector<int>& etree) {
    etree.clear();
    etree.resize(level_numbering.size(), 0);
    for (int hmd_id = 0; hmd_id < level_numbering.size(); hmd_id++) {
        int etree_idx = level_numbering[hmd_id];
        int etree_value = parth.hmd.HMD_tree[hmd_id].DOFs.size();
        etree[etree_idx] = etree_value;
    }
}

void ParthOrdering::assemble_perm(std::vector<int>& level_numbering, std::vector<int>& perm) {
    perm.clear();
    perm.resize(parth.M_n, -1);
    std::vector<int> etree_inverse(level_numbering.size(), 0);
    for(int hmd_id = 0; hmd_id < level_numbering.size(); hmd_id++){
        etree_inverse[level_numbering[hmd_id]] = hmd_id;
    }

    int offset = 0;
    for(int i = 0; i < etree_inverse.size(); i++){
        int hmd_id = etree_inverse[i];
        auto& node = parth.hmd.HMD_tree[hmd_id];
        if (node.DOFs.empty())
            continue;
        for (int local_node = 0; local_node < node.DOFs.size(); local_node++) {
            int global_node = node.DOFs[local_node];
            int perm_index  = node.permuted_new_label[local_node] + offset;
            assert(global_node >= 0 && global_node < parth.M_n &&
                    "Invalid global node index");
            assert(perm_index >= 0 && perm_index < perm.size() &&
                    "Permutation index out of bounds");
            assert(perm[perm_index] == -1 &&
                    "Permutation slot already filled - duplicate node!");
            perm[perm_index] = global_node;
        }
        offset += node.DOFs.size();
    }
}

void ParthOrdering::compute_permutation(std::vector<int>& perm, std::vector<int>& etree, bool with_etree)
{
    parth.computePermutation(perm, 1);  // 1 = post_order mode, required for correct HMD tree state

    //Apply mapping to the permutation
    if(with_etree) {
        spdlog::info("Computing Etree for CUDSS");
        std::vector<int> level_numbering;
        perm.clear();
        get_level_numbering(parth.hmd.HMD_tree.size(), level_numbering);
        compute_etree(level_numbering, etree);
        assemble_perm(level_numbering, perm);
    }
}


DEMO_ORDERING_TYPE ParthOrdering::type() const
{
    return  DEMO_ORDERING_TYPE::PARTH;
}
std::string ParthOrdering::typeStr() const
{
    return "PARTH";
}


void ParthOrdering::computeRatioOfBoundaryVertices()
{
    int boundary_vertices = 0;
    for (auto& node: parth.hmd.HMD_tree) {
        if (!node.isLeaf()) {
            boundary_vertices+= node.DOFs.size();
        }
    }
    spdlog::info("Total number of boundary vertices: {}", boundary_vertices);
    spdlog::info("The ratio of boundary vertices to total vertices is {:.2f}%",
                 (boundary_vertices * 100.0) / parth.M_n);

}

void ParthOrdering::computeTheStatisticsOfPatches()
{
    int max_patch_size = 0;
    int min_patch_size = 1e9;
    double avg_patch_size = 0;
    for (auto& patch: parth.hmd.HMD_tree) {
        if (patch.isLeaf()) {
            int patch_size = patch.DOFs.size();
            if (patch_size > max_patch_size) {
                max_patch_size = patch_size;
            }
            if (patch_size < min_patch_size) {
                min_patch_size = patch_size;
            }
            avg_patch_size += patch_size;
        }
    }
    spdlog::info("Total number of patches including separators: {}", parth.hmd.HMD_tree.size());
    spdlog::info("The max patch size is {}", max_patch_size);
    spdlog::info("The min patch size is {}", min_patch_size);
    spdlog::info("The avg patch size is {:.2f}", avg_patch_size / parth.hmd.HMD_tree.size());
}


}