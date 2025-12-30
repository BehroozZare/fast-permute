#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <parth/parth.h>
#include <queue>
#include <unordered_set>
#include <spdlog/spdlog.h>

namespace RXMESH_SOLVER {

    void create_patch_with_parth(int A_n, int* Ap, int* Ai, int DIM,
        int patch_size, std::vector<int>& node_to_patch, std::vector<int>& parth_perm,
        std::vector<int>& new_labels, std::vector<int>& sep_ptr){
        //Use Parth for patch creation
        int G_N = A_n / DIM;
        int num_patches = G_N / patch_size;
        int nd_levels = std::log2(num_patches);
        PARTH::ParthAPI parth;
        parth.setNDLevels(nd_levels);
        parth.setMatrix(A_n, Ap, Ai, DIM);
        parth.computePermutation(parth_perm);

        int* Gp = parth.Mp;
        int* Gi = parth.Mi;



        node_to_patch = parth.hmd.DOF_to_HMD_node;
        std::vector<int> patch_sizes(parth.hmd.HMD_tree.size(), 0);
        //Get the initial patch sizes
        for(int i = 0; i < parth.hmd.HMD_tree.size(); i++){
            patch_sizes[i] = parth.hmd.HMD_tree[i].DOFs.size();
        }

        //Get the initial separator nodes that should be assigned to the patches
        std::queue<int> nodes_to_process;
        int number_of_patches = 0;
        for(int i = 0; i < G_N; i++){
            int patch_id = node_to_patch[i];
            auto& patch = parth.hmd.HMD_tree[patch_id];
            if(patch.isLeaf()){
                number_of_patches++;
                continue;
            }
            node_to_patch[i] = - 1;
            nodes_to_process.push(i);
        }
        
        //Process the nodes to assign them to the patches
        while(!nodes_to_process.empty()){
            int current_node = nodes_to_process.front();
            nodes_to_process.pop();
            assert(node_to_patch[current_node] == -1);
            int start_idx = Gp[current_node];
            int end_idx = Gp[current_node + 1];
            std::unordered_set<int> nbr_patches;
            for(int j = start_idx; j < end_idx; j++){
                int nbr_idx = Gi[j];
                int patch_idx = node_to_patch[nbr_idx];
                if(patch_idx == -1) continue;
                nbr_patches.insert(patch_idx);
            }
            int where_to_id = -1;
            int current_small = G_N + 1;
            for(auto& nbr_patch : nbr_patches){
                int nbr_patch_size = patch_sizes[nbr_patch];
                if(nbr_patch_size < current_small){
                    current_small = nbr_patch_size;
                    where_to_id = nbr_patch;
                }
            }
            if(where_to_id == -1){
                nodes_to_process.push(current_node);
                continue;
            }
            node_to_patch[current_node] = where_to_id;
            patch_sizes[where_to_id]++;
        }

        for(int i = 0; i < node_to_patch.size(); i++){
            int patch_idx = node_to_patch[i];
            auto& patch = parth.hmd.HMD_tree[patch_idx];
            if(patch.isLeaf()){
                continue;
            }
            spdlog::error("Node {} is in patch {} with size {}", i, patch_idx, patch.DOFs.size());
            assert(false);
        }


        //Get the new labels and separator pointers
        auto& etree = parth.hmd.HMD_tree;
        int cnt = 0;
        new_labels.resize(G_N, -1);
        sep_ptr.resize(etree.size() + 1, 0);
        for(int i = 0; i < etree.size(); i++){
            auto& node = etree[i];
            for(int j = 0; j < node.DOFs.size(); j++){
                new_labels[cnt] = node.DOFs[j];
                cnt++;
            }
            sep_ptr[i + 1] = cnt;
        }
        assert(cnt == G_N);
        assert(sep_ptr[etree.size()] == G_N);
    }
}