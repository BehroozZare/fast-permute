//
// Created by behrooz on 2025-10-07.
//
#pragma once
#include "Eigen/Core"
#include "Eigen/Sparse"
#include <spdlog/spdlog.h>

namespace RXMESH_SOLVER {

class CPUOrdering_PATCH
{
public:
    CPUOrdering_PATCH();
    ~CPUOrdering_PATCH();
    // Think of this as the input of decompose function in recursive manner
    struct DecompositionInfo
    {
        int              decomposition_node_parent_id = -1;
        std::vector<int> assigned_g_nodes;
    };

    struct DecompositionNode
    {
        int left_node_idx = -1;
        int right_node_idx = -1;
        int node_id = -1;
        int parent_idx = -1;
        int level = -1;
        int offset = -1;
        std::vector<int> assigned_g_nodes;
        std::vector<int> local_new_labels;
        bool is_initialized = false;

        bool isLeaf() const
        {
            if (left_node_idx == -1 && right_node_idx == -1)
                return true;
            return false;
        }

        
        void init_node(int left_node_idx, int right_node_idx,
            int node_id, int parent_idx, int level, std::vector<int> & assigned_g_nodes)
        {
            this->left_node_idx = left_node_idx;
            this->right_node_idx = right_node_idx;
            this->parent_idx = parent_idx;
            this->node_id = node_id;
            this->level = level;
            this->assigned_g_nodes = assigned_g_nodes;
            this->is_initialized = true;
        }

        void set_local_permutation(std::vector<int> & local_permutation) {
            this->local_new_labels.resize(assigned_g_nodes.size());
            for(int i = 0; i < local_permutation.size(); ++i) {
                assert(local_permutation[i] < local_permutation.size());
                this->local_new_labels[local_permutation[i]] = i;
            }
        }

        void set_offset(int offset) {
            this->offset = offset;
        }
    };

    struct DecompositionTree {
        int decomposition_level = -1;
        std::vector<DecompositionNode> decomposition_nodes; // All the nodes of the tree
        std::vector<int> decomposition_node_offset; // The offset of the nodes in the tree for permutation
        std::vector<char> is_sep;
        std::vector<int> g_node_to_tree_node;
        int _num_patches = -1;

        void init_decomposition_tree(int num_decomposition_nodes,
            int decomposition_level, int total_nodes, int num_patches) {
#ifndef NDEBUG
            spdlog::info("Decomposition tree creation .. ");
            spdlog::info("Number of decomposition levels: {}", decomposition_level);
#endif
            this->_num_patches = num_patches;
            this->decomposition_nodes.resize(num_decomposition_nodes);
            this->decomposition_level = decomposition_level;
            this->g_node_to_tree_node.resize(total_nodes, 0);
            this->is_sep.clear();
            this->is_sep.resize(total_nodes, 0);
        }
        inline int get_number_of_decomposition_nodes() const {
            return decomposition_nodes.size();
        }

        inline bool is_separator(const int node_id) const
        {
            return this->is_sep[node_id];
        }

        inline void assign_nodes_to_tree(
            const std::vector<int>& separator_g_nodes, int tree_node_id) {
            //flag the separator nodes in the decomposition tree
            for(int separator_g_node : separator_g_nodes) {
                assert(separator_g_node < g_node_to_tree_node.size());
                assert(is_sep.size() == g_node_to_tree_node.size());
                assert(!this->is_separator(separator_g_node));
                this->is_sep[separator_g_node] = 1;
                this->g_node_to_tree_node[separator_g_node] = tree_node_id;
            }
        }
    };

    struct QuotientGraph {
        int _Q_n;
        std::vector<int> _Q_edge_weights;
        std::vector<int> _Q_node_weights;
        std::vector<int> _Qp;
        std::vector<int> _Qi;
    };    

    struct SubGraph {
        int _num_nodes;
        std::vector<int> _Gp;
        std::vector<int> _Gi;
    };

    QuotientGraph _quotient_graph;
    bool use_patch_separator = true;
    std::string local_permute_method = "amd";
    // std::string separator_refinement_method = "nothing";
    DecompositionTree _decomposition_tree;
    int _decomposition_max_level;

    int _num_patches = -1;
    std::vector<int> _g_node_to_patch;
    int _G_n, _G_nnz;
    int* _Gp, *_Gi;

    double node_to_patch_time = 0;
    double decompose_time = 0;
    double local_permute_time = 0;
    double assemble_time = 0;
    double quotient_graph_creation_time = 0;

    double _separator_ratio = 0.0;

    std::string etree_order = "post_order";//Accept level_order and post_order
    std::vector<int> etree; 
    void setGraph(int* Gp, int* Gi, int G_N, int NNZ);

    //This function updates the q_node_to_tree_node map for the current decomposition node
    //The left and right q nodes are mark as tree_node_idx * 2 + 1 and tree_node_idx * 2 + 2 respectively
    void two_way_Q_partition(int tree_node_idx,///<[in] The index of the current decomposition node
        std::vector<int>& assigned_g_nodes,///<[in] Assigned G nodes for current decomposition
        std::vector<int>& where///<[out] a vector with the size of assigned_g_node
    );
  
    void compute_bipartition(
        int Q_n,
        int* Qp,
        int* Qi,
        std::vector<int>&         Q_node_weights,///<[in] The node weights of the graph
        std::vector<int>&         Q_partition_map///<[out] The partition map of the graph
    );


    void find_separator_superset(
        std::vector<int>& assigned_g_nodes,///<[in] Assigned G nodes for current decomposition
        std::vector<int>& map,///<[in] a mapping from nodes to their partition -> unassigned nodes get -1 as parition id
        std::vector<int>& separator_superset///<[out] The superset of separator nodes
    );

    //Given a set of separator superset, from G, extracts the bipartite graph and refines the separator nodes
    //Based on the Hopcroft-Karp algorithm
    void refine_bipartate_separator(
        int tree_node_idx,///<[in] The tree decomposition node
        std::vector<int>& separator_superset///<[in] The separator superset
        );

    //Refine separator using METIS FM-based node refinement
    //Performs balance pass + refinement pass similar to METIS separator refinement
    void refine_separator_metis(
        int tree_node_idx,///<[in] The tree decomposition node
        std::vector<int>& assigned_g_nodes,///<[in] All nodes in current decomposition level
        std::vector<int>& separator_g_nodes,///<[in/out] The separator nodes to refine
        std::vector<int>& where///<[out] The where array
        );

    //Given a subset of nodes (filtered nodes are marked with -1), partition the graph into three parts: left, right, and separator
    void three_way_G_partition(int tree_node_idx,///<[in] The index of the current decomposition node
        std::vector<int>& assigned_g_nodes,///<[in] Assigned G nodes for current decomposition
        std::vector<int>& where///<[out] the partition assignment where = 2 is separator, 0, and 1 is left and right
    );

    void decompose();

    void compute_local_quotient_graph(
        std::vector<int>& assigned_g_node,///<[in] The index of the current decomposition node
        int& local_Q_n,
        std::vector<int>& local_Qp,
        std::vector<int>& local_Qi,
        std::vector<int>& Q_node_weights,///<[out] The node weights of the local quotient graph
        std::vector<int>& Q_node_to_global_Q_node///<[out] The local Q node to global Q node map
    );

    void local_permute_metis(int G_n, int* Gp, int* Gi,
        std::vector<int> & local_permutation);

    void local_permute_amd(int G_n, int* Gp, int* Gi,
        std::vector<int> & local_permutation);

    void local_permute_unity(int G_n, int* Gp, int* Gi,
        std::vector<int>& local_permutation);

    //Apply fill-in reducing ordering to each node in the decomposition tree
    void local_permute(int G_n, int* Gp, int* Gi,
        std::vector<int> & local_permutation);

    //Given the full binary tree, it numbers each decomposition node in a post-order manner
    int post_order_offset_computation(int offset, int decomposition_node_id);

    //Given the full binary tree, it numbers each decomposition node in the tree starting from 
    //Leaves to the root
    void level_order_offset_computation();

    void compute_sub_graph(
        std::vector<int>&         nodes,
        Eigen::SparseMatrix<int>& local_graph) const;

    //Build a CSR subgraph from a set of global node IDs
    void build_subgraph_csr(
        const std::vector<int>& assigned_g_nodes,  ///< [in] Global node IDs to include
        SubGraph& subgraph                          ///< [out] Populated subgraph in CSR format
    ) const;

    void compute_sub_graphs(std::vector<SubGraph>& sub_graphs);

    double compute_separator_ratio();

    void init_patches(int num_patches,std::vector<int> & g_node_to_patch, int num_levels=9);
    void step1_compute_quotient_graph();
    void step2_create_decomposition_tree();
    void step3_compute_local_permutations();
    void step4_assemble_final_permutation(std::vector<int>& perm);

    void compute_permutation(std::vector<int>& perm);
    void reset();
};
}