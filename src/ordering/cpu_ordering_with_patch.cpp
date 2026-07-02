//
// Created by behrooz on 2025-10-07.
//
#include <metis.h>
#include <cassert>
#include <cmath>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include "amd_order_helper.h"
#include "cpu_ordering_with_patch.h"

#include "spdlog/spdlog.h"

namespace homa {
namespace {

using Clock = std::chrono::high_resolution_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int tree_level_from_id(int tree_node_idx)
{
    int level = 0;
    int node = tree_node_idx + 1;
    while (node > 1) {
        node >>= 1;
        level++;
    }
    return level;
}

} // namespace

CPUOrdering_PATCH::CPUOrdering_PATCH() : _Gp(nullptr), _Gi(nullptr), _G_n(0), _G_nnz(0)
{
}

CPUOrdering_PATCH::~CPUOrdering_PATCH()
{
}

void CPUOrdering_PATCH::setGraph(int* Gp, int* Gi, int G_N, int NNZ)
{
    this->_Gp    = Gp;
    this->_Gi    = Gi;
    this->_G_n   = G_N;
    this->_G_nnz = NNZ;

}


void CPUOrdering_PATCH::local_permute_metis(int G_n, int* Gp, int* Gi,
                                         std::vector<int>& local_permutation)
{
    idx_t N   = G_n;
    idx_t NNZ = Gp[G_n];
    local_permutation.resize(N);
    if (NNZ == 0) {
        assert(N != 0);
        for (int i = 0; i < N; i++) {
            local_permutation[i] = i;
        }
        return;
    }

    std::vector<int> tmp(N);
    
    int status = METIS_NodeND(&N,
                              Gp,
                              Gi,
                              NULL,
                              NULL,
                              local_permutation.data(),
                              tmp.data());
    if (status != METIS_OK) {
        spdlog::error("METIS_NodeND failed (status={})", status);
        throw std::runtime_error("METIS_NodeND failed");
    }
}

void CPUOrdering_PATCH::local_permute_amd(int G_n, int* Gp, int* Gi,
                                       std::vector<int>& local_permutation)
{
    idx_t N   = G_n;
    idx_t NNZ = Gp[G_n];
    local_permutation.resize(N);
    if (NNZ == 0) {
        assert(N != 0);
        for (int i = 0; i < N; i++) {
            local_permutation[i] = i;
        }
        return;
    }
    std::vector<int> tmp(N);
    homa::detail::amd_local_order(N, Gp, Gi, local_permutation.data());
}

void CPUOrdering_PATCH::local_permute_unity(int G_n, int* Gp, int* Gi,
                                         std::vector<int>& local_permutation)
{
    local_permutation.resize(G_n);
    for (int i = 0; i < G_n; i++) {
        local_permutation[i] = i;
    }
}

void CPUOrdering_PATCH::local_permute(int G_n, int* Gp, int* Gi,
                                   std::vector<int>&         local_permutation)
{
    if (this->local_permute_method == "metis") {
        local_permute_metis(G_n, Gp, Gi, local_permutation);
    } else if (this->local_permute_method == "amd") {
        //local_permute_amd(G_n, Gp, Gi, local_permutation);
        const double global_avg_degree =
            this->_G_n > 0
                ? static_cast<double>(this->_G_nnz) / static_cast<double>(this->_G_n)
                : 0.0;
        const int serial_amd_min_nodes =
            (this->_G_n < 3000 || global_avg_degree >= 5.5) ? 256 : 512;
        const int amd_min_nodes =
            use_serial_ordering_path() ? serial_amd_min_nodes
                                       : this->local_amd_min_nodes;
        if (G_n <= amd_min_nodes) {
            local_permute_unity(G_n, Gp, Gi, local_permutation);
        } else {
            local_permute_amd(G_n, Gp, Gi, local_permutation);
        }
    } else if (this->local_permute_method == "unity") {
        local_permute_unity(G_n, Gp, Gi, local_permutation);
    } else {
        spdlog::error("Invalid local permutation method: {}",
                      this->local_permute_method);
        return;
    }
}

void CPUOrdering_PATCH::compute_local_quotient_graph(
    std::vector<int>& assigned_g_node,///<[in] The index of the current decomposition node
    int& local_Q_n,
    std::vector<int>& local_Qp,
    std::vector<int>& local_Qi,
    std::vector<int>& local_Q_node_weights,///<[out] The node weights of the local quotient graph
    std::vector<int>& q_local_to_global_map///<[in/out] The map from current assigned quotient nodes to tree nodes
){
    local_Qp.clear();
    local_Qi.clear();
    local_Q_node_weights.clear();
    q_local_to_global_map.clear();
    //Find the Qs in this node and create the mapping
    q_local_to_global_map.clear();
    q_local_to_global_map.reserve(this->_num_patches);
    std::vector<int> global_q_to_local_map(this->_num_patches, -1);
    std::vector<char> seen_q(this->_num_patches, 0);
    for(auto& g_node: assigned_g_node) {
        int q = this->_g_node_to_patch[g_node];
        if (seen_q[q] != 0) continue;
        seen_q[q] = 1;
        q_local_to_global_map.push_back(q);
    }
    std::sort(q_local_to_global_map.begin(), q_local_to_global_map.end());
    for (int q = 0; q < q_local_to_global_map.size(); q++) {
        global_q_to_local_map[q_local_to_global_map[q]] = q;
    }
    //Create the local Q
    local_Q_n = q_local_to_global_map.size();
    local_Qp.resize(local_Q_n + 1, 0);
    local_Qi.reserve(this->_quotient_graph._Qp[this->_quotient_graph._Q_n]);
    local_Q_node_weights.resize(q_local_to_global_map.size(), 0);
    int cnt = 0;
    for(size_t local_q = 0; local_q < q_local_to_global_map.size(); local_q++) {
        int global_q = q_local_to_global_map[local_q];
        local_Q_node_weights[local_q] = this->_quotient_graph._Q_node_weights[global_q];
        for(int nbr_ptr = this->_quotient_graph._Qp[global_q]; nbr_ptr < this->_quotient_graph._Qp[global_q + 1]; nbr_ptr++) {
            int nbr_q_global = this->_quotient_graph._Qi[nbr_ptr];
            int nbr_q_local = global_q_to_local_map[nbr_q_global];
            if(nbr_q_local == -1) continue;
            local_Qi.push_back(nbr_q_local);
            cnt++;
        }
        local_Qp[local_q + 1] = cnt;
    }
    assert(local_Qp.back() == local_Qi.size());
}

void CPUOrdering_PATCH::compute_bipartition(
    int Q_n,
    int* Qp,
    int* Qi,
    std::vector<int>&         Q_node_weights,
    std::vector<int>&         Q_partition_map)
{
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_PTYPE] = METIS_PTYPE_KWAY;
    options[METIS_OPTION_OBJTYPE] =
        METIS_OBJTYPE_VOL;  // Total communication volume minimization.
    options[METIS_OPTION_NUMBERING] = 0;
    options[METIS_OPTION_CONTIG]    = 0;
    options[METIS_OPTION_COMPRESS]  = 0;
    options[METIS_OPTION_DBGLVL]    = 0;

    idx_t   nvtxs  = Q_n;
    idx_t   ncon   = 1;
    idx_t*  vwgt   = NULL;
    idx_t*  vsize  = NULL;
    idx_t   nparts = 2;
    real_t* tpwgts = NULL;
    real_t* ubvec  = NULL;
    idx_t   objval = 0;

    Q_partition_map.resize(Q_n, 0);

    int metis_status = METIS_PartGraphKway(&nvtxs,
                                           &ncon,
                                           Qp,
                                           Qi,
                                           Q_node_weights.data(),
                                           vsize,
                                           nullptr,//Q.valuePtr(),//Edge weights are not used for Q
                                           &nparts,
                                           tpwgts,
                                           ubvec,
                                           options,
                                           &objval,
                                           Q_partition_map.data());

    if (metis_status != METIS_OK) {
        spdlog::error("MetisPatcher: METIS_PartGraphKway failed (status={})", metis_status);
        throw std::runtime_error("METIS_PartGraphKway failed");
    }    
}


void CPUOrdering_PATCH::two_way_Q_partition(
    int tree_node_idx,///<[in] The index of the current decomposition node
    std::vector<int>& assigned_g_nodes,///<[in] Assigned G nodes for current decomposition
    std::vector<int>& where///<[out] a vector with the size of assigned_g_node
){
    //Compute local quotient graph
    std::vector<int> two_way_q_partition_map;
    if (tree_node_idx != 0) {
        int local_Q_n;
        std::vector<int> local_Qp;
        std::vector<int> local_Qi;
        std::vector<int> local_Q_node_to_global_Q_node;
        std::vector<int> local_Q_node_weights;
        compute_local_quotient_graph(assigned_g_nodes,
            local_Q_n,
            local_Qp,
            local_Qi,
            local_Q_node_weights,
            local_Q_node_to_global_Q_node);

        compute_bipartition(local_Q_n, local_Qp.data(), local_Qi.data(), local_Q_node_weights, two_way_q_partition_map);

        //Converting the q map to node map
        where.resize(assigned_g_nodes.size());
        std::vector<int> global_q_node_to_local(_quotient_graph._Q_n, -1);
        for (int i = 0; i < local_Q_node_to_global_Q_node.size(); i++) {
            global_q_node_to_local[local_Q_node_to_global_Q_node[i]] = i;
        }

        for(size_t i = 0; i < assigned_g_nodes.size(); i++) {
            int g_node = assigned_g_nodes[i];
            int q_node = _g_node_to_patch[g_node];
            int q_local = global_q_node_to_local[q_node];
            where[i] = two_way_q_partition_map[q_local];
            assert(where[i] == 0 || where[i] == 1);
        }
    } else {
        compute_bipartition(_quotient_graph._Q_n, _quotient_graph._Qp.data(), _quotient_graph._Qi.data(), _quotient_graph._Q_node_weights, two_way_q_partition_map);
        where.resize(assigned_g_nodes.size());
        for (int i = 0; i <assigned_g_nodes.size(); i++) {
            int g_node = assigned_g_nodes[i];
            int q_node = _g_node_to_patch[g_node];
            where[i] = two_way_q_partition_map[q_node];
        }
    }

}

double CPUOrdering_PATCH::compute_separator_ratio()
{
    return this->_separator_ratio;
}


void CPUOrdering_PATCH::find_separator_superset(
    std::vector<int>& assigned_g_nodes,///<[in] Assigned G nodes for current decomposition
    std::vector<int>& where,///<[in] local node partition assignment
    int marker_token,///<[in] token marking nodes in this decomposition node
    std::vector<int>& separator_superset///<[out] The superset of separator nodes
)
{

    separator_superset.clear();
    for(size_t local_node = 0; local_node < assigned_g_nodes.size(); ++local_node) {
        int g_node = assigned_g_nodes[local_node];
        assert(g_node < this->_G_n);
        int partition_id = where[local_node];
        assert(partition_id != -1);
        for (int nbr_ptr = this->_Gp[g_node]; nbr_ptr < this->_Gp[g_node + 1]; ++nbr_ptr) {
            int nbr_id = this->_Gi[nbr_ptr];
            if (this->_decompose_marker[nbr_id] != marker_token) continue;
            int nbr_local_id = this->_decompose_local_index[nbr_id];
            int nbr_partition_id = where[nbr_local_id];
            if(partition_id == nbr_partition_id) continue;
            separator_superset.push_back(g_node);
            break;
        }
    }
    //Erase repetitive nodes
    // std::sort(separator_superset.begin(), separator_superset.end());
    #ifndef NDEBUG
    int prev_size = separator_superset.size();
    separator_superset.erase(std::unique(separator_superset.begin(), separator_superset.end()), separator_superset.end());
    int after_size = separator_superset.size();
    assert(prev_size == after_size);
    #endif
}

void CPUOrdering_PATCH::refine_separator_metis(
    int tree_node_idx,
    std::vector<int>& assigned_g_nodes,
    std::vector<int>& separator_g_nodes,
    std::vector<int>& where,
    int marker_token)
{
    #ifndef NDEBUG
    spdlog::info("METIS refinement: separator size before refinement: {}", separator_g_nodes.size());
    #endif
    if (separator_g_nodes.empty()) return;
    
    int local_nvtxs = assigned_g_nodes.size();
    if (local_nvtxs == 0) return;

    // 1. Mark separator superset nodes
    std::vector<char> is_sepsuper(local_nvtxs, 0);
    for(size_t i = 0; i < separator_g_nodes.size(); i++) {
        int g_node = separator_g_nodes[i];
        int local_node = this->_decompose_local_index[g_node];
        assert(this->_decompose_marker[g_node] == marker_token);
        is_sepsuper[local_node] = 1;
    }
    separator_g_nodes.clear();

    // 2. Build local CSR graph using reusable function
    SubGraph subgraph;
    build_subgraph_csr(assigned_g_nodes, subgraph, marker_token);

    // 3. Initialize where array from bipartition + separator
    // where: 0 = left partition, 1 = right partition, 2 = separator
    where.resize(local_nvtxs, 0);
    for (int i = 0; i < local_nvtxs; i++) {
        if (is_sepsuper[i] == 1) {
            where[i] = 2;  // Separator
        }
    }

    // 4. All vertices can move freely during refinement
    std::vector<idx_t> hmarker(local_nvtxs, -1);
    real_t ubfactor = 1.03;  // 3% imbalance tolerance

    int status = METIS_NodeRefine(local_nvtxs, subgraph._Gp.data(), nullptr,
                                  subgraph._Gi.data(), where.data(),
                                  hmarker.data(), ubfactor);
    if (status != METIS_OK) {
        spdlog::error("MetisPatcher: METIS_NodeRefine failed (status={})", status);
        throw std::runtime_error("METIS_NodeRefine failed");
    }  

#ifndef NDEBUG
    int total = 0;
    for (int i = 0; i < where.size(); i++) {
        if (where[i] == 2) {
            total++;
        }
    }
    spdlog::info("METIS refinement: separator size after refinement: {}", total);
#endif
}

void CPUOrdering_PATCH::refine_separator_min_vertex_cover(
    int tree_node_idx,
    std::vector<int>& assigned_g_nodes,
    std::vector<int>& separator_g_nodes,
    std::vector<int>& where,
    int marker_token)
{
    (void)tree_node_idx;
    (void)assigned_g_nodes;

    if (separator_g_nodes.empty()) return;

    static thread_local std::vector<int> boundary_token;
    static thread_local std::vector<int> boundary_local_index;
    static thread_local int current_token = 0;

    if (static_cast<int>(boundary_token.size()) < this->_G_n) {
        boundary_token.assign(this->_G_n, 0);
        boundary_local_index.assign(this->_G_n, -1);
        current_token = 0;
    }
    if (current_token == std::numeric_limits<int>::max()) {
        std::fill(boundary_token.begin(), boundary_token.end(), 0);
        current_token = 0;
    }
    ++current_token;

    const std::vector<int> boundary_global_nodes = separator_g_nodes;
    for (int i = 0; i < static_cast<int>(boundary_global_nodes.size()); ++i) {
        const int g_node = boundary_global_nodes[i];
        boundary_token[g_node] = current_token;
        boundary_local_index[g_node] = i;
    }

    std::vector<int> left_boundary_nodes;
    std::vector<int> right_boundary_nodes;
    std::vector<int> right_index_by_boundary(boundary_global_nodes.size(), -1);
    left_boundary_nodes.reserve(boundary_global_nodes.size());
    right_boundary_nodes.reserve(boundary_global_nodes.size());

    for (int boundary_id = 0;
         boundary_id < static_cast<int>(boundary_global_nodes.size());
         ++boundary_id) {
        const int g_node = boundary_global_nodes[boundary_id];
        assert(this->_decompose_marker[g_node] == marker_token);
        const int local_node = this->_decompose_local_index[g_node];
        if (where[local_node] == 0) {
            left_boundary_nodes.push_back(boundary_id);
        } else {
            assert(where[local_node] == 1);
            right_index_by_boundary[boundary_id] =
                static_cast<int>(right_boundary_nodes.size());
            right_boundary_nodes.push_back(boundary_id);
        }
    }

    if (left_boundary_nodes.empty() || right_boundary_nodes.empty()) {
        for (int separator_g_node : separator_g_nodes) {
            where[this->_decompose_local_index[separator_g_node]] = 2;
        }
        return;
    }

    std::vector<std::vector<int>> left_adj(left_boundary_nodes.size());
    for (int left_idx = 0;
         left_idx < static_cast<int>(left_boundary_nodes.size());
         ++left_idx) {
        const int left_boundary_id = left_boundary_nodes[left_idx];
        const int g_node = boundary_global_nodes[left_boundary_id];
        for (int nbr_ptr = this->_Gp[g_node]; nbr_ptr < this->_Gp[g_node + 1];
             ++nbr_ptr) {
            const int nbr = this->_Gi[nbr_ptr];
            if (boundary_token[nbr] != current_token) continue;
            const int nbr_boundary_id = boundary_local_index[nbr];
            const int right_idx = right_index_by_boundary[nbr_boundary_id];
            if (right_idx == -1) continue;
            left_adj[left_idx].push_back(right_idx);
        }
    }

    const int n_left = static_cast<int>(left_boundary_nodes.size());
    const int n_right = static_cast<int>(right_boundary_nodes.size());
    std::vector<int> pair_left(n_left, -1);
    std::vector<int> pair_right(n_right, -1);
    std::vector<int> dist(n_left, 0);

    auto bfs = [&]() -> bool {
        std::vector<int> queue;
        queue.reserve(n_left);
        bool found_free_right = false;
        for (int u = 0; u < n_left; ++u) {
            if (pair_left[u] == -1) {
                dist[u] = 0;
                queue.push_back(u);
            } else {
                dist[u] = -1;
            }
        }

        size_t head = 0;
        while (head < queue.size()) {
            const int u = queue[head++];
            for (int v : left_adj[u]) {
                const int matched_left = pair_right[v];
                if (matched_left == -1) {
                    found_free_right = true;
                } else if (dist[matched_left] == -1) {
                    dist[matched_left] = dist[u] + 1;
                    queue.push_back(matched_left);
                }
            }
        }
        return found_free_right;
    };

    auto dfs = [&](auto&& self, int u) -> bool {
        for (int v : left_adj[u]) {
            const int matched_left = pair_right[v];
            if (matched_left == -1 ||
                (dist[matched_left] == dist[u] + 1 &&
                 self(self, matched_left))) {
                pair_left[u] = v;
                pair_right[v] = u;
                return true;
            }
        }
        dist[u] = -1;
        return false;
    };

    while (bfs()) {
        for (int u = 0; u < n_left; ++u) {
            if (pair_left[u] == -1) {
                dfs(dfs, u);
            }
        }
    }

    std::vector<char> visited_left(n_left, 0);
    std::vector<char> visited_right(n_right, 0);
    std::vector<int> queue;
    queue.reserve(n_left);
    for (int u = 0; u < n_left; ++u) {
        if (pair_left[u] == -1) {
            visited_left[u] = 1;
            queue.push_back(u);
        }
    }

    size_t head = 0;
    while (head < queue.size()) {
        const int u = queue[head++];
        for (int v : left_adj[u]) {
            if (pair_left[u] == v || visited_right[v]) continue;
            visited_right[v] = 1;
            const int matched_left = pair_right[v];
            if (matched_left != -1 && !visited_left[matched_left]) {
                visited_left[matched_left] = 1;
                queue.push_back(matched_left);
            }
        }
    }

    std::vector<int> refined_separator;
    refined_separator.reserve(separator_g_nodes.size());
    for (int u = 0; u < n_left; ++u) {
        if (!visited_left[u]) {
            refined_separator.push_back(
                boundary_global_nodes[left_boundary_nodes[u]]);
        }
    }
    for (int v = 0; v < n_right; ++v) {
        if (visited_right[v]) {
            refined_separator.push_back(
                boundary_global_nodes[right_boundary_nodes[v]]);
        }
    }

    if (refined_separator.empty()) {
        for (int separator_g_node : separator_g_nodes) {
            where[this->_decompose_local_index[separator_g_node]] = 2;
        }
        return;
    }

    separator_g_nodes.swap(refined_separator);
    for (int separator_g_node : separator_g_nodes) {
        where[this->_decompose_local_index[separator_g_node]] = 2;
    }
}


void CPUOrdering_PATCH::compute_vertex_separator(
    int tree_node_idx,
    std::vector<int>& assigned_g_nodes,
    std::vector<int>& where)
{
    int marker_token = tree_node_idx + 1;
    for (int i = 0; i < static_cast<int>(assigned_g_nodes.size()); i++) {
        int g_node = assigned_g_nodes[i];
        this->_decompose_marker[g_node] = marker_token;
        this->_decompose_local_index[g_node] = i;
    }

    std::vector<int> separator_g_nodes;
    find_separator_superset(assigned_g_nodes, where, marker_token, separator_g_nodes);

    #ifdef DEBUG
    auto check_valid_separator = [&](std::vector<int>& map) -> bool {
        std::vector<char> check_separator(this->_G_n, 0);
        std::vector<int> l_to_g(assigned_g_nodes.size(), -1);
        std::vector<int> g_to_l(_G_n, -1);
        for (int i = 0; i < assigned_g_nodes.size(); i++) {
            l_to_g[i] = assigned_g_nodes[i];
            g_to_l[assigned_g_nodes[i]] = i;
        }
    
        for(int i = 0; i < assigned_g_nodes.size(); i++) {
            int map_id = map[i];
            int g_node = assigned_g_nodes[i];
            if (map_id == 2) continue;
            for(int j = this->_Gp[g_node]; j < this->_Gp[g_node + 1]; j++) {
                int nbr = this->_Gi[j];
                if (g_to_l[nbr] == -1) continue;
                int nbr_map_id = map[g_to_l[nbr]];
                if (nbr_map_id == 2) continue;
                //If the nbr is in the same part, continue
                if(map_id == nbr_map_id) continue;
                //If the nbr is not in the separator, return false
                if(check_separator[nbr] == 0)
                    return false;
            }
        }
        return true;
    };
    std::vector<int> map(assigned_g_nodes.size(), -1);
    std::vector<int> check(_G_n, 0);
    for (auto& sep: separator_g_nodes) {
        check[sep] = 1;
    }
    for(int i = 0; i < assigned_g_nodes.size(); i++) {
        int g_node = assigned_g_nodes[i];
        int map_id = where[i];
        if (check[g_node] == 1) {
            map[i] = 2;
        } else if(map_id == tree_node_idx * 2 + 1) {
            map[i] = 0;
        } else if(map_id == tree_node_idx * 2 + 2) {
            map[i] = 1;
        }
    }
    bool is_valid = check_valid_separator(map);
    assert(is_valid && "The separator is not valid before refinement");
    #endif

    if (separator_g_nodes.empty()) {
        return;
    }

    const int tree_level = tree_level_from_id(tree_node_idx);
    const bool should_refine_separator =
        always_refine_separators ||
        tree_level <= separator_refinement_max_level;
    if (!should_refine_separator) {
        for (int separator_g_node : separator_g_nodes) {
            int local_node = this->_decompose_local_index[separator_g_node];
            assert(this->_decompose_marker[separator_g_node] == marker_token);
            where[local_node] = 2;
        }
        return;
    }


    // Refine the separator superset down to a minimal separator. Default is the
    // bipartite minimum-vertex-cover refiner; refine_separator_metis is the
    // opt-in METIS FM alternative.
    if (use_min_vertex_cover_refinement) {
        refine_separator_min_vertex_cover(
            tree_node_idx, assigned_g_nodes, separator_g_nodes, where, marker_token);
    } else {
        refine_separator_metis(
            tree_node_idx, assigned_g_nodes, separator_g_nodes, where, marker_token);
    }
    #ifdef DEBUG
    is_valid = check_valid_separator(where);
    assert(is_valid && "The separator is not valid");
    #endif
}


void CPUOrdering_PATCH::level_order_offset_computation()
{
    int offset = 0;
    for(size_t i =  _decomposition_tree.get_number_of_decomposition_nodes() - 1; i >= 0; i--){
        auto& decomposition_node = this->_decomposition_tree.decomposition_nodes[i];
        decomposition_node.offset = offset;
        offset += decomposition_node.assigned_g_nodes.size();
    }
}

int CPUOrdering_PATCH::post_order_offset_computation(int offset,
                                                  int decomposition_node_id)
{
    assert(decomposition_node_id <
           this->_decomposition_tree.get_number_of_decomposition_nodes());
    auto& decomposition_node =
        this->_decomposition_tree.decomposition_nodes[decomposition_node_id];
    // Compute the offset for the left and right children
    int left_node    = decomposition_node.left_node_idx;
    int right_node   = decomposition_node.right_node_idx;
    int right_offset = offset;
    if (left_node != -1) {
        right_offset = post_order_offset_computation(offset, left_node);
    }
    int separator_offset = right_offset;
    if (right_node != -1) {
        separator_offset =
            post_order_offset_computation(right_offset, right_node);
    }
    decomposition_node.offset = separator_offset;
    offset = separator_offset + decomposition_node.assigned_g_nodes.size();
    return offset;
}

bool CPUOrdering_PATCH::use_serial_ordering_path() const
{
    return this->_G_n > 0 && this->_G_n <= this->serial_ordering_max_nodes;
}

void CPUOrdering_PATCH::decompose_node(
    int tree_node_id,
    int level,
    std::vector<DecompositionInfo>& decomposition_info_stack)
{
    // Get the input information for the current node
#ifndef NDEBUG
    spdlog::info("*********** Working on node {} ***********", tree_node_id);
#endif
    int tree_node_parent_id =
        decomposition_info_stack[tree_node_id].decomposition_node_parent_id;
    std::vector<int>& assigned_g_nodes =
        decomposition_info_stack[tree_node_id].assigned_g_nodes;
    auto& cur_decomposition_node =
        this->_decomposition_tree.decomposition_nodes[tree_node_id];

    //++++++ if it is not decomposable, skip it ******
    if(assigned_g_nodes.empty()) return;

    #pragma omp atomic
    this->decompose_processed_nodes++;


    if (tree_node_id == -1 ||
        tree_node_parent_id == -1) {
        if (tree_node_id != 0) {
            spdlog::error("The info is not initialized correctly.");
            spdlog::error("The level is {}", level);
            spdlog::error("The node_id {}, and parent_id {}",
                          tree_node_id,
                          tree_node_parent_id);
        }
    }


    //+++++++++++++ If it is a leaf node ++++++++++++++++
    if (level == this->_decomposition_tree.decomposition_level) {
        // Add all the nodes in the patches to the dofs
        cur_decomposition_node.init_node(
            -1,
            -1,
            tree_node_id,
            tree_node_parent_id,
            level,
            assigned_g_nodes);
        //Assign nodes to tree array
        for (const auto& node: assigned_g_nodes) {
            this->_decomposition_tree.g_node_to_tree_node[node] = tree_node_id;
        }
        return;
    }

    //+++++++++++++ If it is not a leaf node ++++++++++++++++
    // Overall flow:
    // Step 1: Compute two equal size partitions from the assigned dofs
    // Step 2: Find the separator nodes of the two partitions
    // Step 3: Initialize the input of the left and right

    // Step 1: Compute two equal size partitions from the assigned dofs
    // auto two_way_start = std::chrono::high_resolution_clock::now();
    std::vector<int> separator_g_nodes;
    std::vector<int> where;

    const bool use_direct_separator =
        !use_patch_separator ||
        (direct_separator_max_level >= 0 &&
         level <= direct_separator_max_level &&
         static_cast<int>(assigned_g_nodes.size()) >=
             direct_separator_min_nodes);

    if(!use_direct_separator) {
        auto partition_start = Clock::now();
        this->two_way_Q_partition(tree_node_id,
            assigned_g_nodes,
            where);
        double partition_ms =
            elapsed_ms(partition_start, Clock::now());
        #pragma omp atomic
        this->decompose_partition_time_ms += partition_ms;

        // Step 2: Find the separator nodes of the two partitions
        auto separator_start = Clock::now();
        this->compute_vertex_separator(tree_node_id, assigned_g_nodes, where);
        double separator_ms =
            elapsed_ms(separator_start, Clock::now());
        #pragma omp atomic
        this->decompose_separator_time_ms += separator_ms;
        assert(where.size() == assigned_g_nodes.size());
    } else {
        auto separator_start = Clock::now();
        // Build local subgraph in CSR format
        SubGraph subgraph;
        build_subgraph_csr(assigned_g_nodes, subgraph);

        // Convert to METIS idx_t type
        idx_t nVertices = subgraph._num_nodes;
        idx_t sepsize;
        where.resize(nVertices);
        int ret = METIS_ComputeVertexSeparator(&nVertices, subgraph._Gp.data(), subgraph._Gi.data(),
                                               nullptr, nullptr, &sepsize,
                                               where.data());
        if (ret != METIS_OK) {
            spdlog::error("MetisPatcher: METIS_ComputeVertexSeparator failed (status={})", ret);
            throw std::runtime_error("METIS_ComputeVertexSeparator failed");
        }
        double separator_ms =
            elapsed_ms(separator_start, Clock::now());
        #pragma omp atomic
        this->decompose_separator_time_ms += separator_ms;
    }



    auto bookkeeping_start = Clock::now();
    for (int i = 0; i < where.size(); i++) {
        if (where[i] == 2) {
            separator_g_nodes.emplace_back(assigned_g_nodes[i]);
        }
    }
    this->_decomposition_tree.assign_nodes_to_tree(separator_g_nodes, tree_node_id);
    long long separator_count =
        static_cast<long long>(separator_g_nodes.size());
    #pragma omp atomic
    this->decompose_separator_vertices += separator_count;

    // spdlog::info("two_way_Q_partition time: {} ms", two_way_duration);
    // spdlog::info("compute_vertex_separator time: {} ms", three_way_duration);
    std::vector<int> left_assigned_g_nodes, right_assigned_g_nodes;

    //Assign the nodes to the left and right assigned dofs
    left_assigned_g_nodes.reserve(assigned_g_nodes.size());
    right_assigned_g_nodes.reserve(assigned_g_nodes.size());
    for(size_t i = 0; i < assigned_g_nodes.size(); i++) {
        if (where[i] == 2) continue;
        if (where[i] == 0){
            left_assigned_g_nodes.push_back(assigned_g_nodes[i]);
        } else if (where[i] == 1){
            right_assigned_g_nodes.push_back(assigned_g_nodes[i]);
        } else {
            spdlog::info("The where is invalid");
            assert(false);
        }
    }
#ifndef NDEBUG
    spdlog::info("The left size is: {}, the right size is: {} and the separator size is {}: ",
        left_assigned_g_nodes.size(),
        right_assigned_g_nodes.size(),
        separator_g_nodes.size());
#endif

    // Initialize the input of the left and right children for the
    // next wavefront
    int left_node_idx  = tree_node_id * 2 + 1;
    int right_node_idx = tree_node_id * 2 + 2;
    if (left_assigned_g_nodes.empty()) {
        left_node_idx = -1;
    }
    if (right_assigned_g_nodes.empty()) {
        right_node_idx = -1;
    }


    //Update the quotient graph by removing the effect of separator nodes
    for(size_t i = 0; i < separator_g_nodes.size(); i++) {
        int g_node = separator_g_nodes[i];
        int q_node = this->_g_node_to_patch[g_node];
        this->_quotient_graph._Q_node_weights[q_node]--;
        //TODO: Remove the edge weights next if the performance degrades
    }

    cur_decomposition_node.init_node(left_node_idx,
                                     right_node_idx,
                                     tree_node_id,
                                     tree_node_parent_id,
                                     level,
                                     separator_g_nodes);

    if (left_node_idx != -1) {
        assert(left_node_idx >= 0 &&
               left_node_idx < decomposition_info_stack.size());
        assert(!left_assigned_g_nodes.empty() &&
               "Left child should have assigned patches");
        decomposition_info_stack[left_node_idx]
            .decomposition_node_parent_id = tree_node_id;
        decomposition_info_stack[left_node_idx].assigned_g_nodes =
            left_assigned_g_nodes;
    }

    // Initialize the right child
    if (right_node_idx != -1) {
        assert(right_node_idx >= 0 &&
               right_node_idx < decomposition_info_stack.size());
        assert(!right_assigned_g_nodes.empty() &&
               "Right child should have assigned patches");
        decomposition_info_stack[right_node_idx]
            .decomposition_node_parent_id = tree_node_id;
        decomposition_info_stack[right_node_idx].assigned_g_nodes =
            right_assigned_g_nodes;
    }

    double bookkeeping_ms =
        elapsed_ms(bookkeeping_start, Clock::now());
    #pragma omp atomic
    this->decompose_bookkeeping_time_ms += bookkeeping_ms;
}

void CPUOrdering_PATCH::decompose()
{
    // Initialize the first call for the decomposition info stack
    std::vector<DecompositionInfo> decomposition_info_stack(
        this->_decomposition_tree.get_number_of_decomposition_nodes());
    decomposition_info_stack[0].decomposition_node_parent_id = -1;
    decomposition_info_stack[0].assigned_g_nodes.resize(_G_n);
    for(int i = 0; i < _G_n; i++) {
        decomposition_info_stack[0].assigned_g_nodes[i] = i;
    }

    if (use_serial_ordering_path()) {
        for (int l = 0; l < this->_decomposition_tree.decomposition_level + 1; l++) {
            int start_level_idx = (1 << l) - 1;
            int end_level_idx   = (1 << (l + 1)) - 1;
            assert(end_level_idx <= this->_decomposition_tree.get_number_of_decomposition_nodes());
            for (int tree_node_id = start_level_idx; tree_node_id < end_level_idx;
                 ++tree_node_id) {
                decompose_node(tree_node_id, l, decomposition_info_stack);
            }
        }
        return;
    }

    // auto omp_parallel_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel
    {
        for (int l = 0; l < this->_decomposition_tree.decomposition_level + 1; l++) {
            int start_level_idx = (1 << l) - 1;
            int end_level_idx   = (1 << (l + 1)) - 1;
            assert(end_level_idx <= this->_decomposition_tree.get_number_of_decomposition_nodes());
            #pragma omp for schedule(static)
            for (int tree_node_id = start_level_idx; tree_node_id < end_level_idx;
                 ++tree_node_id) {
                decompose_node(tree_node_id, l, decomposition_info_stack);
            }
        }
    }
    // auto omp_parallel_end = std::chrono::high_resolution_clock::now();
    // auto omp_parallel_duration = std::chrono::duration_cast<std::chrono::milliseconds>(omp_parallel_end - omp_parallel_start).count();
    // spdlog::info("OpenMP parallel region wall clock time: {} ms", omp_parallel_duration);
}


void CPUOrdering_PATCH::init_patches(int num_patches,std::vector<int> & g_node_to_patch, int num_levels)
{
    // Init the hirerchical tree memory
    assert(g_node_to_patch.size() == static_cast<size_t>(this->_G_n));
    assert(this->_G_n > 0);
    assert(num_patches > 0);
    this->_num_patches = num_patches;
    this->_g_node_to_patch = g_node_to_patch;
    this->_decompose_marker.assign(this->_G_n, 0);
    this->_decompose_local_index.assign(this->_G_n, -1);

    int reasonable_num_levels = static_cast<int>(std::log2(num_patches));
    if (direct_separator_max_level >= reasonable_num_levels) {
        int ceil_num_levels = 0;
        while ((1 << ceil_num_levels) < num_patches) {
            ++ceil_num_levels;
        }
        reasonable_num_levels = ceil_num_levels;
    }
    int final_num_levels = std::min(num_levels, reasonable_num_levels);
    spdlog::info("The final number of levels is: {}", final_num_levels);
    int total_number_of_decomposition_nodes = (1 << (final_num_levels + 1)) - 1;
    this->_decomposition_tree.init_decomposition_tree(
        total_number_of_decomposition_nodes,
        final_num_levels, this->_G_n, this->_num_patches);
}


void CPUOrdering_PATCH::build_subgraph_csr(
    const std::vector<int>& assigned_g_nodes,
    SubGraph& subgraph,
    int marker_token) const
{
    int local_nvtxs = assigned_g_nodes.size();
    subgraph._num_nodes = local_nvtxs;
    
    if (local_nvtxs == 0) {
        subgraph._Gp.clear();
        subgraph._Gi.clear();
        return;
    }

    std::vector<int> fallback_global_to_local;
    const bool use_marker = marker_token != 0;
    if (!use_marker) {
        // 1. Build global-to-local mapping
        fallback_global_to_local.assign(this->_G_n, -1);
        for (int i = 0; i < local_nvtxs; i++) {
            int g_node = assigned_g_nodes[i];
            fallback_global_to_local[g_node] = i;
        }
    }

    auto local_index = [&](int g_node) -> int {
        if (use_marker) {
            return this->_decompose_marker[g_node] == marker_token
                       ? this->_decompose_local_index[g_node]
                       : -1;
        }
        return fallback_global_to_local[g_node];
    };

    // 2. Build local CSR graph - count degrees first
    subgraph._Gp.assign(local_nvtxs + 1, 0);
    for (int i = 0; i < local_nvtxs; i++) {
        int g_node = assigned_g_nodes[i];
        for (int nbr_ptr = this->_Gp[g_node]; nbr_ptr < this->_Gp[g_node + 1]; nbr_ptr++) {
            int nbr_id = this->_Gi[nbr_ptr];
            if (local_index(nbr_id) != -1) {
                subgraph._Gp[i + 1]++;
            }
        }
    }

    // Prefix sum to get row pointers
    for (int i = 0; i < local_nvtxs; i++) {
        subgraph._Gp[i + 1] += subgraph._Gp[i];
    }

    // Allocate and fill adjacency array
    int total_edges = subgraph._Gp[local_nvtxs];
    subgraph._Gi.resize(total_edges);
    std::vector<int> write_pos(subgraph._Gp.begin(), subgraph._Gp.end() - 1);

    for (int i = 0; i < local_nvtxs; i++) {
        int g_node = assigned_g_nodes[i];
        for (int nbr_ptr = this->_Gp[g_node]; nbr_ptr < this->_Gp[g_node + 1]; nbr_ptr++) {
            int nbr_id = this->_Gi[nbr_ptr];
            int local_nbr = local_index(nbr_id);
            if (local_nbr != -1) {
                subgraph._Gi[write_pos[i]++] = local_nbr;
            }
        }
    }
}


void CPUOrdering_PATCH::step1_compute_quotient_graph()
{
    // Given node to patch, first give each separator node a unique patch ID
    // Step 1: assign patch-id -1 to each boundary vertex
    // Count the number of patch ids

    // auto patch_fix_start = std::chrono::high_resolution_clock::now();
    auto quotient_compaction_start = Clock::now();
    std::vector<int> patch_offset(_num_patches, 1);
    // auto patch_fix_end = std::chrono::high_resolution_clock::now();
    // auto patch_time = std::chrono::duration_cast<std::chrono::milliseconds>(patch_fix_end - patch_fix_start).count();
    // spdlog::info("Patch fix time: {} ms for vector of size {}", patch_time, _num_patches);
    for (size_t i = 0; i < _g_node_to_patch.size(); i++) {
        int patch_id = _g_node_to_patch[i];
        patch_offset[patch_id] = 0;
    }


    //prefix scan
    for (int i = 1; i < _num_patches; ++i) {
        patch_offset[i] += patch_offset[i - 1];
    }

    for (size_t i = 0; i < _g_node_to_patch.size(); i++) {
        int patch_id = _g_node_to_patch[i];
        _g_node_to_patch[i] -= patch_offset[patch_id];
    }
    this->_num_patches -= patch_offset.back();
    this->quotient_patch_compaction_time_ms +=
        elapsed_ms(quotient_compaction_start, Clock::now());

    assert(this->_g_node_to_patch.size() == this->_G_n);

    
    //Create the local quotient graph
    // auto quotient_start = std::chrono::high_resolution_clock::now();
    
    // 1. Node weights (patch sizes) and mapping nodes -> patches
    auto quotient_patch_nodes_start = Clock::now();
    _quotient_graph._Q_node_weights.assign(_num_patches, 0);

    // Optional but very useful: nodes of each patch
    std::vector<std::vector<int>> patch_nodes(_num_patches);
    patch_nodes.reserve(_num_patches);

    for (int g = 0; g < _G_n; ++g) {
        int p = this->_g_node_to_patch[g];
        assert(p >= 0 && p < _num_patches);
        _quotient_graph._Q_node_weights[p]++;
        patch_nodes[p].push_back(g);
    }
    this->quotient_patch_nodes_time_ms +=
        elapsed_ms(quotient_patch_nodes_start, Clock::now());

    // 2. Build adjacency lists of quotient graph (per patch)
    auto quotient_adjacency_start = Clock::now();
    std::vector<std::vector<int>> Q_adj(_num_patches);
    std::vector<int> mark(_num_patches, -1);  // mark[q] == p means q already added as neighbor of p

    for (int p = 0; p < _num_patches; ++p) {
        auto &nodes = patch_nodes[p];
        if (nodes.empty()) continue;

        for (int g : nodes) {
            for (int nbr_ptr = this->_Gp[g]; nbr_ptr < this->_Gp[g + 1]; ++nbr_ptr) {
                int nbr_id = this->_Gi[nbr_ptr];
                int q = this->_g_node_to_patch[nbr_id];
                if (q == p) continue;  // stay inside patch -> no quotient edge

                // Deduplicate neighbors for this p using "mark"
                if (mark[q] == p) continue;
                mark[q] = p;
                Q_adj[p].push_back(q);
            }
        }
    }
    this->quotient_adjacency_time_ms +=
        elapsed_ms(quotient_adjacency_start, Clock::now());

    auto quotient_csr_start = Clock::now();
    int q_nnz = 0;
    for (auto& nbrs: Q_adj) {
        std::sort(nbrs.begin(), nbrs.end());
        q_nnz += nbrs.size();
    }
    _quotient_graph._Q_n = this->_num_patches;
    _quotient_graph._Qp.resize(this->_num_patches + 1, 0);
    _quotient_graph._Qi.resize(q_nnz, 0);
    int cnt = 0;
    for(int q_id = 0; q_id < this->_num_patches; q_id++) {
        auto& nbrs = Q_adj[q_id];
        for (const auto&  nbr: nbrs) {
            _quotient_graph._Qi[cnt] = nbr;
            cnt++;
        }
        _quotient_graph._Qp[q_id + 1] = cnt;
    }
    assert(cnt == q_nnz);
    this->quotient_csr_time_ms +=
        elapsed_ms(quotient_csr_start, Clock::now());
    #ifdef DEBUG
    // Check if the quotient graph CSR matrix is symmetric
    bool is_symmetric = true;
    for (int i = 0; i < _quotient_graph._Q_n; ++i) {
        for (int idx = _quotient_graph._Qp[i]; idx < _quotient_graph._Qp[i + 1]; ++idx) {
            int j = _quotient_graph._Qi[idx];
            
            // Check if i exists in row j's neighbors
            bool found = false;
            for (int jdx = _quotient_graph._Qp[j]; jdx < _quotient_graph._Qp[j + 1]; ++jdx) {
                if (_quotient_graph._Qi[jdx] == i) {
                    found = true;
                    break;
                }
            }
            
            if (!found) {
                is_symmetric = false;
                spdlog::warn("CSR matrix is NOT symmetric: edge ({}, {}) exists but ({}, {}) does not", i, j, j, i);
                break;
            }
        }
        if (!is_symmetric) break;
    }
    assert(is_symmetric);
    #endif
}

void CPUOrdering_PATCH::step2_create_decomposition_tree()
{
    this->decompose();
#ifndef NDEBUG
    spdlog::info("Decomposition tree is created.");
#endif

#ifndef NDEBUG
    //Check to see if all the nodes exist
    spdlog::info("Checking decomposition validation.");
    std::vector<bool> is_visited(this->_G_n, false);
    for(auto& node : this->_decomposition_tree.decomposition_nodes) {
        if (node.assigned_g_nodes.empty())
            continue;
        for (auto& g_node : node.assigned_g_nodes) {
            assert(is_visited[g_node] == false);
            is_visited[g_node] = true;
        }
    }
    for (int i = 0; i < is_visited.size(); ++i) {
        assert(is_visited[i] == true);
    }
    spdlog::info("Decomposition contains all the nodes");

    //Check for whether separators are valid
#endif
}

void CPUOrdering_PATCH::compute_sub_graphs(std::vector<SubGraph>& sub_graphs){
    // This loop creates the global-to-local mapping as well as number of nodes per group
    std::vector<int> global_to_local(this->_G_n, -1);
    sub_graphs.clear();
    sub_graphs.resize(this->_decomposition_tree.decomposition_nodes.size());

    for (auto& node : this->_decomposition_tree.decomposition_nodes) {
        if (node.assigned_g_nodes.empty()) continue;

        // assign local ids 0..k-1 within this subgraph
        for (size_t i = 0; i < (int)node.assigned_g_nodes.size(); i++) {
            int g_node = node.assigned_g_nodes[i];
            assert(global_to_local[g_node] == -1);
            global_to_local[g_node] = i;
        }

        sub_graphs[node.node_id]._num_nodes = (int)node.assigned_g_nodes.size();
        sub_graphs[node.node_id]._Gp.assign(node.assigned_g_nodes.size() + 1, 0);
    }

    // Pass 1: count degree per local node within its group to build _Gp (row pointers)
    for (int g_node = 0; g_node < this->_G_n; g_node++) {
        int group_id = this->_decomposition_tree.g_node_to_tree_node[g_node];
        assert(group_id >= 0 && group_id < sub_graphs.size());
        int local_i = global_to_local[g_node];
        assert(local_i >= 0 && local_i < sub_graphs[group_id]._num_nodes);

        SubGraph& sg = sub_graphs[group_id];
        if (sg._num_nodes == 0) continue;

        for (int nbr_ptr = _Gp[g_node]; nbr_ptr < _Gp[g_node + 1]; nbr_ptr++) {
            int nbr_id       = _Gi[nbr_ptr];
            int nbr_group_id = this->_decomposition_tree.g_node_to_tree_node[nbr_id];
            if (nbr_group_id != group_id) continue;

            int local_j = global_to_local[nbr_id];
            assert(local_j >= 0);
            // Count one edge from local_i
            sg._Gp[local_i + 1]++;
        }
    }

    // Prefix-sum to convert degrees into CSR row pointers
    for (size_t gid = 0; gid < (int)sub_graphs.size(); gid++) {
        SubGraph& sg = sub_graphs[gid];
        if (sg._num_nodes == 0) continue;

        for (int r = 0; r < sg._num_nodes; r++) {
            sg._Gp[r + 1] += sg._Gp[r];
        }

        sg._Gi.resize(sg._Gp[sg._num_nodes], 0);
    }

    // Pass 2: fill _Gi using a set of write cursors, one per group
    std::vector<std::vector<int>> write_pos(sub_graphs.size());
    for (size_t gid = 0; gid < sub_graphs.size(); gid++) {
        if (sub_graphs[gid]._num_nodes == 0) continue;
        write_pos[gid] = sub_graphs[gid]._Gp; // start at row starts
    }

    for (int g_node = 0; g_node < this->_G_n; g_node++) {
        int group_id = this->_decomposition_tree.g_node_to_tree_node[g_node];
        assert(group_id >= 0 && group_id < sub_graphs.size());
        int local_i = global_to_local[g_node];

        SubGraph& sg = sub_graphs[group_id];
        if (sg._num_nodes == 0) continue;

        for (int nbr_ptr = _Gp[g_node]; nbr_ptr < _Gp[g_node + 1]; nbr_ptr++) {
            int nbr_id       = _Gi[nbr_ptr];
            int nbr_group_id = this->_decomposition_tree.g_node_to_tree_node[nbr_id];
            if (nbr_group_id != group_id) continue;

            int local_j = global_to_local[nbr_id];
            assert(local_j >= 0);

            int& pos = write_pos[group_id][local_i];
            sg._Gi[pos] = local_j;
            pos++;
        }
    }
}

void CPUOrdering_PATCH::step3_compute_local_permutations()
{
    std::vector<SubGraph> sub_graphs;

    auto sub_graph_start = Clock::now();
    this->compute_sub_graphs(sub_graphs);
    this->local_subgraph_time_ms += elapsed_ms(sub_graph_start, Clock::now());

    const bool use_serial_local_ordering = use_serial_ordering_path();
    auto compute_local_ordering = [&](int i) {
        std::vector<int> local_permutation;
        if (sub_graphs[i]._num_nodes == 0) return;
        auto local_order_start = Clock::now();
        local_permute(sub_graphs[i]._num_nodes, sub_graphs[i]._Gp.data(), sub_graphs[i]._Gi.data(), local_permutation);
        this->_decomposition_tree.decomposition_nodes[i].set_local_permutation(local_permutation);
        double local_order_ms = elapsed_ms(local_order_start, Clock::now());
        if (use_serial_local_ordering) {
            this->local_order_time_ms += local_order_ms;
            this->local_order_blocks++;
        } else {
            #pragma omp atomic
            this->local_order_time_ms += local_order_ms;
            #pragma omp atomic
            this->local_order_blocks++;
        }
    };

    //Compute the local permutations
    if (use_serial_local_ordering) {
        for (int i = 0; i < this->_decomposition_tree.decomposition_nodes.size(); i++) {
            compute_local_ordering(i);
        }
    } else {
        #pragma omp parallel for
        for (int i = 0; i < this->_decomposition_tree.decomposition_nodes.size(); i++) {
            compute_local_ordering(i);
        }
    }
}

    
void CPUOrdering_PATCH::step4_assemble_final_permutation(std::vector<int>& perm)
{
    // Apply the offset to the decomposition nodes
    if (etree_order == "post_order") {
        post_order_offset_computation(0, 0);
    } else if (etree_order == "level_order") {
        level_order_offset_computation();
    } else {
        spdlog::error("Unknown etree order. Use post_order or level_order");
    }

    perm.clear();
    perm.resize(this->_G_n, -1);
    for (auto& node : this->_decomposition_tree.decomposition_nodes) {
        if (node.assigned_g_nodes.empty())
            continue;
        for (size_t local_node = 0; local_node < node.assigned_g_nodes.size(); local_node++) {
            int global_node = node.assigned_g_nodes[local_node];
            int perm_index  = node.local_new_labels[local_node] + node.offset;
            assert(global_node >= 0 && global_node < this->_G_n &&
                   "Invalid global node index");
            assert(perm_index >= 0 && perm_index < perm.size() &&
                   "Permutation index out of bounds");
            assert(perm[perm_index] == -1 &&
                   "Permutation slot already filled - duplicate node!");
            perm[perm_index] = global_node;
        }
    }
}

void CPUOrdering_PATCH::compute_permutation(std::vector<int>& perm)
{
    reset_timing_stats();

    if (this->_Gp == nullptr || this->_Gi == nullptr || this->_G_n == 0 || this->_G_nnz == 0) {
        spdlog::error(
            "Graph not set. Please call setGraph() before "
            "compute_permutation().");
        return;
    }

    auto total_start_time = Clock::now();

    // Step 1: Compute node to patch map
    auto start_time = Clock::now();
    step1_compute_quotient_graph();
    auto end_time = Clock::now();
    quotient_graph_time = std::chrono::duration<double>(end_time - start_time).count();
    // spdlog::info("Step 1 (compute quotient graph) completed in {:.6f} seconds", quotient_graph_time);

    // Step 2: Create decomposition tree
    start_time = Clock::now();
    step2_create_decomposition_tree();
    end_time = Clock::now();
    decompose_time = std::chrono::duration<double>(end_time - start_time).count();
    // spdlog::info("Step 2 (decomposition tree) completed in {:.6f} seconds", decompose_time);

    // Step 3: Compute the local permutations
    start_time = Clock::now();
    step3_compute_local_permutations();
    end_time = Clock::now();
    local_permute_time = std::chrono::duration<double>(end_time - start_time).count();
    // spdlog::info("Step 3 (local permutations) completed in {:.6f} seconds", local_permute_time);

    // Step 4: Assemble the final permutation
    start_time = Clock::now();
    step4_assemble_final_permutation(perm);
    end_time = Clock::now();
    assemble_time = std::chrono::duration<double>(end_time - start_time).count();
    // spdlog::info("Step 4 (assemble permutation) completed in {:.6f} seconds", assemble_time);

    ordering_total_time_ms = elapsed_ms(total_start_time, Clock::now());
    const int quotient_edges = _quotient_graph._Qp.empty()
                                   ? 0
                                   : _quotient_graph._Qp.back();

    spdlog::info(
        "HOMA CPU ordering timing: total={:.3f} ms quotient={:.3f} ms "
        "decompose={:.3f} ms local={:.3f} ms assemble={:.3f} ms",
        ordering_total_time_ms,
        quotient_graph_time * 1000.0,
        decompose_time * 1000.0,
        local_permute_time * 1000.0,
        assemble_time * 1000.0);
    spdlog::info(
        "HOMA CPU ordering timing: quotient compact={:.3f} ms "
        "patch_nodes={:.3f} ms adjacency={:.3f} ms csr={:.3f} ms "
        "patches={} q_edges={}",
        quotient_patch_compaction_time_ms,
        quotient_patch_nodes_time_ms,
        quotient_adjacency_time_ms,
        quotient_csr_time_ms,
        _quotient_graph._Q_n,
        quotient_edges);
    spdlog::info(
        "HOMA CPU ordering timing: decompose partition_sum={:.3f} ms "
        "separator_sum={:.3f} ms bookkeeping_sum={:.3f} ms nodes={} "
        "separator_vertices={}",
        decompose_partition_time_ms,
        decompose_separator_time_ms,
        decompose_bookkeeping_time_ms,
        decompose_processed_nodes,
        decompose_separator_vertices);
    spdlog::info(
        "HOMA CPU ordering timing: local subgraphs={:.3f} ms "
        "local_order_sum={:.3f} ms blocks={}",
        local_subgraph_time_ms,
        local_order_time_ms,
        local_order_blocks);
}

void CPUOrdering_PATCH::reset_timing_stats()
{
    quotient_graph_time = 0;
    decompose_time = 0;
    local_permute_time = 0;
    assemble_time = 0;
    ordering_total_time_ms = 0;

    quotient_patch_compaction_time_ms = 0;
    quotient_patch_nodes_time_ms = 0;
    quotient_adjacency_time_ms = 0;
    quotient_csr_time_ms = 0;

    decompose_partition_time_ms = 0;
    decompose_separator_time_ms = 0;
    decompose_bookkeeping_time_ms = 0;
    decompose_processed_nodes = 0;
    decompose_separator_vertices = 0;

    local_subgraph_time_ms = 0;
    local_order_time_ms = 0;
    local_order_blocks = 0;
}

void CPUOrdering_PATCH::reset(){
    reset_timing_stats();
    this->_Gp = nullptr;
    this->_Gi = nullptr;
    this->_G_n = 0;
    this->_G_nnz = 0;
    this->_num_patches = -1;
    this->_decompose_marker.clear();
    this->_decompose_local_index.clear();
    this->_decomposition_tree.decomposition_level = -1;
    this->_decomposition_tree.decomposition_nodes.clear();
    this->_decomposition_tree.decomposition_node_offset.clear();
    this->_decomposition_tree.is_sep.clear();
    this->_decomposition_tree.g_node_to_tree_node.clear();
    this->_decomposition_tree._num_patches = -1;
}

}  // namespace homa
