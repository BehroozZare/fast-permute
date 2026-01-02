//
// Created by behrooz on 2025-09-29.
//

#include "patch_ordering.h"
#include "csv_utils.h"
#include <rxmesh/rxmesh_static.h>

namespace RXMESH_SOLVER {

// Define the custom deleter - requires complete type
void RXMeshDeleter::operator()(rxmesh::RXMeshStatic* ptr) const {
    delete ptr;
}

PatchOrdering::~PatchOrdering() = default;

void PatchOrdering::setGraph(int* Gp, int* Gi, int G_N, int NNZ)
{
    this->Gp = Gp;
    this->Gi = Gi;
    this->G_N = G_N;
    this->G_NNZ = NNZ;
    // Also set the graph in the base GPUOrdering class
    if(_use_gpu) {
        this->_gpu_order.setGraph(Gp, Gi, G_N, NNZ);
    } else {
        this->_cpu_order.setGraph(Gp, Gi, G_N, NNZ);
    }
}

void PatchOrdering::setMesh(const double* V_data, int V_rows, int V_cols,
                          const int* F_data, int F_rows, int F_cols)
{
    m_has_mesh = true;
    spdlog::info("Mesh has {} vertices and {} faces", V_rows, F_rows);
    spdlog::info("Faces have {} vertices each", F_cols);

    // Convert raw data to std::vector format for RXMesh
    _fv.resize(F_rows);
    for (int i = 0; i < F_rows; ++i) {
        _fv[i].resize(F_cols);
        for (int j = 0; j < F_cols; ++j) {
            // Eigen stores data in column-major order by default
            _fv[i][j] = static_cast<uint32_t>(F_data[i + j * F_rows]);
        }
    }


    // Optionally add vertex coordinates (not strictly needed for ND ordering)
    _vertices.resize(V_rows);
    for (int i = 0; i < V_rows; ++i) {
        _vertices[i].resize(V_cols);
        for (int j = 0; j < V_cols; ++j) {
            // Eigen stores data in column-major order by default
            _vertices[i][j] = static_cast<float>(V_data[i + j * V_rows]);
        }
    }
}

void PatchOrdering::init(){
    spdlog::info("Initializing Patches for {} binary level", this->_binary_level);
    int number_of_leaves_in_etree = (1 << (this->_binary_level));
    spdlog::info("Number of leaves in the etree: {}", number_of_leaves_in_etree);
    int number_of_patches = 2 * number_of_leaves_in_etree;
    this->_patch_size = G_N / number_of_patches;
    spdlog::info(" First estimation Number of patches: {} with patch size {}", number_of_patches, _patch_size);
    //Round to nearest power of 2
    size_t power = std::ceil(std::log2(this->_patch_size));
    this->_patch_size = 1 << power;
    //Make the upperbound to 512
    if (this->_patch_size > 512) {
        this->_patch_size = 512;
    }
    if (this->_patch_size < 128) {
        _patch_size = 128;
    }
    spdlog::info("Final Patch size: {}", this->_patch_size);

    if (_patch_size == 1) {
        _num_patches = G_N;
        _g_node_to_patch.resize(G_N);
        for (int i = 0; i < G_N; ++i) {
            _g_node_to_patch[i] = i;
        }
        spdlog::info("Skip patching because patch size is 1.");
    } else if(_patch_ordering_type == PatchOrderingType::RXMESH_PATCH) {
        rxmesh::rx_init(0);
        _rxmesh.reset(new rxmesh::RXMeshStatic(_fv, "", this->_patch_size));
        _patching_time = _rxmesh->get_patching_time();
        spdlog::info(
            "RXMesh initialized with {} vertices, {} edges, {} faces, {} patches",
            _rxmesh->get_num_vertices(),
            _rxmesh->get_num_edges(),
            _rxmesh->get_num_faces(),
            _rxmesh->get_num_patches());

        this->_g_node_to_patch.resize(_rxmesh->get_num_vertices());
        this->_num_patches = _rxmesh->get_num_patches();
        _rxmesh->for_each_vertex(
            rxmesh::HOST,
            [&](const rxmesh::VertexHandle vh) {
                uint32_t node_id       = _rxmesh->map_to_global(vh);
                this->_g_node_to_patch[node_id] = static_cast<int>(vh.patch_id());
            },
            NULL,
            false);
    } else if(_patch_ordering_type == PatchOrderingType::METIS_KWAY_PATCH) {
        rxmesh::rx_init(0);
        _rxmesh.reset(new rxmesh::RXMeshStatic(_fv, "", this->_patch_size, true));
        _patching_time = _rxmesh->get_patching_time();
        spdlog::info(
            "RXMesh initialized with {} vertices, {} edges, {} faces, {} patches",
            _rxmesh->get_num_vertices(),
            _rxmesh->get_num_edges(),
            _rxmesh->get_num_faces(),
            _rxmesh->get_num_patches());

        this->_g_node_to_patch.resize(_rxmesh->get_num_vertices());
        this->_num_patches = _rxmesh->get_num_patches();
        _rxmesh->for_each_vertex(
            rxmesh::HOST,
            [&](const rxmesh::VertexHandle vh) {
                uint32_t node_id       = _rxmesh->map_to_global(vh);
                this->_g_node_to_patch[node_id] = static_cast<int>(vh.patch_id());
            },
            NULL,
            false);
    } else if(_patch_ordering_type == PatchOrderingType::METIS_SEPARATOR_PATCH) {
        // TODO: Implement METIS separator patch ordering
    } else if(_patch_ordering_type == PatchOrderingType::REUSE_PATCH) {
        //Reuse the previous patch
        spdlog::info("Reusing the previous patch to node mapping with size {}", this->_g_node_to_patch.size());
    }
    spdlog::info("Patching time is {}.", _patching_time);

    if(_use_gpu) {
        this->_gpu_order.init_patches(this->_num_patches, this->_g_node_to_patch, this->_binary_level);
    } else {
        this->_cpu_order.init_patches(this->_num_patches, this->_g_node_to_patch, this->_binary_level);
    }


}
bool PatchOrdering::needsMesh() const
{
    return true;
}


void PatchOrdering::get_level_numbering(int binary_tree_size, std::vector<int>& level_numbering) {
    level_numbering.clear();
    level_numbering.resize(binary_tree_size, 0);
    for(int i = 0; i < binary_tree_size; i++){
        int hmd_id = binary_tree_size - 1 - i;
        level_numbering[hmd_id] = i;
    }
}

void PatchOrdering::compute_etree(std::vector<int>& level_numbering, std::vector<int>& etree) {
    etree.clear();
    etree.resize(level_numbering.size(), 0);
    for (int hmd_id = 0; hmd_id < level_numbering.size(); hmd_id++) {
        int etree_idx = level_numbering[hmd_id];
        int etree_value = this->_cpu_order._decomposition_tree.decomposition_nodes[hmd_id].assigned_g_nodes.size();
        etree[etree_idx] = etree_value;
    }
}

void PatchOrdering::assemble_perm(std::vector<int>& level_numbering, std::vector<int>& perm) {
    perm.clear();
    perm.resize(_cpu_order._G_n, -1);
    std::vector<int> etree_inverse(level_numbering.size(), 0);
    for(int hmd_id = 0; hmd_id < level_numbering.size(); hmd_id++){
        etree_inverse[level_numbering[hmd_id]] = hmd_id;
    }

    int offset = 0;
    for(int i = 0; i < etree_inverse.size(); i++){
        int hmd_id = etree_inverse[i];
        auto& node = this->_cpu_order._decomposition_tree.decomposition_nodes[hmd_id];
        if (node.assigned_g_nodes.empty())
            continue;
        for (int local_node = 0; local_node < node.assigned_g_nodes.size(); local_node++) {
            int global_node = node.assigned_g_nodes[local_node];
            int perm_index  = node.local_new_labels[local_node] + offset;
            assert(global_node >= 0 && global_node < this->G_N &&
                    "Invalid global node index");
            assert(perm_index >= 0 && perm_index < perm.size() &&
                    "Permutation index out of bounds");
            assert(perm[perm_index] == -1 &&
                    "Permutation slot already filled - duplicate node!");
            perm[perm_index] = global_node;
        }
        offset += node.assigned_g_nodes.size();
    }
}



void PatchOrdering::compute_permutation(std::vector<int>& perm, std::vector<int>& etree, bool with_etree)
{
    assert(!_g_node_to_patch.empty());
    if(_use_gpu) {
        this->_gpu_order.compute_permutation(perm);
    } else {
        this->_cpu_order.compute_permutation(perm);
    }
    if(with_etree) {
        perm.clear();
        std::vector<int> level_numbering;
        get_level_numbering(_cpu_order._decomposition_tree.decomposition_nodes.size(), level_numbering);
        compute_etree(level_numbering, etree);
        assemble_perm(level_numbering, perm);
    }
}


DEMO_ORDERING_TYPE PatchOrdering::type() const
{
    return DEMO_ORDERING_TYPE::PATCH_ORDERING;
}

std::string PatchOrdering::typeStr() const
{
    return "PATCH_ORDERING";
}

void PatchOrdering::setOptions(const std::map<std::string, std::string>& options)
{
    if (options.find("local_permute_method") != options.end()) {
        if(_use_gpu) {
            this->_gpu_order.local_permute_method = options.at("local_permute_method");
        } else {
            this->_cpu_order.local_permute_method = options.at("local_permute_method");
        }
    }

    if (options.find("use_gpu") != options.end()) {
        this->_use_gpu = std::stoi(options.at("use_gpu"));
    }

    if (options.find("patch_type") != options.end()) {
        std::string patch_type = options.at("patch_type");
        if(patch_type == "rxmesh") {
            this->_patch_ordering_type = PatchOrderingType::RXMESH_PATCH;
        } else if(patch_type == "metis_kway") {
            this->_patch_ordering_type = PatchOrderingType::METIS_KWAY_PATCH;
        } else if(patch_type == "metis_separator") {
            this->_patch_ordering_type = PatchOrderingType::METIS_SEPARATOR_PATCH;
        } else if(patch_type == "reuse_patch") {
            this->_patch_ordering_type = PatchOrderingType::REUSE_PATCH;
        } else {
            throw std::runtime_error("Invalid patch type: " + patch_type);
        }
    }
        if (options.find("patch_size") != options.end()) {
            _patch_size = std::stoi(options.at("patch_size"));
        }

    if (options.find("binary_level") != options.end()) {
        _binary_level = std::stoi(options.at("binary_level"));
    }

    // if(options.find("separator_finding_method") != options.end()) {
    //     this->gpu_order.separator_finding_method = options.at("separator_finding_method");
    // } else {
    //     this->gpu_order.separator_finding_method = "max_degree";
    // }
    //
    // if (options.find("separator_refinement_method") != options.end()) {
    //     this->_gpu_order.separator_refinement_method = options.at("separator_refinement_method");
    // } else {
    //     this->_gpu_order.separator_refinement_method = "nothing";
    // }
}


double PatchOrdering::compute_separator_ratio()
{
    if(_use_gpu) {
        return this->_gpu_order.compute_separator_ratio();
    } else {
        return this->_cpu_order.compute_separator_ratio();
    }
    return 0.0;
}


void PatchOrdering::add_record(std::string save_address, std::map<std::string, double> extra_info, std::string mesh_name)
{
    std::string csv_name = save_address + "/sep_runtime_analysis";
    std::vector<std::string> header;
    header.emplace_back("mesh_name");
    header.emplace_back("G_N");
    header.emplace_back("G_NNZ");

    header.emplace_back("ordering_type");
    header.emplace_back("local_permute_method");
    // header.emplace_back("separator_finding_method");
    // header.emplace_back("separator_refinement_method");
    header.emplace_back("separator_ratio");
    header.emplace_back("fill-ratio");

    RXMESH_SOLVER::CSVManager runtime_csv(csv_name, "some address", header,
                                  false);
    runtime_csv.addElementToRecord(mesh_name, "mesh_name");
    runtime_csv.addElementToRecord(G_N, "G_N");
    runtime_csv.addElementToRecord(G_NNZ, "G_NNZ");
    runtime_csv.addElementToRecord(typeStr(), "ordering_type");
    if(_use_gpu) {
        runtime_csv.addElementToRecord(this->_gpu_order.local_permute_method, "local_permute_method");
    } else {
        runtime_csv.addElementToRecord(this->_cpu_order.local_permute_method, "local_permute_method");
    }
    // runtime_csv.addElementToRecord(gpu_order.separator_finding_method, "separator_finding_method");
    // runtime_csv.addElementToRecord(gpu_order.separator_refinement_method, "separator_refinement_method");
    runtime_csv.addElementToRecord(compute_separator_ratio(), "separator_ratio");
    runtime_csv.addElementToRecord(extra_info.at("fill-ratio"), "fill-ratio");
    runtime_csv.addRecord();
}


void PatchOrdering::reset()
{
    if(_use_gpu) {
        this->_gpu_order.reset();
    } else {
        this->_cpu_order.reset();
    }
}


void PatchOrdering::getEtree(std::vector<int> &new_labels, std::vector<int> &sep_ptr) {
    //Get the new labels and separator pointers
    auto& etree = _cpu_order._decomposition_tree.decomposition_nodes;
    int cnt = 0;
    new_labels.resize(G_N, -1);
    sep_ptr.resize(etree.size() + 1, 0);
    for(int i = 0; i < etree.size(); i++){
        auto& node = etree[i];
        for(int j = 0; j < node.assigned_g_nodes.size(); j++){
            new_labels[cnt] = node.assigned_g_nodes[j];
            cnt++;
        }
        sep_ptr[i + 1] = cnt;
    }
    assert(cnt == G_N);
    assert(sep_ptr[etree.size()] == G_N);
}

void PatchOrdering::getStatistics(std::map<std::string, double>& stat) {
    stat["patch_size"] = this->_patch_size;
    stat["binary_level"] = this->_binary_level;
    stat["patching_time"] = this->_patching_time;
    stat["num_patches"] = this->_num_patches;
}

void PatchOrdering::getPatch(std::vector<int> &patches) {
    patches = _g_node_to_patch;
}
}
