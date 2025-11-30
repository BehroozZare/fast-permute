//
// Created by behrooz on 2025-09-29.
//

#include "patch_ordering.h"
#include "csv_utils.h"

namespace RXMESH_SOLVER {

    PatchOrdering::~PatchOrdering()
{
}

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
    if(_patch_ordering_type == PatchOrderingType::RXMESH_PATCH) {
        rxmesh::rx_init(0);
        _rxmesh = std::make_unique<rxmesh::RXMeshStatic>(_fv, "", this->_patch_size);

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
        _rxmesh = std::make_unique<rxmesh::RXMeshStatic>(_fv, "", this->_patch_size, true);

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
    }


    if(_use_gpu) {
        this->_gpu_order.init_patches(this->_num_patches, this->_g_node_to_patch);
    } else {
        this->_cpu_order.init_patches(this->_num_patches, this->_g_node_to_patch);
    }


}
bool PatchOrdering::needsMesh() const
{
    return true;
}

void PatchOrdering::compute_permutation(std::vector<int>& perm, std::vector<int>& etree, bool compute_etree)
{
    assert(m_has_mesh);
    if(_use_gpu) {
        this->_gpu_order.compute_permutation(perm);
    } else {
        this->_cpu_order.compute_permutation(perm);
    }
    if(compute_etree) {
        spdlog::info("Getting etree for Patch ordering is not supported.");
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
        } else {
            throw std::runtime_error("Invalid patch type: " + patch_type);
        }
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

    PARTH::CSVManager runtime_csv(csv_name, "some address", header,
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



}