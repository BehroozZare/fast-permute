//
// Created by behrooz on 2025-01-15.
// Generates separator data from mesh ordering
//

#include <igl/read_triangle_mesh.h>
#include <spdlog/spdlog.h>
#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include <iostream>
#include <filesystem>
#include <algorithm>

#include "SPD_cot_matrix.h"
#include "ordering.h"
#include "remove_diagonal.h"
#include "check_valid_permutation.h"
#include "save_vector.h"
#include "utils/create_patch_with_metis.h"

struct CLIArgs
{
    std::string input_mesh = "/media/behrooz/FarazHard/Last_Project/BenchmarkMesh/tri-mesh/final/dragon.obj";
    std::string output_folder = "/home/behrooz/Desktop/Last_Project/gpu_ordering/output/render_data";
    int binary_level = 7;
    std::string ordering_type = "PATCH_ORDERING";
    std::string patch_type = "rxmesh";
    int patch_size = 512;
    int patch_number = -1;
    bool use_gpu = false;
    int use_patch_separator = 1;
    std::string patch_ordering_local_permute_method = "amd";

    CLIArgs(int argc, char* argv[])
    {
        CLI::App app{"Separator Data Generator"};
        app.add_option("-i,--input", input_mesh, "input mesh name");
        app.add_option("-o,--output", output_folder, "output folder name");
        app.add_option("-a,--ordering", ordering_type, "ordering type");
        app.add_option("-p,--patch_type", patch_type, "how to patch the graph/mesh");
        app.add_option("-z,--patch_size", patch_size, "patch size");
        app.add_option("-n,--patch_number", patch_number, "patch number");
        try {
            app.parse(argc, argv);
        } catch (const CLI::ParseError& e) {
            exit(app.exit(e));
        }
    }
};


int main(int argc, char* argv[])
{
    // Parse CLI arguments
    CLIArgs args(argc, argv);

    std::cout << "Loading mesh from: " << args.input_mesh << std::endl;
    // Load the mesh
    Eigen::MatrixXd OV;
    Eigen::MatrixXi OF;
    if (!igl::read_triangle_mesh(args.input_mesh, OV, OF)) {
        std::cerr << "Failed to read the mesh: " << args.input_mesh << std::endl;
        return 1;
    }


    // Create SPD cotangent matrix (already positive definite with regularization)
    Eigen::SparseMatrix<double> OL;
    RXMESH_SOLVER::computeSPD_cot_matrix(OV, OF, OL);

    if (args.patch_number != -1) {
        spdlog::info("Using patch number: {}", args.patch_number);
        args.patch_size = OL.rows() / args.patch_number;
        spdlog::info("Using patch size: {}", args.patch_size);
    } else {
        spdlog::info("Using patch size: {}", args.patch_size);
    }

    // Print matrix info
    spdlog::info("Number of rows: {}", OL.rows());
    spdlog::info("Number of non-zeros: {}", OL.nonZeros());
    spdlog::info(
        "Sparsity: {:.2f}%",
        (1 - (OL.nonZeros() / static_cast<double>(OL.rows() * OL.rows()))) * 100);

    // Create the graph (remove diagonal for ordering)
    std::vector<int> Gp;
    std::vector<int> Gi;
    RXMESH_SOLVER::remove_diagonal(
        OL.rows(), OL.outerIndexPtr(), OL.innerIndexPtr(), Gp, Gi);

    // Init permuter
    std::vector<int> perm;
    std::vector<int> etree;
    RXMESH_SOLVER::Ordering* ordering = nullptr;
    std::string mesh_name = std::filesystem::path(args.input_mesh).stem().string();

    if (args.ordering_type == "PATCH_ORDERING") {
        spdlog::info("Using PATCH_ORDERING ordering.");
        ordering = RXMESH_SOLVER::Ordering::create(
            RXMESH_SOLVER::DEMO_ORDERING_TYPE::PATCH_ORDERING);
        ordering->setOptions(
            {{"use_gpu", args.use_gpu ? "1" : "0"},
             {"patch_type", args.patch_type},
             {"patch_size", std::to_string(args.patch_size)},
             {"use_patch_separator", std::to_string(args.use_patch_separator)},
             {"patch_ordering_local_permute_method", args.patch_ordering_local_permute_method},
             {"binary_level", std::to_string(args.binary_level)}});
    } else if (args.ordering_type == "PARTH") {
        spdlog::info("Using PARTH ordering.");
        ordering = RXMESH_SOLVER::Ordering::create(
            RXMESH_SOLVER::DEMO_ORDERING_TYPE::PARTH);
        ordering->setOptions({{"binary_level", std::to_string(args.binary_level)}});
    } else {
        spdlog::error("Unknown ordering type: {}", args.ordering_type);
        return 1;
    }

    // Provide mesh data if the ordering needs it (e.g., RXMesh ND)
    if (ordering->needsMesh()) {
        ordering->setMesh(OV.data(), OV.rows(), OV.cols(),
                          OF.data(), OF.rows(), OF.cols());
    }
    ordering->setGraph(Gp.data(), Gi.data(), OL.rows(), Gi.size());

    // Create patches with METIS for PATCH_ORDERING, otherwise use init()
    std::vector<int> node_to_patch;
    if (args.ordering_type == "PATCH_ORDERING") {
        spdlog::info("Creating patches with METIS...");
        RXMESH_SOLVER::create_patch_with_metis(OL.rows(), OL.outerIndexPtr(), OL.innerIndexPtr(),
            1, args.patch_size, node_to_patch);
        ordering->setPatch(node_to_patch);
        spdlog::info("Patches created with {} nodes", node_to_patch.size());
    } else {
        spdlog::info("Initializing ordering...");
        ordering->init();
    }

    spdlog::info("Computing permutation...");
    ordering->compute_permutation(perm, etree, true); // Get level-based etree

    // Validate permutation
    if (!RXMESH_SOLVER::check_valid_permutation(perm.data(), perm.size())) {
        spdlog::error("Permutation is not valid!");
        delete ordering;
        return 1;
    }
    std::vector<std::pair<int, int>> node_to_etree_mapping;
    ordering->getNodeToEtreeMapping(node_to_etree_mapping);
    spdlog::info("node to etree mapping size: {}", node_to_etree_mapping.size());
    spdlog::info("Permutation size: {}", perm.size());
    spdlog::info("Etree size: {}", etree.size());
    spdlog::info("Ordering complete.");

    //Save the node to etree mapping into two separate files
    std::vector<int> assigned_nodes;
    std::vector<int> etree_nodes;
    for(auto& mapping : node_to_etree_mapping){
        assigned_nodes.push_back(mapping.first);
        etree_nodes.push_back(mapping.second);
    }

    // Calculate number of patches for file naming
    int num_patches = 0;
    if (!node_to_patch.empty()) {
        num_patches = *std::max_element(node_to_patch.begin(), node_to_patch.end()) + 1;
    }
    std::string patch_suffix = (num_patches > 0) ? "_patches" + std::to_string(num_patches) : "";

    RXMESH_SOLVER::save_vector_to_file(assigned_nodes, args.output_folder + "/" + args.ordering_type + "_assigned_nodes_" + mesh_name + patch_suffix + ".txt");
    RXMESH_SOLVER::save_vector_to_file(etree_nodes, args.output_folder + "/" + args.ordering_type + "_etree_nodes_" + mesh_name + patch_suffix + ".txt");
    spdlog::info("Node to etree mapping saved to: {}", args.output_folder + "/" + args.ordering_type + "_assigned_nodes_" + mesh_name + patch_suffix + ".txt");
    spdlog::info("Etree nodes saved to: {}", args.output_folder + "/" + args.ordering_type + "_etree_nodes_" + mesh_name + patch_suffix + ".txt");

    // Save vertex to patch ID mapping
    if (args.ordering_type == "PATCH_ORDERING" && !node_to_patch.empty()) {
        std::string patch_mapping_file = args.output_folder + "/" + mesh_name + "_vertex_to_patch" + patch_suffix + ".txt";
        RXMESH_SOLVER::save_vector_to_file(node_to_patch, patch_mapping_file);
        spdlog::info("Vertex to patch mapping saved to: {}", patch_mapping_file);
    }

    delete ordering;
    return 0;
}
