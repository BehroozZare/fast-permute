//
// Created by behrooz on 2026-01-22.
// Loads a mesh, computes an ordering permutation, forms A_perm = P*A*P^T,
// and saves both matrices as MatrixMarket (.mtx) to an output folder.
//

#include <igl/read_triangle_mesh.h>
#include <spdlog/spdlog.h>
#include <CLI/CLI.hpp>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>  // Eigen::saveMarket

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "SPD_cot_matrix.h"
#include "ordering.h"
#include "remove_diagonal.h"
#include "check_valid_permutation.h"
#include "save_vector.h"

struct CLIArgs
{
    std::string input_mesh =
        "/media/behrooz/FarazHard/Last_Project/BenchmarkMesh/tri-mesh/final/beetle.obj";
    std::string output_folder =
        "/home/behrooz/Desktop/Last_Project/gpu_ordering/output/render_data";

    int         binary_level = 7;
    std::string ordering_type = "PARTH";

    // PATCH_ORDERING options (mirrors gen_sep_data.cpp)
    std::string patch_type = "rxmesh";
    int         patch_size = 512;
    bool        use_gpu = false;
    int         use_patch_separator = 1;
    std::string patch_ordering_local_permute_method = "amd";

    CLIArgs(int argc, char* argv[])
    {
        CLI::App app{"Save matrix + permuted matrix (MatrixMarket)"};
        app.add_option("-i,--input", input_mesh, "input mesh name");
        app.add_option("-o,--output", output_folder, "output folder name");
        app.add_option("-a,--ordering", ordering_type, "ordering type (PATCH_ORDERING|PARTH)");

        app.add_option("-p,--patch_type", patch_type, "how to patch the graph/mesh");
        app.add_option("-z,--patch_size", patch_size, "patch size");
        app.add_option("-b,--binary_level", binary_level, "binary level for binary tree ordering");
        app.add_option("-g,--use_gpu", use_gpu, "use gpu");
        app.add_option("-u,--use_patch_separator", use_patch_separator, "use patch separator");
        app.add_option("-m,--patch_ordering_local_permute_method",
                       patch_ordering_local_permute_method,
                       "patch ordering local permute method");
        try {
            app.parse(argc, argv);
        } catch (const CLI::ParseError& e) {
            exit(app.exit(e));
        }
    }
};

static Eigen::SparseMatrix<double>
makePermutationMatrixFromVector(const std::vector<int>& perm)
{
    const int n = static_cast<int>(perm.size());
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(n);

    for (int i = 0; i < n; ++i) {
        triplets.emplace_back(i, perm[i], 1.0);
    }

    Eigen::SparseMatrix<double> P(n, n);
    P.setFromTriplets(triplets.begin(), triplets.end());
    return P;
}

int main(int argc, char* argv[])
{
    CLIArgs args(argc, argv);

    spdlog::info("Loading mesh from: {}", args.input_mesh);
    spdlog::info("Output folder: {}", args.output_folder);

    // Load mesh
    Eigen::MatrixXd OV;
    Eigen::MatrixXi OF;
    if (!igl::read_triangle_mesh(args.input_mesh, OV, OF)) {
        spdlog::error("Failed to read the mesh: {}", args.input_mesh);
        return 1;
    }

    // Compute SPD cotangent matrix A
    Eigen::SparseMatrix<double> A;
    RXMESH_SOLVER::computeSPD_cot_matrix(OV, OF, A);

    spdlog::info("A rows: {}", A.rows());
    spdlog::info("A nnz: {}", A.nonZeros());

    // Create the graph (remove diagonal for ordering)
    std::vector<int> Gp;
    std::vector<int> Gi;
    RXMESH_SOLVER::remove_diagonal(A.rows(), A.outerIndexPtr(), A.innerIndexPtr(), Gp, Gi);

    // Create ordering
    std::vector<int> perm;
    std::vector<int> etree;
    RXMESH_SOLVER::Ordering* ordering = nullptr;

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
        ordering =
            RXMESH_SOLVER::Ordering::create(RXMESH_SOLVER::DEMO_ORDERING_TYPE::PARTH);
        ordering->setOptions({{"binary_level", std::to_string(args.binary_level)}});
    } else {
        spdlog::error("Unknown ordering type: {}", args.ordering_type);
        return 1;
    }

    // Provide mesh data if the ordering needs it (e.g., RXMesh ND)
    if (ordering->needsMesh()) {
        ordering->setMesh(OV.data(),
                          static_cast<int>(OV.rows()),
                          static_cast<int>(OV.cols()),
                          OF.data(),
                          static_cast<int>(OF.rows()),
                          static_cast<int>(OF.cols()));
    }
    ordering->setGraph(Gp.data(), Gi.data(), A.rows(), static_cast<int>(Gi.size()));

    spdlog::info("Initializing ordering...");
    ordering->init();

    spdlog::info("Computing permutation...");
    ordering->compute_permutation(perm, etree, true);
    if (!RXMESH_SOLVER::check_valid_permutation(perm.data(), static_cast<int>(perm.size()))) {
        spdlog::error("Permutation is not valid!");
        delete ordering;
        return 1;
    }
    if (static_cast<int>(perm.size()) != A.rows()) {
        spdlog::error("Permutation size ({}) != matrix size ({})", perm.size(), A.rows());
        delete ordering;
        return 1;
    }

    // Compute permuted matrix: A_perm = P * A * P^T
    spdlog::info("Computing permuted matrix (A_perm = P * A * P^T) ...");
    Eigen::SparseMatrix<double> P = makePermutationMatrixFromVector(perm);
    Eigen::SparseMatrix<double> A_perm = P * A * P.transpose();
    A_perm.makeCompressed();

    // Save outputs
    std::filesystem::create_directories(args.output_folder);
    const std::string mesh_stem = std::filesystem::path(args.input_mesh).stem().string();

    const std::string A_path = (std::filesystem::path(args.output_folder) /
                                (mesh_stem + "_orig.mtx"))
                                   .string();
    const std::string A_perm_path = (std::filesystem::path(args.output_folder) /
                                     (mesh_stem + "_" + args.ordering_type + "_perm.mtx"))
                                        .string();
    const std::string perm_path = (std::filesystem::path(args.output_folder) /
                                   (mesh_stem + "_" + args.ordering_type + "_perm.txt"))
                                      .string();

    spdlog::info("Saving A to: {}", A_path);
    if (!Eigen::saveMarket(A, A_path)) {
        spdlog::error("Failed to save A to: {}", A_path);
        delete ordering;
        return 1;
    }

    spdlog::info("Saving A_perm to: {}", A_perm_path);
    if (!Eigen::saveMarket(A_perm, A_perm_path)) {
        spdlog::error("Failed to save A_perm to: {}", A_perm_path);
        delete ordering;
        return 1;
    }

    spdlog::info("Saving perm to: {}", perm_path);
    RXMESH_SOLVER::save_vector_to_file(perm, perm_path);

    delete ordering;
    spdlog::info("Done.");
    return 0;
}
