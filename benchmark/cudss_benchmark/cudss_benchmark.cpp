//
// Created by behrooz on 2025-09-28.
//


#include <igl/cotmatrix.h>
#include <igl/read_triangle_mesh.h>
#include <spdlog/spdlog.h>
#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include <chrono>
#include <unsupported/Eigen/SparseExtra>
#include <iostream>
#include <filesystem>

#include "LinSysSolver.hpp"
#include "get_factor_nnz.h"
#include "check_valid_permutation.h"
#include "ordering.h"
#include "remove_diagonal.h"
#include "parth/parth.h"
#include "csv_utils.h"


struct CLIArgs
{
    std::string input_mesh;
    int binary_level = 9; // It is zero-based so 9 is 10 levels
    std::string output_csv_address;//Include absolute path with csv file name without .csv extension
    std::string solver_type   = "CHOLMOD";
    std::string ordering_type = "DEFAULT";
    std::string patch_type = "rxmesh";
    int patch_size = 24;
    bool use_gpu = false;

    CLIArgs(int argc, char* argv[])
    {
        CLI::App app{"Separator analysis"};
        app.add_option("-a,--ordering", ordering_type, "ordering type");
        app.add_option("-s,--solver", solver_type, "solver type");
        app.add_option("-o,--output", output_csv_address, "output folder name");
        app.add_option("-i,--input", input_mesh, "input mesh name");
        app.add_option("-g,--use_gpu", use_gpu, "use gpu");
        app.add_option("-p,--patch_type", patch_type, "how to patch the graph/mesh");
        app.add_option("-z,--patch_size", patch_size, "patch size");
        app.add_option("-b,--binary_level", binary_level, "binary level for binary tree ordering");

        try {
            app.parse(argc, argv);
        } catch (const CLI::ParseError& e) {
            exit(app.exit(e));
        }
    }
};


Eigen::SparseMatrix<double> computeSmootherMatrix(Eigen::MatrixXd& V, Eigen::MatrixXi& F)
{
    Eigen::SparseMatrix<double> L;
    igl::cotmatrix(V, F, L);
    // Make sure the matrix is semi-positive definite by adding values to the diagonal
    L.diagonal().array() += 100;
    return L;

}


int main(int argc, char* argv[])
{
    // Load the mesh
    CLIArgs args(argc, argv);

    if (args.input_mesh.empty()) {
        std::cerr << "Error: Input mesh file not specified. Use -i or --input "
                     "to specify the mesh file."
                  << std::endl;
        return 1;
    }

    std::cout << "Loading mesh from: " << args.input_mesh << std::endl;
    std::cout << "Output folder: " << args.output_csv_address << std::endl;

    Eigen::MatrixXd OV;
    Eigen::MatrixXi OF;
    if (!igl::read_triangle_mesh(args.input_mesh, OV, OF)) {
        std::cerr << "Failed to read the mesh: " << args.input_mesh
                  << std::endl;
        return 1;
    }

    // Create laplacian matrix
    Eigen::SparseMatrix<double> OL;
    igl::cotmatrix(OV, OF, OL);

    // Print laplacian size and sparsity
    spdlog::info("Number of rows: {}", OL.rows());
    spdlog::info("Number of non-zeros: {}", OL.nonZeros());
    spdlog::info(
        "Sparsity: {:.2f}%",
        (1 - (OL.nonZeros() / static_cast<double>(OL.rows() * OL.rows()))) *
            100);

    // Make sure the matrix is symmetric positive definite by adding to diagonal
    // The cotangent Laplacian is negative semi-definite, so we add a constant
    // to shift all eigenvalues to be positive
    for (int i = 0; i < OL.rows(); ++i) {
        OL.coeffRef(i, i) += 300.0;
    }
    Eigen::VectorXd rhs = Eigen::VectorXd::Random(OL.rows());
    Eigen::VectorXd result;

    // Init permuter
    std::vector<int>         perm;
    std::vector<int> etree;
    RXMESH_SOLVER::Ordering* ordering = nullptr;
    if (args.ordering_type == "DEFAULT") {
        spdlog::info("Using default ordering (default for each solver).");
        ordering = nullptr;
    } else if (args.ordering_type == "METIS") {
        spdlog::info("Using METIS ordering.");
        ordering = RXMESH_SOLVER::Ordering::create(
            RXMESH_SOLVER::DEMO_ORDERING_TYPE::METIS);
        if(args.solver_type == "CUDSS") {
            std::cerr << "METIS ordering is not supported with CUDSS solver." << std::endl;
            return 1;
        }
    } else if (args.ordering_type == "RXMESH_ND") {
        spdlog::info("Using RXMESH ordering.");
        ordering = RXMESH_SOLVER::Ordering::create(
            RXMESH_SOLVER::DEMO_ORDERING_TYPE::RXMESH_ND);
    } else if (args.ordering_type == "PATCH_ORDERING") {
        spdlog::info("Using PATCH_ORDERING ordering.");
        ordering = RXMESH_SOLVER::Ordering::create(
            RXMESH_SOLVER::DEMO_ORDERING_TYPE::PATCH_ORDERING);
        ordering->setOptions(
            {{"use_gpu", args.use_gpu ? "1" : "0"},
                {"patch_type", args.patch_type},
                {"patch_size", std::to_string(args.patch_size)},
                {"binary_level", std::to_string(args.binary_level)}});
    } else if (args.ordering_type == "PARTH") {
        ordering = RXMESH_SOLVER::Ordering::create(
            RXMESH_SOLVER::DEMO_ORDERING_TYPE::PARTH);
        ordering->setOptions({{"binary_level", std::to_string(args.binary_level)}});
    } else if (args.ordering_type == "NEUTRAL"){
        spdlog::info("Using NEUTRAL ordering.");
        ordering = RXMESH_SOLVER::Ordering::create(
            RXMESH_SOLVER::DEMO_ORDERING_TYPE::NEUTRAL);
    } else {
        spdlog::error("Unknown Ordering type.");
    }

    //Init solver
    RXMESH_SOLVER::LinSysSolver* solver = nullptr;
    if (args.solver_type == "CHOLMOD") {
        solver = RXMESH_SOLVER::LinSysSolver::create(
            RXMESH_SOLVER::LinSysSolverType::CPU_CHOLMOD);
        spdlog::info("Using CHOLMOD direct solver.");
    } else if (args.solver_type == "CUDSS") {
        solver = RXMESH_SOLVER::LinSysSolver::create(
            RXMESH_SOLVER::LinSysSolverType::GPU_CUDSS);
        spdlog::info("Using CUDSS direct solver.");
    } else if (args.solver_type == "PARTH_SOLVER") {
        solver = RXMESH_SOLVER::LinSysSolver::create(
            RXMESH_SOLVER::LinSysSolverType::PARTH_SOLVER);
        spdlog::info("Using PARTH direct solver.");
    } else if (args.solver_type == "STRUMPACK"){
        solver = RXMESH_SOLVER::LinSysSolver::create(
            RXMESH_SOLVER::LinSysSolverType::GPU_STRUMPACK);
    } else {
        spdlog::error("Unknown solver type.");
    }
    assert(solver != nullptr);

    // Create the graph
    std::vector<int> Gp;
    std::vector<int> Gi;
    RXMESH_SOLVER::remove_diagonal(
        OL.rows(), OL.outerIndexPtr(), OL.innerIndexPtr(), Gp, Gi);


    long int ordering_init_time = -1;
    long int ordering_time = -1;
    long int analysis_time = -1;
    long int factorization_time = -1;
    long int solve_time = -1;
    long int factor_nnz = -1;
    // Init the permuter
    if (ordering != nullptr) {
        // Provide mesh data if the ordering needs it (e.g., RXMesh ND)
        if (ordering->needsMesh()) {
            // Pass raw pointers to avoid ABI issues between C++ and CUDA compilation
            ordering->setMesh(OV.data(), OV.rows(), OV.cols(),
                            OF.data(), OF.rows(), OF.cols());
        }
        ordering->setGraph(Gp.data(), Gi.data(), OL.rows(), Gi.size());
        auto ordering_init_start = std::chrono::high_resolution_clock::now();
        ordering->init();
        auto ordering_init_end = std::chrono::high_resolution_clock::now();
        ordering_init_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                         ordering_init_end - ordering_init_start)
                         .count();
        spdlog::info("Ordering initialization time: {} ms",
                     ordering_init_time);
        auto ordering_start = std::chrono::high_resolution_clock::now();
        if(args.solver_type=="CUDSS") {
            ordering->compute_permutation(perm, etree, true);
        } else {
            ordering->compute_permutation(perm, etree, false);
        }

        auto ordering_end = std::chrono::high_resolution_clock::now();
        ordering_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            ordering_end - ordering_start)
            .count();
            
        //Check for correct perm
        if (!RXMESH_SOLVER::check_valid_permutation(perm.data(), perm.size())) {
            spdlog::error("Permutation is not valid!");
        }
        spdlog::info("Ordering time: {} ms",
                     ordering_time);
        assert(perm.size() == OL.rows());

        factor_nnz = RXMESH_SOLVER::get_factor_nnz(OL.outerIndexPtr(),
                                                       OL.innerIndexPtr(),
                                                       OL.valuePtr(),
                                                       OL.rows(),
                                                       OL.nonZeros(),
                                                       perm);
        spdlog::info(
            "The ratio of factor non-zeros to matrix non-zeros given custom reordering: {}",
            (factor_nnz * 1.0 /OL.nonZeros()));
        solver->ordering_name = ordering->typeStr();
    }

    //Save the matrix
    Eigen::saveMarket(OL, "/home/behrooz/Desktop/Last_Project/RXMesh-dev/output/nefertiti.mtx");
    solver->setMatrix(OL.outerIndexPtr(),
                      OL.innerIndexPtr(),
                      OL.valuePtr(),
                      OL.rows(),
                      OL.nonZeros());

    // Symbolic analysis time
    auto start = std::chrono::high_resolution_clock::now();
    solver->analyze_pattern(perm, etree);
    auto end = std::chrono::high_resolution_clock::now();
    analysis_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    spdlog::info(
        "Analysis time: {} ms",
        analysis_time);

    // Factorization time
    start = std::chrono::high_resolution_clock::now();
    solver->factorize();
    end = std::chrono::high_resolution_clock::now();
    factorization_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    spdlog::info(
        "Factorization time: {} ms",
        factorization_time);

    // Solve time
    start = std::chrono::high_resolution_clock::now();
    solver->solve(rhs, result);
    end = std::chrono::high_resolution_clock::now();
    solve_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    spdlog::info(
        "Solve time: {} ms",
        solve_time);

    // Compute residual
    assert(OL.rows() == OL.cols());
    double residual = (rhs - OL * result).norm();
    spdlog::info("Residual: {}", residual);
    spdlog::info("Final factor/matrix NNZ ratio: {}",
                 solver->getFactorNNZ() * 1.0 / OL.nonZeros());


    //Save data to a csv file
    std::string csv_name = args.output_csv_address;
    std::string mesh_name = std::filesystem::path(args.input_mesh).stem().string();
    std::vector<std::string> header;
    header.emplace_back("mesh_name");
    header.emplace_back("G_N");
    header.emplace_back("G_NNZ");
    header.emplace_back("solver_type");
    header.emplace_back("ordering_type");
    header.emplace_back("nd_levels");
    header.emplace_back("patch_type");
    header.emplace_back("patch_size");
    header.emplace_back("factor/matrix NNZ ratio");
    header.emplace_back("ordering_time");
    header.emplace_back("analysis_time");
    header.emplace_back("factorization_time");
    header.emplace_back("solve_time");
    header.emplace_back("residual");


    PARTH::CSVManager runtime_csv(csv_name, "some address", header, false);
    runtime_csv.addElementToRecord(mesh_name, "mesh_name");
    runtime_csv.addElementToRecord(solver->N, "G_N");
    runtime_csv.addElementToRecord(solver->NNZ, "G_NNZ");
    runtime_csv.addElementToRecord(args.solver_type, "solver_type");
    if(ordering!=nullptr) {
    runtime_csv.addElementToRecord(ordering->typeStr(), "ordering_type");
    } else {
        runtime_csv.addElementToRecord("DEFAULT", "ordering_type");
    }
    int nd_levels = std::log2(etree.size() + 1);
    runtime_csv.addElementToRecord(nd_levels, "nd_levels");
    runtime_csv.addElementToRecord(args.patch_type, "patch_type");
    runtime_csv.addElementToRecord(args.patch_size, "patch_size");
    runtime_csv.addElementToRecord(factor_nnz * 1.0 / OL.nonZeros(), "factor/matrix NNZ ratio");
    runtime_csv.addElementToRecord(ordering_time, "ordering_time");
    runtime_csv.addElementToRecord(analysis_time, "analysis_time");
    runtime_csv.addElementToRecord(factorization_time, "factorization_time");
    runtime_csv.addElementToRecord(solve_time, "solve_time");
    runtime_csv.addElementToRecord(residual, "residual");
    runtime_csv.addRecord();


    delete solver;
    delete ordering;
    return 0;
}