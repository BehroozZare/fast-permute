//
// Created by behrooz on 2025-09-28.
//
// SLIM Benchmark: Computes ordering once, performs symbolic analysis once,
// then loops through Hessian/gradient pairs for repeated factorization and solve.
//

#include <igl/read_triangle_mesh.h>
#include <spdlog/spdlog.h>
#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include <chrono>
#include <cmath>
#include <map>
#include <unsupported/Eigen/SparseExtra>
#include <iostream>
#include <filesystem>

#include "LinSysSolver.hpp"
#include "get_factor_nnz.h"
#include "check_valid_permutation.h"
#include "utils/slim_prep_benchmark.h"
#include "csv_utils.h"
#include "save_vector.h"
#include <parth/parth.h>
#include <ordering.h>

#ifndef _CUDA_ERROR_
#define _CUDA_ERROR_
inline void HandleError(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess) {
        printf("\n%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define CUDA_ERROR(err) (HandleError(err, __FILE__, __LINE__))
#endif

#define CUDA_SYNC_CHECK() do {                               \
    CUDA_ERROR(cudaGetLastError());                          \
    CUDA_ERROR(cudaDeviceSynchronize());                     \
} while(0)


struct CLIArgs
{
    int binary_level = 9;
    int DIM = 2;
    std::string output_csv_address = "/home/behrooz/Desktop/Last_Project/gpu_ordering/output/SLIM/slim";
    std::string solver_type   = "CUDSS";
    std::string ordering_type = "DEFAULT";
    std::string patch_type = "rxmesh";
    std::string check_point_address = "/media/behrooz/FarazHard/Last_Project/slim_benchmark/armadillo_cut_high";
    bool use_gpu = false;

    CLIArgs(int argc, char* argv[])
    {
        CLI::App app{"SLIM Benchmark - Multiple factorization with static sparsity pattern"};
        app.add_option("-a,--ordering", ordering_type, "Ordering type: DEFAULT, PATCH_ORDERING, PARTH");
        app.add_option("-s,--solver", solver_type, "Solver type: CUDSS, MKL");
        app.add_option("-o,--output", output_csv_address, "Output CSV file path (without .csv extension)");
        app.add_option("-p,--patch_type", patch_type, "Patch type for PATCH_ORDERING: rxmesh, metis");
        app.add_option("-b,--binary_level", binary_level, "Binary level for nested dissection tree");
        app.add_option("-k,--check_point_address", check_point_address, "Folder containing SLIM benchmark data");
        app.add_option("-d,--DIM", DIM, "Dimension of the mesh (2 for SLIM 2D)");
        app.add_option("-g,--use_gpu", use_gpu, "Use GPU for ordering");
        try {
            app.parse(argc, argv);
        } catch (const CLI::ParseError& e) {
            exit(app.exit(e));
        }
    }
};


int main(int argc, char* argv[])
{
    CLIArgs args(argc, argv);

    spdlog::info("=== SLIM Benchmark ===");
    spdlog::info("Loading benchmark data from: {}", args.check_point_address);
    spdlog::info("Output CSV address: {}", args.output_csv_address);
    spdlog::info("Solver: {}", args.solver_type);
    spdlog::info("Ordering: {}", args.ordering_type);
    spdlog::info("DIM: {}", args.DIM);

    // ========== Load benchmark data ==========
    std::vector<std::string> hessian_addresses;
    std::string obj_address;
    RXMESH_SOLVER::prepare_benchmark_data(args.check_point_address, hessian_addresses, obj_address);
    
    spdlog::info("Number of Hessian files: {}", hessian_addresses.size());
    spdlog::info("OBJ file: {}", obj_address);

    if (hessian_addresses.empty()) {
        spdlog::error("No Hessian files found. Exiting.");
        return 1;
    }

    // ========== Load mesh from OBJ file (needed for PATCH_ORDERING) ==========
    Eigen::MatrixXd OV;
    Eigen::MatrixXi OF;
    if (!igl::read_triangle_mesh(obj_address, OV, OF)) {
        spdlog::error("Failed to read mesh from: {}", obj_address);
        return 1;
    }
    spdlog::info("Loaded mesh: {} vertices, {} faces", OV.rows(), OF.rows());

    // ========== Load the first Hessian to get matrix structure ==========
    Eigen::SparseMatrix<double> base_hessian;
    Eigen::loadMarket(base_hessian, hessian_addresses[0]);
    base_hessian = -1 * base_hessian;
    spdlog::info("Base Hessian size: {} x {}, NNZ: {}", base_hessian.rows(), base_hessian.cols(), base_hessian.nonZeros());

    // ========== Initialize solver ==========
    RXMESH_SOLVER::LinSysSolver* solver = nullptr;
    if (args.solver_type == "CUDSS") {
        solver = RXMESH_SOLVER::LinSysSolver::create(RXMESH_SOLVER::LinSysSolverType::GPU_CUDSS);
        spdlog::info("Using CUDSS direct solver.");
    } else if (args.solver_type == "MKL") {
        solver = RXMESH_SOLVER::LinSysSolver::create(RXMESH_SOLVER::LinSysSolverType::CPU_MKL);
        spdlog::info("Using Intel MKL PARDISO direct solver.");
    } else {
        spdlog::error("Unknown solver type: {}. Supported: CUDSS, MKL", args.solver_type);
        return 1;
    }
    assert(solver != nullptr);

    // ========== Compress graph with ParthAPI for DIM-based compression ==========
    PARTH::ParthAPI parth_compressor;
    parth_compressor.setMatrix(base_hessian.rows(), base_hessian.outerIndexPtr(), base_hessian.innerIndexPtr(), args.DIM);
    int* Gp = parth_compressor.Mp;
    int* Gi = parth_compressor.Mi;
    int G_N = parth_compressor.M_n;
    int G_NNZ = Gp[G_N];
    spdlog::info("Compressed graph: N={}, NNZ={}", G_N, G_NNZ);
    assert(G_N == OV.rows());

    // ========== Compute ordering (once, outside the loop) ==========
    std::vector<int> matrix_perm;
    std::vector<int> matrix_etree;
    long int ordering_time = -1;
    long int ordering_init_time = -1;
    RXMESH_SOLVER::Ordering* ordering = nullptr;

    if (args.ordering_type == "PATCH_ORDERING") {
        spdlog::info("Using PATCH_ORDERING.");
        
        // Create ordering following laplace_benchmark approach
        ordering = RXMESH_SOLVER::Ordering::create(RXMESH_SOLVER::DEMO_ORDERING_TYPE::PATCH_ORDERING);
        ordering->setOptions({
            {"use_gpu", args.use_gpu ? "1" : "0"},
            {"patch_type", args.patch_type},
            {"binary_level", std::to_string(args.binary_level)}
        });

        // Provide mesh data if the ordering needs it
        if (ordering->needsMesh()) {
            if (OV.rows() == 0 || OF.rows() == 0) {
                spdlog::error("PATCH_ORDERING requires mesh data but OBJ file not loaded.");
                delete solver;
                return 1;
            }
            ordering->setMesh(OV.data(), OV.rows(), OV.cols(),
                              OF.data(), OF.rows(), OF.cols());
        }
        ordering->setGraph(Gp, Gi, G_N, G_NNZ);
        
        // Initialize ordering (timed separately like laplace_benchmark)
        auto ordering_init_start = std::chrono::high_resolution_clock::now();
        ordering->init();
        auto ordering_init_end = std::chrono::high_resolution_clock::now();
        ordering_init_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            ordering_init_end - ordering_init_start).count();
        spdlog::info("Ordering initialization time: {} ms", ordering_init_time);

        // Compute permutation on compressed graph (timed separately)
        std::vector<int> compressed_perm, compressed_etree;
        bool compute_etree = (args.solver_type == "CUDSS");
        auto ordering_start = std::chrono::high_resolution_clock::now();
        ordering->compute_permutation(compressed_perm, compressed_etree, compute_etree);
        auto ordering_end = std::chrono::high_resolution_clock::now();
        ordering_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            ordering_end - ordering_start).count();
        spdlog::info("Ordering time: {} ms", ordering_time);

        // Expand permutation to full matrix size
        matrix_perm.resize(compressed_perm.size() * args.DIM);
        for (size_t i = 0; i < compressed_perm.size(); i++) {
            for (int j = 0; j < args.DIM; j++) {
                matrix_perm[i * args.DIM + j] = compressed_perm[i] * args.DIM + j;
            }
        }
        
        // Expand etree
        if (compute_etree) {
            matrix_etree = compressed_etree;
            for (auto& value : matrix_etree) {
                value = value * args.DIM;
            }
        }

    } else if (args.ordering_type == "PARTH") {
        spdlog::info("Using PARTH ordering.");
        
        ordering = RXMESH_SOLVER::Ordering::create(RXMESH_SOLVER::DEMO_ORDERING_TYPE::PARTH);
        ordering->setOptions({{"binary_level", std::to_string(args.binary_level)}});
        ordering->setGraph(Gp, Gi, G_N, G_NNZ);

        // Initialize ordering (timed separately like laplace_benchmark)
        auto ordering_init_start = std::chrono::high_resolution_clock::now();
        ordering->init();
        auto ordering_init_end = std::chrono::high_resolution_clock::now();
        ordering_init_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            ordering_init_end - ordering_init_start).count();
        spdlog::info("Ordering initialization time: {} ms", ordering_init_time);

        // Compute permutation on compressed graph (timed separately)
        std::vector<int> compressed_perm, compressed_etree;
        bool compute_etree = (args.solver_type == "CUDSS");
        auto ordering_start = std::chrono::high_resolution_clock::now();
        ordering->compute_permutation(compressed_perm, compressed_etree, compute_etree);
        auto ordering_end = std::chrono::high_resolution_clock::now();
        ordering_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            ordering_end - ordering_start).count();
        spdlog::info("Ordering time: {} ms", ordering_time);

        // Expand permutation to full matrix size
        matrix_perm.resize(compressed_perm.size() * args.DIM);
        for (size_t i = 0; i < compressed_perm.size(); i++) {
            for (int j = 0; j < args.DIM; j++) {
                matrix_perm[i * args.DIM + j] = compressed_perm[i] * args.DIM + j;
            }
        }

        // Expand etree
        if (compute_etree) {
            matrix_etree = compressed_etree;
            for (auto& value : matrix_etree) {
                value = value * args.DIM;
            }
        }

    } else if (args.ordering_type == "DEFAULT") {
        spdlog::info("Using DEFAULT ordering (solver's built-in).");
        // matrix_perm and matrix_etree remain empty
    } else {
        spdlog::error("Unknown ordering type: {}. Supported: DEFAULT, PATCH_ORDERING, PARTH", args.ordering_type);
        delete solver;
        return 1;
    }

    // Validate permutation if using custom ordering
    if (!matrix_perm.empty()) {
        if (!RXMESH_SOLVER::check_valid_permutation(matrix_perm.data(), matrix_perm.size())) {
            spdlog::error("Permutation is not valid!");
            delete solver;
            delete ordering;
            return 1;
        }
        spdlog::info("Permutation validated: size = {}", matrix_perm.size());
    }

    // ========== Compute factor NNZ for custom ordering ==========
    long int factor_nnz = -1;
    if (!matrix_perm.empty()) {
        factor_nnz = RXMESH_SOLVER::get_factor_nnz(base_hessian.outerIndexPtr(),
                                                   base_hessian.innerIndexPtr(),
                                                   base_hessian.valuePtr(),
                                                   base_hessian.rows(),
                                                   base_hessian.nonZeros(),
                                                   matrix_perm);
        spdlog::info("Factor NNZ ratio: {:.4f}", factor_nnz * 1.0 / base_hessian.nonZeros());
        if (ordering != nullptr) {
            solver->ordering_name = ordering->typeStr();
        }
    }

    // ========== Perform symbolic analysis once ==========
    // For MKL, we need lower triangular matrix
    Eigen::SparseMatrix<double> lower_hessian;
    if (args.solver_type == "MKL") {
        lower_hessian = base_hessian.triangularView<Eigen::Lower>();
        solver->setMatrix(lower_hessian.outerIndexPtr(),
                          lower_hessian.innerIndexPtr(),
                          lower_hessian.valuePtr(),
                          lower_hessian.rows(),
                          lower_hessian.nonZeros());
    } else {
        solver->setMatrix(base_hessian.outerIndexPtr(),
                          base_hessian.innerIndexPtr(),
                          base_hessian.valuePtr(),
                          base_hessian.rows(),
                          base_hessian.nonZeros());
    }

    // Set ordering
    auto ordering_integration_start = std::chrono::high_resolution_clock::now();
    solver->ordering(matrix_perm, matrix_etree);
    auto ordering_integration_end = std::chrono::high_resolution_clock::now();
    long int ordering_integration_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        ordering_integration_end - ordering_integration_start).count();
    spdlog::info("Ordering integration time: {} ms", ordering_integration_time);

    // Symbolic analysis (once)
    auto analysis_start = std::chrono::high_resolution_clock::now();
    solver->analyze_pattern(matrix_perm, matrix_etree);
    if (args.solver_type == "CUDSS") {
        CUDA_SYNC_CHECK();
    }
    auto analysis_end = std::chrono::high_resolution_clock::now();
    long int analysis_time = std::chrono::duration_cast<std::chrono::milliseconds>(analysis_end - analysis_start).count();
    spdlog::info("Symbolic analysis time: {} ms", analysis_time);

    // ========== Extract mesh name from obj file ==========
    std::string mesh_name = "unknown";
    if (!obj_address.empty()) {
        mesh_name = std::filesystem::path(obj_address).stem().string();
    }
    spdlog::info("Mesh name: {}", mesh_name);

    // ========== Prepare CSV output ==========
    std::vector<std::string> header;
    header.emplace_back("mesh_name");
    header.emplace_back("iteration");
    header.emplace_back("ordering_type");
    header.emplace_back("solver_type");
    header.emplace_back("G_N");
    header.emplace_back("G_NNZ");
    header.emplace_back("nd_levels");
    header.emplace_back("patch_type");
    header.emplace_back("patch_size");
    header.emplace_back("patch_time");
    header.emplace_back("factor/matrix NNZ ratio");
    header.emplace_back("ordering_init_time");
    header.emplace_back("ordering_time");
    header.emplace_back("ordering_integration_time");
    header.emplace_back("analysis_time");
    header.emplace_back("factorization_time");
    header.emplace_back("solve_time");
    header.emplace_back("residual");

    RXMESH_SOLVER::CSVManager runtime_csv(args.output_csv_address, "SLIM_benchmark", header, false);

    // ========== Solve loop: iterate through all Hessian/gradient pairs ==========
    spdlog::info("Starting solve loop with {} iterations...", hessian_addresses.size());

    for (size_t iter = 0; iter < hessian_addresses.size(); iter++) {
        spdlog::info("--- Iteration {} ---", iter);

        // Load Hessian matrix
        Eigen::SparseMatrix<double> hessian;
        Eigen::loadMarket(hessian, hessian_addresses[iter]);
        // Generate random RHS vector with the same size as the matrix
        Eigen::VectorXd rhs = Eigen::VectorXd::Random(hessian.rows());
        // For result
        Eigen::VectorXd result;

        // Set matrix values (same sparsity pattern)
        if (args.solver_type == "MKL") {
            lower_hessian = hessian.triangularView<Eigen::Lower>();
            solver->setMatrix(lower_hessian.outerIndexPtr(),
                              lower_hessian.innerIndexPtr(),
                              lower_hessian.valuePtr(),
                              lower_hessian.rows(),
                              lower_hessian.nonZeros());
        } else {
            solver->setMatrix(hessian.outerIndexPtr(),
                              hessian.innerIndexPtr(),
                              hessian.valuePtr(),
                              hessian.rows(),
                              hessian.nonZeros());
        }

        // Factorize
        auto factor_start = std::chrono::high_resolution_clock::now();
        solver->factorize();
        if (args.solver_type == "CUDSS") {
            CUDA_SYNC_CHECK();
        }
        auto factor_end = std::chrono::high_resolution_clock::now();
        long int factorization_time = std::chrono::duration_cast<std::chrono::milliseconds>(factor_end - factor_start).count();
        spdlog::info("Factorization time: {} ms", factorization_time);

        // Solve
        auto solve_start = std::chrono::high_resolution_clock::now();
        solver->solve(rhs, result);
        if (args.solver_type == "CUDSS") {
            CUDA_SYNC_CHECK();
        }
        auto solve_end = std::chrono::high_resolution_clock::now();
        long int solve_time = std::chrono::duration_cast<std::chrono::milliseconds>(solve_end - solve_start).count();
        spdlog::info("Solve time: {} ms", solve_time);

        // Compute residual
        // Make hessian full (it might be stored as lower triangular)
        Eigen::SparseMatrix<double> hessian_full = hessian.selfadjointView<Eigen::Lower>();
        double residual = (rhs - hessian_full * result).norm() / rhs.norm();
        spdlog::info("Residual: {}", residual);

        // Record to CSV
        runtime_csv.addElementToRecord(mesh_name, "mesh_name");
        runtime_csv.addElementToRecord(static_cast<int>(iter), "iteration");
        runtime_csv.addElementToRecord(args.ordering_type, "ordering_type");
        runtime_csv.addElementToRecord(args.solver_type, "solver_type");
        runtime_csv.addElementToRecord(hessian.rows(), "G_N");
        runtime_csv.addElementToRecord(hessian.nonZeros(), "G_NNZ");
        
        // ND levels from etree
        int nd_levels = matrix_etree.empty() ? 0 : static_cast<int>(std::log2(matrix_etree.size() + 1));
        runtime_csv.addElementToRecord(nd_levels, "nd_levels");
        
        // Patch statistics (similar to laplace_benchmark)
        if (args.ordering_type == "PATCH_ORDERING" && ordering != nullptr) {
            std::map<std::string, double> stat;
            ordering->getStatistics(stat);
            runtime_csv.addElementToRecord(args.patch_type, "patch_type");
            runtime_csv.addElementToRecord(stat["patch_size"], "patch_size");
            runtime_csv.addElementToRecord(stat["patching_time"], "patch_time");
        } else {
            runtime_csv.addElementToRecord("", "patch_type");
            runtime_csv.addElementToRecord(0, "patch_size");
            runtime_csv.addElementToRecord(0, "patch_time");
        }
        
        if (factor_nnz > 0) {
            runtime_csv.addElementToRecord(factor_nnz * 1.0 / hessian.nonZeros(), "factor/matrix NNZ ratio");
        } else {
            runtime_csv.addElementToRecord(solver->getFactorNNZ() * 1.0 / hessian.nonZeros(), "factor/matrix NNZ ratio");
        }
        runtime_csv.addElementToRecord(ordering_init_time, "ordering_init_time");
        runtime_csv.addElementToRecord(ordering_time, "ordering_time");
        runtime_csv.addElementToRecord(ordering_integration_time, "ordering_integration_time");
        runtime_csv.addElementToRecord(analysis_time, "analysis_time");
        runtime_csv.addElementToRecord(factorization_time, "factorization_time");
        runtime_csv.addElementToRecord(solve_time, "solve_time");
        runtime_csv.addElementToRecord(residual, "residual");
        runtime_csv.addRecord();
    }

    spdlog::info("=== SLIM Benchmark Complete ===");
    spdlog::info("Results saved to: {}", args.output_csv_address);

    // Cleanup
    delete solver;
    if (ordering != nullptr) {
        delete ordering;
    }

    return 0;
}
