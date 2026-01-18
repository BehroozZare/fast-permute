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
#include <cmath>
#include <unordered_set>
#include <queue>

#include "LinSysSolver.hpp"
#include "get_factor_nnz.h"
#include "check_valid_permutation.h"
#include "utils/ipc_prep_benchmark.h"
#include "csv_utils.h"
#include "save_vector.h"
#include <parth/parth.h>
#include <ordering.h>
#include "create_patch_with_metis.h"

#ifndef _CUDA_ERROR_
#define _CUDA_ERROR_
inline void HandleError(cudaError_t err, const char* file, int line)
{
    // Error handling micro, wrap it around function whenever possible
    if (err != cudaSuccess) {
        printf("\n%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define CUDA_ERROR(err) (HandleError(err, __FILE__, __LINE__))
#endif


#define CUDA_SYNC_CHECK() do {                               \
CUDA_ERROR(cudaGetLastError());                            \
CUDA_ERROR(cudaDeviceSynchronize());                       \
} while(0)


struct CLIArgs
{
    int binary_level = 5; // It is zero-based so 9 is 10 levels
    std::string output_csv_address = "/home/behrooz/Desktop/Last_Project/gpu_ordering/output/IPC/ipc";//Include absolute path with csv file name without .csv extension
    std::string solver_type   = "CUDSS";
    std::string ordering_type = "DEFAULT";
    std::string patch_type = "rxmesh";
    // std::string check_point_address = "/media/behrooz/FarazHard/IPC_matrices/MatOnBoard/mat225x225t40-mat225x225t40_fall_NH_BE_interiorPoint_20260115020027";
    std::string check_point_address = "/media/behrooz/FarazHard/IPC_matrices/test";
    std::string V_address = "";
    std::string F_address = "";
    int patch_size = 256;
    double patch_update_nnz_threshold = 0.05;  // Update patches if NNZ changes by more than this percentage (e.g., 0.05 = 5%)

    CLIArgs(int argc, char* argv[])
    {
        CLI::App app{"Separator analysis"};
        app.add_option("-a,--ordering", ordering_type, "ordering type");
        app.add_option("-s,--solver", solver_type, "solver type");
        app.add_option("-o,--output", output_csv_address, "output folder name");
        app.add_option("-p,--patch_type", patch_type, "how to patch the graph/mesh");
        app.add_option("-z,--patch_size", patch_size, "patch size");
        app.add_option("-b,--binary_level", binary_level, "binary level for binary tree ordering");
        app.add_option("-k,--check_point_address", check_point_address, "check point address");
        app.add_option("-v,--V_address", V_address, "V address");
        app.add_option("-f,--F_address", F_address, "F address");
        app.add_option("-u,--patch_update_nnz_threshold", patch_update_nnz_threshold, "NNZ change percentage threshold for updating patches (e.g., 0.05 = 5%)");

        try {
            app.parse(argc, argv);
        } catch (const CLI::ParseError& e) {
            exit(app.exit(e));
        }
    }
};



int main(int argc, char* argv[])
{
    // Load the mesh
    CLIArgs args(argc, argv);

    std::cout << "Loading matrix from: " << args.check_point_address << std::endl;
    std::cout << "Output csv address: " << args.output_csv_address << std::endl;

    std::vector<std::string> matrix_addresses = RXMESH_SOLVER::prepare_benchmark_data(args.check_point_address);
    spdlog::info("Number of matrices: {}", matrix_addresses.size());
    Eigen::SparseMatrix<double> base; // Or Eigen::MatrixXd for dense matrices
    std::string filename = matrix_addresses[0];
    Eigen::loadMarket(base, filename);

    int DIM = 3;
    std::vector<int> node_to_patch;
    std::vector<int> parth_perm, parth_new_labels, parth_sep_ptr;

    Eigen::VectorXd rhs = Eigen::VectorXd::Random(base.rows());
    Eigen::VectorXd result;

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
    } else {
        spdlog::error("Unknown solver type.");
    }
    assert(solver != nullptr);

    int* Gp;
    int* Gi;
    int G_N;
    std::vector<int> Gp_prev;
    std::vector<int> Gi_prev;

    std::vector<int> lag_perm;
    std::vector<int> metis_perm;

    int last_patch_update_nnz = -1;  // Track G_NNZ when patches were last updated

    for(auto & matrix_address : matrix_addresses) {
        //==============Read the matrix==============
        //From the name of the matrix get the frame and iteration number
        int frame = -1;
        int iteration = -1;
        std::string matrix_name = std::filesystem::path(matrix_address).filename().string();
        std::regex pattern(R"(hessian_(\d+)_(\d+)_last_IPC\.mtx)");
        std::smatch match;
        if(std::regex_match(matrix_name, match, pattern)){
            frame = std::stoi(match[1].str());
            iteration = std::stoi(match[2].str());
        }
        assert(frame != -1 && iteration != -1);
        spdlog::info("Frame: {}, Iteration: {}", frame, iteration);
        spdlog::info("Loading matrix {}", matrix_address);

        Eigen::SparseMatrix<double> mat;
        Eigen::loadMarket(mat, matrix_address);

        
        //==============Check if the graph is the same as the previous one==============
        PARTH::ParthAPI parth_compressor;
        parth_compressor.setMatrix(mat.rows(), mat.outerIndexPtr(), mat.innerIndexPtr(), DIM);
        Gp = parth_compressor.Mp;
        Gi = parth_compressor.Mi;
        G_N = parth_compressor.M_n;
        bool is_graph_equal = false;
        if (Gp_prev.empty()) {
            Gp_prev.resize(G_N + 1);
            Gi_prev.resize(Gp[G_N]);
            std::copy(Gp, Gp + G_N + 1, Gp_prev.begin());
            std::copy(Gi, Gi + Gp[G_N], Gi_prev.begin());
            is_graph_equal = false;
        } else {
            if (RXMESH_SOLVER::is_graph_equal(Gp_prev.data(), Gi_prev.data(), Gp, Gi, G_N, G_N)){
                spdlog::info("Graph is the same as the previous one. Skipping symbolic analysis.");
                is_graph_equal = true;
            }
        }

        // Update patches when graph changes and NNZ change exceeds threshold (only for PATCH_ORDERING)
        int current_nnz = Gp[G_N];  // NNZ of the compressed graph
        if (args.ordering_type == "PATCH_ORDERING") {
            double nnz_change_ratio = (last_patch_update_nnz > 0) ? 
                std::abs(current_nnz - last_patch_update_nnz) / static_cast<double>(last_patch_update_nnz) : 1.0;
            bool should_update_patches = !is_graph_equal && 
                                         (last_patch_update_nnz < 0 || 
                                          nnz_change_ratio >= args.patch_update_nnz_threshold);
            if (should_update_patches) {
                RXMESH_SOLVER::create_patch_with_metis(mat.rows(), mat.outerIndexPtr(), mat.innerIndexPtr(), 
                                                       DIM, args.patch_size, node_to_patch);
                spdlog::info("Updated node_to_patch at frame {} (NNZ changed by {:.2f}%, from {} to {})", 
                            frame, nnz_change_ratio * 100.0, last_patch_update_nnz, current_nnz);
                last_patch_update_nnz = current_nnz;
                
                if (node_to_patch.size() != (mat.rows() / DIM)) {
                    spdlog::info("node to patch size is :{}, and the matrix size is: {}", node_to_patch.size(), mat.rows());
                    throw std::runtime_error("Node to patch size is not equal to the number of rows of the matrix");
                } else {
                    spdlog::info("Node to patch size: {}", node_to_patch.size());
                }
            }
        }

        // Create ordering based on ordering_type
        RXMESH_SOLVER::Ordering* ordering = nullptr;
        if (args.ordering_type == "PATCH_ORDERING") {
            ordering = RXMESH_SOLVER::Ordering::create(RXMESH_SOLVER::DEMO_ORDERING_TYPE::PATCH_ORDERING);
            ordering->setGraph(Gp, Gi, G_N, Gp[G_N]);
            ordering->setPatch(node_to_patch);
        } else if (args.ordering_type == "PARTH") {
            ordering = RXMESH_SOLVER::Ordering::create(RXMESH_SOLVER::DEMO_ORDERING_TYPE::PARTH);
            ordering->setGraph(Gp, Gi, G_N, Gp[G_N]);
            ordering->setOptions({{"binary_level", std::to_string(args.binary_level)}});
        }

        double residual = 0;
        long int ordering_init_time = -1;
        long int ordering_time = -1;
        long int ordering_integration_time = -1;
        long int analysis_time = -1;
        long int factorization_time = -1;
        long int solve_time = -1;
        long int factor_nnz = -1;

        std::vector<int> matrix_perm, matrix_etree;
        auto ordering_start = std::chrono::high_resolution_clock::now();
        if (args.ordering_type != "DEFAULT" && !is_graph_equal && ordering != nullptr) {
            std::vector<int> perm, etree;
            ordering->compute_permutation(perm, etree, true);
            //Map to global permutation (expand DIM times)
            matrix_perm.resize(perm.size() * DIM);
            for(int i1 = 0; i1 < perm.size(); i1++){
                for(int j = 0; j < DIM; j++){
                    matrix_perm[i1 * DIM + j] = perm[i1] * DIM + j;
                }
            }
            matrix_etree = etree;
            for(auto& value: matrix_etree){
                value = value * DIM;
            }
            assert(matrix_perm.size() == mat.rows());
        }
        auto ordering_end = std::chrono::high_resolution_clock::now();
        ordering_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            ordering_end - ordering_start)
            .count();
                
        //Check for correct perm
        if (!RXMESH_SOLVER::check_valid_permutation(matrix_perm.data(), matrix_perm.size())) {
            spdlog::error("Permutation is not valid!");
        }
        spdlog::info("Ordering time: {} ms",
                    ordering_time);

        double patch_factor_nnz, metis_factor_nnz, lag_factor_nnz;
        if (args.ordering_type != "DEFAULT" && !is_graph_equal) {
            patch_factor_nnz = RXMESH_SOLVER::get_factor_nnz(mat.outerIndexPtr(),
                                mat.innerIndexPtr(),
                                mat.valuePtr(),
                                mat.rows(),
                                mat.nonZeros(),
                                matrix_perm);

            if (!matrix_perm .empty()) {
                RXMESH_SOLVER::save_vector_to_file(matrix_perm, args.check_point_address + "/perm_" + std::to_string(frame) + "_" + std::to_string(iteration) + "_Ordering_time_" + std::to_string(ordering_time) + ".txt");
            }
            if (!matrix_etree.empty()) {
                RXMESH_SOLVER::save_vector_to_file(matrix_etree, args.check_point_address + "/etree_" + std::to_string(frame) + "_" + std::to_string(iteration) + "_Ordering_time_" + std::to_string(ordering_time) + ".txt");
            }

            RXMESH_SOLVER::Ordering* metis_ordering = RXMESH_SOLVER::Ordering::create(
                RXMESH_SOLVER::DEMO_ORDERING_TYPE::METIS);
            std::vector<int> metis_etree, metis_perm;
            metis_ordering->setGraph(Gp, Gi, G_N, Gp[G_N]);
            metis_ordering->compute_permutation(metis_perm, metis_etree, false);
            std::vector<int> matrix_metis_perm(metis_perm.size() * DIM);
            for(int i = 0; i < metis_perm.size(); i++){
                for(int j = 0; j < DIM; j++){
                    matrix_metis_perm[i * DIM + j] = metis_perm[i] * DIM + j;
                }
            }
            if(lag_perm.empty()) {
                lag_perm = matrix_metis_perm;
            }

            metis_factor_nnz = RXMESH_SOLVER::get_factor_nnz(mat.outerIndexPtr(),
                                                        mat.innerIndexPtr(),
                                                        mat.valuePtr(),
                                                        mat.rows(),
                                                        mat.nonZeros(),
                                                        matrix_metis_perm);

            lag_factor_nnz = RXMESH_SOLVER::get_factor_nnz(mat.outerIndexPtr(),
                                                        mat.innerIndexPtr(),
                                                        mat.valuePtr(),
                                                        mat.rows(),
                                                        mat.nonZeros(),
                                                        lag_perm);
        }

        // For CUDSS, we need:
        // 1. Expand symmetric matrix to full format (MatrixMarket only stores lower triangle)
        // 2. Convert to CSR format (Eigen default is CSC, cuDSS requires CSR)
        Eigen::SparseMatrix<double> mat_full_csc = mat.selfadjointView<Eigen::Lower>();
        Eigen::SparseMatrix<double, Eigen::RowMajor> mat_full(mat_full_csc);  // Convert CSC to CSR
        solver->setMatrix(mat_full.outerIndexPtr(),
                          mat_full.innerIndexPtr(),
                          mat_full.valuePtr(),
                          mat_full.rows(),
                          mat_full.nonZeros());

        // Reordering phase (required before symbolic analysis)
        auto ordering_int_start = std::chrono::high_resolution_clock::now();
        if (!is_graph_equal) {
            solver->ordering(matrix_perm, matrix_etree);
        }
        auto ordering_int_end = std::chrono::high_resolution_clock::now();
        if (!is_graph_equal) {
            ordering_integration_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                ordering_int_end - ordering_int_start).count();
            spdlog::info("Ordering integration time: {} ms", ordering_integration_time);
        } else {
            ordering_integration_time = 0;
        }

        // Symbolic analysis time
        auto start = std::chrono::high_resolution_clock::now();
        if (!is_graph_equal) {
            solver->analyze_pattern(matrix_perm, matrix_etree);
            CUDA_SYNC_CHECK();
        }
        auto end = std::chrono::high_resolution_clock::now();
        if (!is_graph_equal) {
            analysis_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            spdlog::info(
                "Analysis time: {} ms",
                analysis_time);
        } else {
            analysis_time = 0;
            spdlog::info("Analysis time: 0 ms");
        }
        //Factorization time
        start = std::chrono::high_resolution_clock::now();
        solver->factorize();
        CUDA_SYNC_CHECK();
        end = std::chrono::high_resolution_clock::now();
        factorization_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        spdlog::info(
            "Factorization time: {} ms",
            factorization_time);

        //Solve time
        start = std::chrono::high_resolution_clock::now();
        solver->solve(rhs, result);
        CUDA_SYNC_CHECK();
        end = std::chrono::high_resolution_clock::now();
        solve_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        spdlog::info(
            "Solve time: {} ms",
            solve_time);

        // Compute residual
        assert(mat.rows() == mat.cols());
        // mat_full was already computed above for the solver
        residual = (rhs - mat_full * result).norm();
        spdlog::info("Residual: {}", residual);
        spdlog::info("Final factor/matrix NNZ ratio: {}",
                     solver->getFactorNNZ() * 1.0 / mat.nonZeros());


        //Save data to a csv file
        std::string csv_name = args.output_csv_address;
        std::vector<std::string> header;
        header.emplace_back("frame");
        header.emplace_back("iteration");
        header.emplace_back("ordering_type");
        header.emplace_back("G_N");
        header.emplace_back("G_NNZ");
        header.emplace_back("factor/matrix NNZ ratio");
        header.emplace_back("patch_factor_ratio");
        header.emplace_back("lag_factor_ratio");
        header.emplace_back("ordering_time");
        header.emplace_back("ordering_integration_time");
        header.emplace_back("analysis_time");
        header.emplace_back("factorization_time");
        header.emplace_back("solve_time");
        header.emplace_back("residual");


        RXMESH_SOLVER::CSVManager runtime_csv(csv_name, "some address", header, false);
        runtime_csv.addElementToRecord(frame, "frame");
        runtime_csv.addElementToRecord(iteration, "iteration");
        runtime_csv.addElementToRecord(args.ordering_type, "ordering_type");
        runtime_csv.addElementToRecord(mat.rows(), "G_N");
        runtime_csv.addElementToRecord(mat.nonZeros(), "G_NNZ");
        if(args.ordering_type != "DEFAULT") {
            runtime_csv.addElementToRecord(patch_factor_nnz * 1.0 / mat.nonZeros(), "factor/matrix NNZ ratio");
            runtime_csv.addElementToRecord(patch_factor_nnz / metis_factor_nnz, "patch_factor_ratio");
            runtime_csv.addElementToRecord(lag_factor_nnz / metis_factor_nnz, "lag_factor_ratio");
        } else {
            runtime_csv.addElementToRecord(0, "factor/matrix NNZ ratio");
            runtime_csv.addElementToRecord(0, "patch_factor_ratio");
            runtime_csv.addElementToRecord(0, "lag_factor_ratio");
        }
        runtime_csv.addElementToRecord(ordering_time, "ordering_time");
        runtime_csv.addElementToRecord(ordering_integration_time, "ordering_integration_time");
        runtime_csv.addElementToRecord(analysis_time, "analysis_time");
        runtime_csv.addElementToRecord(factorization_time, "factorization_time");
        runtime_csv.addElementToRecord(solve_time, "solve_time");
        runtime_csv.addElementToRecord(residual, "residual");
        runtime_csv.addRecord();

        // Reset timing variables for next iteration (ordering/analysis times are only meaningful when graph changes)
        ordering_init_time = 0;
        ordering_time = 0;
        ordering_integration_time = 0;
        analysis_time = 0;

        //Save the Gp_curr into Gp_prev and Gi_curr into Gi_prev
        Gp_prev.resize(G_N + 1);
        Gi_prev.resize(Gp[G_N]);
        std::copy(Gp, Gp + G_N + 1, Gp_prev.begin());
        std::copy(Gi, Gi + G_N, Gi_prev.begin());
        if (ordering != nullptr) {
            delete ordering;
        }
    }
    delete solver;
    // delete metis_ordering;

    return 0;
}