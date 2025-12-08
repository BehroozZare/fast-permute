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
#include <unordered_set>
#include <queue>

#include "LinSysSolver.hpp"
#include "get_factor_nnz.h"
#include "check_valid_permutation.h"
#include "cpu_ordering_with_patch.h"
#include "remove_diagonal.h"
#include "compress_hessian.h"
#include "ipc_prep_benchmark.h"
#include "csv_utils.h"
#include "save_vector.h"
#include <parth//parth.h>
#include <ordering.h>


struct CLIArgs
{
    int binary_level = 9; // It is zero-based so 9 is 10 levels
    std::string output_csv_address = "/home/behrooz/Desktop/Last_Project/gpu_ordering/output/IPC";//Include absolute path with csv file name without .csv extension
    std::string solver_type   = "CHOLMOD";
    std::string ordering_type = "DEFAULT";
    std::string patch_type = "rxmesh";
    std::string check_point_address = "/media/behrooz/FarazHard/IPC_matrices/MatOnBoard/mat100x100t40-mat100x100t40_fall_NH_BE_interiorPoint_20251207132538";
    int patch_size = 24;
    bool use_gpu = false;

    CLIArgs(int argc, char* argv[])
    {
        CLI::App app{"Separator analysis"};
        app.add_option("-a,--ordering", ordering_type, "ordering type");
        app.add_option("-s,--solver", solver_type, "solver type");
        app.add_option("-o,--output", output_csv_address, "output folder name");
        app.add_option("-g,--use_gpu", use_gpu, "use gpu");
        app.add_option("-p,--patch_type", patch_type, "how to patch the graph/mesh");
        app.add_option("-z,--patch_size", patch_size, "patch size");
        app.add_option("-b,--binary_level", binary_level, "binary level for binary tree ordering");
        app.add_option("-k,--check_point_address", check_point_address, "check point address");

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

    //Use Parth for patch creation
    int DIM = 3;
    PARTH::ParthAPI parth;
    std::vector<int> perm;
    parth.setMatrix(base.rows(), base.outerIndexPtr(), base.innerIndexPtr(), DIM);
    parth.computePermutation(perm);


    std::vector<int> node_to_patch = parth.hmd.DOF_to_HMD_node;
    std::vector<int> patch_sizes(parth.hmd.HMD_tree.size(), 0);
    //Get the initial patch sizes
    for(int i = 0; i < parth.hmd.HMD_tree.size(); i++){
        patch_sizes[i] = parth.hmd.HMD_tree[i].DOFs.size();
    }
    int* Gp_curr = parth.Mp;
    int* Gi_curr = parth.Mi;
    int G_N = parth.M_n;
    assert(G_N == base.rows() / DIM);

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
        int start_idx = Gp_curr[current_node];
        int end_idx = Gp_curr[current_node + 1];
        std::unordered_set<int> nbr_patches;
        for(int j = start_idx; j < end_idx; j++){
            int nbr_idx = Gi_curr[j];
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
    


    RXMESH_SOLVER::Ordering* ordering;
    ordering = RXMESH_SOLVER::Ordering::create(RXMESH_SOLVER::DEMO_ORDERING_TYPE::METIS);
    std::vector<int> Gp_prev;
    std::vector<int> Gi_prev;
    std::vector<int> org_metis_perm;

    for(int i = 0; i < matrix_addresses.size(); i++) {
        auto& matrix_address = matrix_addresses[i];
        PARTH::ParthAPI parth_compressor;
        Eigen::SparseMatrix<double> mat;
        Eigen::loadMarket(mat, matrix_address);
        parth_compressor.setMatrix(mat.rows(), mat.outerIndexPtr(), mat.innerIndexPtr(), DIM);
        Gp_curr = parth_compressor.Mp;
        Gi_curr = parth_compressor.Mi;
        G_N = parth_compressor.M_n;
        int G_NNZ = Gp_curr[G_N];
        assert(G_N == mat.rows() / DIM);
        spdlog::info("Loading matrix {}", matrix_address);
        if(RXMESH_SOLVER::is_graph_equal(Gp_prev.data(), Gi_prev.data(), Gp_curr, Gi_curr, G_N, Gp_prev.size() - 1)){
            spdlog::info("Graph is the same as the previous one. Skipping...");
            continue;
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

        RXMESH_SOLVER::CPUOrdering_PATCH cpu_ordering_with_patch;
        cpu_ordering_with_patch.setGraph(Gp_curr, Gi_curr, G_N, G_NNZ);
        cpu_ordering_with_patch.init_patches(number_of_patches, node_to_patch, args.binary_level);

        //Patch to node mapping

        double residual = 0;
        long int ordering_init_time = -1;
        long int ordering_time = -1;
        long int analysis_time = -1;
        long int factorization_time = -1;
        long int solve_time = -1;
        long int factor_nnz = -1;

        auto ordering_start = std::chrono::high_resolution_clock::now();
        cpu_ordering_with_patch.compute_permutation(perm);
        //Map to global permutation
        std::vector<int> matrix_perm(perm.size() * DIM);
        for(int i = 0; i < perm.size(); i++){
            for(int j = 0; j < DIM; j++){
                matrix_perm[i * DIM + j] = perm[i] * DIM + j;
            }
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
        assert(matrix_perm.size() == mat.rows());

        factor_nnz = RXMESH_SOLVER::get_factor_nnz(mat.outerIndexPtr(),
                                                    mat.innerIndexPtr(),
                                                    mat.valuePtr(),
                                                    mat.rows(),
                                                    mat.nonZeros(),
                                                    matrix_perm);

        ordering->setGraph(Gp_curr, Gi_curr, G_N, G_NNZ);
        std::vector<int> metis_etree, metis_perm;
        ordering->compute_permutation(metis_perm, metis_etree, false);
        std::vector<int> matrix_metis_perm(metis_perm.size() * DIM);
        for(int i = 0; i < metis_perm.size(); i++){
            for(int j = 0; j < DIM; j++){
                matrix_metis_perm[i * DIM + j] = metis_perm[i] * DIM + j;
            }
        }
        double metis_factor_nnz = RXMESH_SOLVER::get_factor_nnz(mat.outerIndexPtr(),
                                                    mat.innerIndexPtr(),
                                                    mat.valuePtr(),
                                                    mat.rows(),
                                                    mat.nonZeros(),
                                                    matrix_metis_perm);
        spdlog::info(
            "The ratio of factor non-zeros to matrix non-zeros given custom reordering: {}",
            (factor_nnz * 1.0 /mat.nonZeros()));
        spdlog::info("The ratio of patching factor to metis factor is {}",
            (factor_nnz * 1.0 /metis_factor_nnz));
        spdlog::info("Customize Ordering is done.");
        if (i == 0) {
            org_metis_perm = matrix_metis_perm;
        }
        double org_metis_factor_nnz = RXMESH_SOLVER::get_factor_nnz(mat.outerIndexPtr(),
                                                    mat.innerIndexPtr(),
                                                    mat.valuePtr(),
                                                    mat.rows(),
                                                    mat.nonZeros(),
                                                    org_metis_perm);
        spdlog::info("The ratio of patching factor to metis factor is {}", (org_metis_factor_nnz * 1.0 /metis_factor_nnz));


        // solver->setMatrix(OL.outerIndexPtr(),
        //                   OL.innerIndexPtr(),
        //                   OL.valuePtr(),
        //                   OL.rows(),
        //                   OL.nonZeros());

        // // Symbolic analysis time
        // auto start = std::chrono::high_resolution_clock::now();
        // solver->analyze_pattern(perm, etree);
        // auto end = std::chrono::high_resolution_clock::now();
        // analysis_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        // spdlog::info(
        //     "Analysis time: {} ms",
        //     analysis_time);

        // Factorization time
        // start = std::chrono::high_resolution_clock::now();
        // solver->factorize();
        // end = std::chrono::high_resolution_clock::now();
        // factorization_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        // spdlog::info(
        //     "Factorization time: {} ms",
        //     factorization_time);

        // Solve time
        // start = std::chrono::high_resolution_clock::now();
        // solver->solve(rhs, result);
        // end = std::chrono::high_resolution_clock::now();
        // solve_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        // spdlog::info(
        //     "Solve time: {} ms",
        //     solve_time);

        // // Compute residual
        // assert(OL.rows() == OL.cols());
        // residual = (rhs - OL * result).norm();
        // spdlog::info("Residual: {}", residual);
        // spdlog::info("Final factor/matrix NNZ ratio: {}",
        //              solver->getFactorNNZ() * 1.0 / OL.nonZeros());


        //Save data to a csv file
        std::string csv_name = args.output_csv_address;
        std::vector<std::string> header;
        header.emplace_back("G_N");
        header.emplace_back("G_NNZ");
        header.emplace_back("factor/matrix NNZ ratio");
        header.emplace_back("factor/metis factor");
        header.emplace_back("org factor/metis factor");
        header.emplace_back("ordering_time");
        header.emplace_back("analysis_time");
        header.emplace_back("factorization_time");
        header.emplace_back("solve_time");
        header.emplace_back("residual");


        RXMESH_SOLVER::CSVManager runtime_csv(csv_name, "some address", header, false);
        runtime_csv.addElementToRecord(mat.rows(), "G_N");
        runtime_csv.addElementToRecord(mat.nonZeros(), "G_NNZ");
        runtime_csv.addElementToRecord(factor_nnz * 1.0 / mat.nonZeros(), "factor/matrix NNZ ratio");
        runtime_csv.addElementToRecord(factor_nnz / metis_factor_nnz, "factor/metis factor");
        runtime_csv.addElementToRecord(org_metis_factor_nnz / metis_factor_nnz, "org factor/metis factor");
        runtime_csv.addElementToRecord(ordering_time, "ordering_time");
        runtime_csv.addElementToRecord(analysis_time, "analysis_time");
        runtime_csv.addElementToRecord(factorization_time, "factorization_time");
        runtime_csv.addElementToRecord(solve_time, "solve_time");
        runtime_csv.addElementToRecord(residual, "residual");
        runtime_csv.addRecord();


        //Save the Gp_curr into Gp_prev and Gi_curr into Gi_prev
        Gp_prev.resize(G_N + 1);
        Gi_prev.resize(G_NNZ);
        std::copy(Gp_curr, Gp_curr + G_N + 1, Gp_prev.begin());
        std::copy(Gi_curr, Gi_curr + G_NNZ, Gi_prev.begin());
        delete solver;
    }
    delete ordering;
    return 0;
}