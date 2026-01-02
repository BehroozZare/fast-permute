//
// Created by behrooz on 2025-09-28.
//



#include <igl/read_triangle_mesh.h>
#include <spdlog/spdlog.h>
#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include <chrono>
#include <unsupported/Eigen/SparseExtra>
#include <iostream>
#include <filesystem>
#include <parth/parth.h>
#include <unordered_set>

#include "SPD_cot_matrix.h"
#include "LinSysSolver.hpp"
#include "get_factor_nnz.h"
#include "check_valid_permutation.h"
#include "ordering.h"
#include "remove_diagonal.h"
#include "csv_utils.h"
#include "save_vector.h"



struct CLIArgs
{
    std::string input_node_to_patch = "/home/behrooz/Desktop/Last_Project/gpu_ordering/benchmark/single_factorization/test_data/rocker-arm_67_node_to_patch";//It should be in the form of "<mesh_name>_<time(ms)>_node_to_patch.txt"
    std::string input_matrix = "/home/behrooz/Desktop/Last_Project/gpu_ordering/benchmark/single_factorization/test_data/rocker-arm.mtx";//It should be in the form of "<mesh_name>.mtx"
    int binary_level = 7; // It is zero-based so 9 is 10 levels
    std::string output_csv_address="/home/behrooz/Desktop/Last_Project/gpu_ordering/output/single_factorization/general";//Include absolute path with csv file name without .csv extension
    std::string solver_type   = "CUDSS";
    std::string ordering_type = "DEFAULT";
    std::string patch_type = "rxmesh";
    int DIM = 3;
    
    CLIArgs(int argc, char* argv[])
    {
        CLI::App app{"Separator analysis"};
        app.add_option("-a,--ordering", ordering_type, "ordering type");
        app.add_option("-s,--solver", solver_type, "solver type");
        app.add_option("-o,--output", output_csv_address, "output folder name");
        app.add_option("-i,--input_patch", input_node_to_patch, "input patch file name");
        app.add_option("-m,--input_matrix", input_matrix, "input matrix name");
        app.add_option("-p,--patch_type", patch_type, "how to patch the graph/mesh");
        app.add_option("-d,--DIM", DIM, "dimension of the mesh");
        app.add_option("-b,--binary_level", binary_level, "binary level for binary tree ordering");

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

    if (args.input_node_to_patch.empty()) {
        std::cerr << "Error: Input node_to_patch file not specified. Use -i or --input "
                     "to specify the patch file."
                  << std::endl;
        return 1;
    }

    if (args.input_matrix.empty()) {
        std::cerr << "Error: Input matrix file not specified. Use -m or --input_matrix "
                     "to specify the matrix file."
                  << std::endl;
        return 1;
    }

    spdlog::info("reading patch information from: {}", args.input_node_to_patch);
    spdlog::info("Loading matrix from: {}", args.input_matrix);
    spdlog::info("Output csv folder is: {}", args.output_csv_address);
    std::string mesh_name = std::filesystem::path(args.input_matrix).stem().string();
    spdlog::info("Mesh name is: {}", mesh_name);

    Eigen::SparseMatrix<double> OL;
    Eigen::loadMarket(OL, args.input_matrix);
    assert(OL.rows() == OL.cols());
    spdlog::info("Number of rows: {}", OL.rows());
    spdlog::info("Number of non-zeros: {}", OL.nonZeros());
    spdlog::info(
        "Sparsity: {:.2f}%",
        (1 - (OL.nonZeros() / static_cast<double>(OL.rows() * OL.rows()))) *
            100);

    std::vector<int> node_to_patch;
    std::ifstream node_to_patch_file(args.input_node_to_patch);
    std::string line;
    while (std::getline(node_to_patch_file, line)) {
        node_to_patch.push_back(std::stoi(line));
    }
    node_to_patch_file.close();
    spdlog::info("Number of nodes to patch: {}", node_to_patch.size());
    std::unordered_set<int> patch_ids;
    for (int i = 0; i < node_to_patch.size(); i++) {
        patch_ids.insert(node_to_patch[i]);
    }
    spdlog::info("Number of patch ids are: {}", patch_ids.size());

    Eigen::VectorXd rhs = Eigen::VectorXd::Random(OL.rows());
    Eigen::VectorXd result;

    // Init permuter
    std::vector<int>     graph_perm;
    std::vector<int>         perm;
    std::vector<int> etree;
    RXMESH_SOLVER::Ordering* ordering = nullptr;
    
    if (args.ordering_type == "DEFAULT") {
        spdlog::info("Using default ordering (default for each solver).");
        ordering = nullptr;
    } else if (args.ordering_type == "PATCH_ORDERING") {
        spdlog::info("Using PATCH_ORDERING ordering.");
        ordering = RXMESH_SOLVER::Ordering::create(
            RXMESH_SOLVER::DEMO_ORDERING_TYPE::PATCH_ORDERING);
        ordering->setOptions({{"binary_level", std::to_string(args.binary_level)}});
    } else if (args.ordering_type == "PARTH") {
        ordering = RXMESH_SOLVER::Ordering::create(
            RXMESH_SOLVER::DEMO_ORDERING_TYPE::PARTH);
        ordering->setOptions({{"binary_level", std::to_string(args.binary_level)}});
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
    PARTH::ParthAPI parth_compressor;
    parth_compressor.setMatrix(OL.rows(), OL.outerIndexPtr(), OL.innerIndexPtr(), DIM);
    int* Gp = parth_compressor.Mp;
    int* Gi = parth_compressor.Mi;
    int G_N = parth_compressor.M_n;
    assert(G_N == OL.rows() / DIM);

    double residual = 0;
    long int ordering_init_time = -1;
    long int ordering_time = -1;
    long int ordering_integration_time = -1;
    long int analysis_time = -1;
    long int factorization_time = -1;
    long int solve_time = -1;
    long int factor_nnz = -1;
    // Init the permuter
    if (ordering != nullptr) {
        spdlog::info("Start Customized Ordering ...");
        ordering->setGraph(Gp, Gi, G_N, Gp[G_N]); 
        auto ordering_init_start = std::chrono::high_resolution_clock::now();
        if(args.ordering_type == "PATCH_ORDERING") {
            reinterpret_cast<RXMESH_SOLVER::PatchOrdering*>(ordering)->_g_node_to_patch = node_to_patch;
            reinterpret_cast<RXMESH_SOLVER::PatchOrdering*>(ordering)->_cpu_order.init_patches(patch_ids.size(), node_to_patch, args.binary_level);
        } else {
            ordering->init();
        }
        auto ordering_init_end = std::chrono::high_resolution_clock::now();
        ordering_init_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                         ordering_init_end - ordering_init_start)
                         .count();
        spdlog::info("Ordering initialization time: {} ms",
                     ordering_init_time);
        auto ordering_start = std::chrono::high_resolution_clock::now();
        if(args.solver_type=="CUDSS") {
            ordering->compute_permutation(graph_perm, etree, true);
        } else {
            ordering->compute_permutation(graph_perm, etree, false);
        }
        if(args.DIM == 1){
            perm = graph_perm;
        } else {
            perm.resize(graph_perm.size() * args.DIM);
            for(int i1 = 0; i1 < graph_perm.size(); i1++){
                for(int j1 = 0; j1 < args.DIM; j1++){
                    perm[i1 * args.DIM + j1] = graph_perm[i1] * args.DIM + j1;
                }
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
        spdlog::info("Customize Ordering is done.");
    }


    solver->setMatrix(OL.outerIndexPtr(),
                      OL.innerIndexPtr(),
                      OL.valuePtr(),
                      OL.rows(),
                      OL.nonZeros());

    auto start = std::chrono::high_resolution_clock::now();
    solver->ordering(perm, etree);
    auto end = std::chrono::high_resolution_clock::now();
    ordering_integration_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    spdlog::info("Ordering integration time: {} ms",
             ordering_integration_time);
    // Symbolic analysis time
    start = std::chrono::high_resolution_clock::now();
    solver->analyze_pattern(perm, etree);
    end = std::chrono::high_resolution_clock::now();
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
    residual = (rhs - OL * result).norm();
    spdlog::info("Residual: {}", residual);
    spdlog::info("Final factor/matrix NNZ ratio: {}",
                 solver->getFactorNNZ() * 1.0 / OL.nonZeros());


    //Save data to a csv file
    std::string csv_name = args.output_csv_address;
    std::vector<std::string> header;
    header.emplace_back("mesh_name");
    header.emplace_back("G_N");
    header.emplace_back("G_NNZ");
    header.emplace_back("solver_type");
    header.emplace_back("ordering_type");
    header.emplace_back("nd_levels");
    header.emplace_back("patch_type");
    header.emplace_back("patch_size");
    header.emplace_back("patch_time");
    header.emplace_back("factor/matrix NNZ ratio");
    header.emplace_back("ordering_time");
    header.emplace_back("ordering_integration_time");
    header.emplace_back("analysis_time");
    header.emplace_back("factorization_time");
    header.emplace_back("solve_time");
    header.emplace_back("residual");


    RXMESH_SOLVER::CSVManager runtime_csv(csv_name, "some address", header, false);
    runtime_csv.addElementToRecord(mesh_name, "mesh_name");
    runtime_csv.addElementToRecord(OL.rows(), "G_N");
    runtime_csv.addElementToRecord(OL.nonZeros(), "G_NNZ");
    runtime_csv.addElementToRecord(args.solver_type, "solver_type");
    if(ordering!=nullptr) {
    runtime_csv.addElementToRecord(ordering->typeStr(), "ordering_type");
    } else {
        runtime_csv.addElementToRecord("DEFAULT", "ordering_type");
    }
    int nd_levels = std::log2(etree.size() + 1);
    runtime_csv.addElementToRecord(nd_levels, "nd_levels");
    runtime_csv.addElementToRecord(-1, "patch_size");
    if (args.ordering_type == "PATCH_ORDERING") {
        std::map<std::string, double> stat;
        assert(ordering != nullptr);
        ordering->getStatistics(stat);
        runtime_csv.addElementToRecord(args.patch_type, "patch_type");
        //Get the patch time from the name of the file
        std::string patch_time_str = args.input_node_to_patch.substr(args.input_node_to_patch.find("_") + 1, args.input_node_to_patch.find("_node_to_patch.txt") - args.input_node_to_patch.find("_") - 1);
        double patch_time = std::stod(patch_time_str);
        runtime_csv.addElementToRecord(patch_time, "patch_time");
    } else {
        runtime_csv.addElementToRecord("", "patch_type");
        runtime_csv.addElementToRecord(0, "patch_time");
    }
    runtime_csv.addElementToRecord(factor_nnz * 1.0 / OL.nonZeros(), "factor/matrix NNZ ratio");
    runtime_csv.addElementToRecord(ordering_time, "ordering_time");
    runtime_csv.addElementToRecord(ordering_integration_time, "ordering_integration_time");
    runtime_csv.addElementToRecord(analysis_time, "analysis_time");
    runtime_csv.addElementToRecord(factorization_time, "factorization_time");
    runtime_csv.addElementToRecord(solve_time, "solve_time");
    runtime_csv.addElementToRecord(residual, "residual");
    runtime_csv.addRecord();

    delete solver;
    delete ordering;
    return 0;
}