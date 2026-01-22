//
// Created by behrooz on 2025-09-28.
//



#include <igl/read_triangle_mesh.h>
#include <igl/readMESH.h>
#include <igl/cotmatrix.h>
#include <spdlog/spdlog.h>
#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include <chrono>
#include <unsupported/Eigen/SparseExtra>
#include <iostream>
#include <filesystem>

#include "SPD_cot_matrix.h"
#include "LinSysSolver.hpp"
#include "get_factor_nnz.h"
#include "check_valid_permutation.h"
#include "ordering.h"
#include "remove_diagonal.h"
#include "csv_utils.h"
#include "save_vector.h"
#include "create_patch_with_metis.h"



struct CLIArgs
{
    std::string input_mesh;
    int binary_level = 8; // It is zero-based so 9 is 10 levels
    std::string output_csv_address="/home/behrooz/Desktop/Last_Project/gpu_ordering/output/single_factorization/laplace";//Include absolute path with csv file name without .csv extension
    std::string solver_type   = "CHOLMOD";
    std::string ordering_type = "DEFAULT";
    std::string default_ordering_type = "METIS";
    std::string patch_type = "rxmesh";
    std::string check_point_address = "/home/behrooz/Desktop/Last_Project/gpu_ordering/benchmark/single_factorization/test_data";
    int patch_size = 512;
    bool use_gpu = false;
    bool store_check_points = false;

    CLIArgs(int argc, char* argv[]) 
    {
        CLI::App app{"Separator analysis"};
        app.add_option("-a,--ordering", ordering_type, "ordering type");
        app.add_option("-d,--default_ordering_type", default_ordering_type, "default ordering type");
        app.add_option("-s,--solver", solver_type, "solver type");
        app.add_option("-o,--output", output_csv_address, "output folder name");
        app.add_option("-i,--input", input_mesh, "input mesh name");
        app.add_option("-g,--use_gpu", use_gpu, "use gpu");
        app.add_option("-z,--patch_size", patch_size, "patch size");
        app.add_option("-b,--binary_level", binary_level, "binary level for binary tree ordering");
        app.add_option("-c,--store_check_points", store_check_points, "store check points");
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

    if (args.input_mesh.empty()) {
        std::cerr << "Error: Input mesh file not specified. Use -i or --input "
                     "to specify the mesh file."
                  << std::endl;
        return 1;
    }

    std::cout << "Loading mesh from: " << args.input_mesh << std::endl;
    std::cout << "Output folder: " << args.output_csv_address << std::endl;

    Eigen::MatrixXd OV;
    Eigen::MatrixXi OT;  // Tetrahedra
    Eigen::MatrixXi OF;  // Boundary faces
    if (!igl::readMESH(args.input_mesh, OV, OT, OF)) {
        std::cerr << "Failed to read the tet mesh: " << args.input_mesh
                  << std::endl;
        return 1;
    }
    std::cout << "Loaded tet mesh with " << OV.rows() << " vertices, " 
              << OT.rows() << " tetrahedra, " << OF.rows() << " boundary faces" << std::endl;

    // Create SPD cotangent matrix (already positive definite with regularization)
    Eigen::SparseMatrix<double> OL;
    RXMESH_SOLVER::computeSPD_cot_matrix(OV, OT, OL);

    // Print laplacian size and sparsity
    spdlog::info("Number of rows: {}", OL.rows());
    spdlog::info("Number of non-zeros: {}", OL.nonZeros());
    spdlog::info(
        "Sparsity: {:.2f}%",
        (1 - (OL.nonZeros() / static_cast<double>(OL.rows() * OL.rows()))) *
            100);

    Eigen::VectorXd rhs = Eigen::VectorXd::Random(OL.rows());
    Eigen::VectorXd result;

    // Init permuter
    std::vector<int>         perm;
    std::vector<int> etree;
    RXMESH_SOLVER::Ordering* ordering = nullptr;
    std::string mesh_name = std::filesystem::path(args.input_mesh).stem().string();
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
    } else if (args.solver_type == "MKL") {
        solver = RXMESH_SOLVER::LinSysSolver::create(
            RXMESH_SOLVER::LinSysSolverType::CPU_MKL);
        spdlog::info("Using Intel MKL PARDISO direct solver.");
        if(args.default_ordering_type == "METIS") {
            solver->ordering_type = "METIS";
        } else if(args.default_ordering_type == "AMD") {
            solver->ordering_type = "AMD";
        } else if (args.default_ordering_type == "ParMETIS") {
            solver->ordering_type = "ParMETIS";
        } else {
            spdlog::error("Unknown default ordering type.");
            return 1;
        }
    } else {
        spdlog::error("Unknown solver type.");
    }
    assert(solver != nullptr);

    // Create the graph
    std::vector<int> Gp;
    std::vector<int> Gi;
    RXMESH_SOLVER::remove_diagonal(
        OL.rows(), OL.outerIndexPtr(), OL.innerIndexPtr(), Gp, Gi);


    double residual = 0;
    long int ordering_init_time = -1;
    long int ordering_time = -1;
    long int ordering_integration_time = -1;
    long int analysis_time = -1;
    long int factorization_time = -1;
    long int solve_time = -1;
    long int factor_nnz = -1;
    // Init the permuter
    double patch_time = 0;
    if (ordering != nullptr) {
        spdlog::info("Start Customized Ordering ...");
        // Provide mesh data if the ordering needs it (e.g., RXMesh ND)
        if (ordering->needsMesh()) {
            // Pass raw pointers to avoid ABI issues between C++ and CUDA compilation
            ordering->setMesh(OV.data(), OV.rows(), OV.cols(),
                            OF.data(), OF.rows(), OF.cols());
        }
        ordering->setGraph(Gp.data(), Gi.data(), OL.rows(), Gi.size());
        auto ordering_init_start = std::chrono::high_resolution_clock::now();
        if(args.ordering_type == "PATCH_ORDERING") {
            auto patch_start = std::chrono::high_resolution_clock::now();
            std::vector<int> node_to_patch;
            RXMESH_SOLVER::create_patch_with_metis(OL.rows(), OL.outerIndexPtr(), OL.innerIndexPtr(),
                1, args.patch_size, node_to_patch);
            ordering->setPatch(node_to_patch);
            auto patch_end = std::chrono::high_resolution_clock::now();
            patch_time = std::chrono::duration_cast<std::chrono::milliseconds>(patch_end - patch_start).count();
            spdlog::info("Patch time: {} ms", patch_time);
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
        spdlog::info("Customize Ordering is done.");
    }


    // MKL PARDISO expects upper triangular in CSR format.
    // Since Eigen uses CSC, passing lower triangular CSC is equivalent
    // (the CSC-to-CSR transpose makes lower become upper).
    Eigen::SparseMatrix<double> lower_OL;
    if (args.solver_type == "MKL") {
        lower_OL = OL.triangularView<Eigen::Lower>();
        solver->setMatrix(lower_OL.outerIndexPtr(),
                          lower_OL.innerIndexPtr(),
                          lower_OL.valuePtr(),
                          lower_OL.rows(),
                          lower_OL.nonZeros());
    } else {
        solver->setMatrix(OL.outerIndexPtr(),
                          OL.innerIndexPtr(),
                          OL.valuePtr(),
                          OL.rows(),
                          OL.nonZeros());
    }

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
    if (args.ordering_type == "PATCH_ORDERING") {
        std::map<std::string, double> stat;
        assert(ordering != nullptr);
        ordering->getStatistics(stat);
        runtime_csv.addElementToRecord(args.patch_type, "patch_type");
        runtime_csv.addElementToRecord(stat["patch_size"], "patch_size");
        runtime_csv.addElementToRecord(patch_time, "patch_time");
    } else {
        runtime_csv.addElementToRecord("", "patch_type");
        runtime_csv.addElementToRecord(0, "patch_size");
        runtime_csv.addElementToRecord(0, "patch_time");
    }


    if(args.solver_type == "MKL") {
        runtime_csv.addElementToRecord(solver->getFactorNNZ() * 1.0 / OL.nonZeros(), "factor/matrix NNZ ratio");
    } else {
        runtime_csv.addElementToRecord(factor_nnz * 1.0 / OL.nonZeros(), "factor/matrix NNZ ratio");
    }
    runtime_csv.addElementToRecord(ordering_time, "ordering_time");
    runtime_csv.addElementToRecord(ordering_integration_time, "ordering_integration_time");
    runtime_csv.addElementToRecord(analysis_time, "analysis_time");
    runtime_csv.addElementToRecord(factorization_time, "factorization_time");
    runtime_csv.addElementToRecord(solve_time, "solve_time");
    runtime_csv.addElementToRecord(residual, "residual");
    runtime_csv.addRecord();


    //Save the matrix
    if(args.store_check_points) {
        spdlog::info("Saving checkpoints ...");
        std::string check_point_address = args.check_point_address;
        std::string matirx_save_address = check_point_address + "/" + mesh_name + ".mtx";
        std::string parameters = "level=" + std::to_string(args.binary_level) +
            ",patch_type=" + args.patch_type + ",patch_size=" + std::to_string(args.patch_size) + ",ordering_time=" + std::to_string(ordering_time);
        std::string ordering_name = "DEFAULT";
        if (ordering != nullptr) {
            ordering_name = ordering->typeStr();
        }
        std::string perm_save_address = check_point_address + "/perm_" + mesh_name + "_" + ordering_name + "_" + parameters + ".txt";
        std::string etree_save_address = check_point_address + "/etree_" + mesh_name + "_" + ordering_name + "_" + parameters + ".txt";
        if (ordering_name == "DEFAULT") {
            Eigen::saveMarket(OL, matirx_save_address);
        }
        if (!perm.empty()) {
            RXMESH_SOLVER::save_vector_to_file(perm, perm_save_address);
        }
        if (!etree.empty()) {
            RXMESH_SOLVER::save_vector_to_file(etree, etree_save_address);
        }
        if(args.ordering_type == "PATCH_ORDERING") {
            std::vector<int> node_to_patch;
            ordering->getPatch(node_to_patch);
            std::string node_to_patch_save_address = check_point_address + "/node_to_patch_" + mesh_name + ".txt";
            RXMESH_SOLVER::save_vector_to_file(node_to_patch, node_to_patch_save_address);
        }
    }

    delete solver;
    delete ordering;
    return 0;
}