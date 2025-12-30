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
#include <cassert>


#include "LinSysSolver.hpp"
#include "get_factor_nnz.h"
#include "check_valid_permutation.h"
#include "csv_utils.h"
#include "save_vector.h"
#include "parth/parth.h"
#include "ordering.h"
#include "create_patch_with_parth.h"
#include "patch_ordering.h"
#include "read_face_and_pos.h"
#include "create_patch_with_metis.h"
#include "show_patches.h"
#include "show_separator_per_level.h"

struct CLIArgs
{
    int binary_level = 7; // It is zero-based so 9 is 10 levels
    std::string output_csv_address = "/home/behrooz/Desktop/Last_Project/gpu_ordering/output/IPC";//Include absolute path with csv file name without .csv extension
    std::string solver_type   = "CUDSS";
    std::string ordering_type = "DEFAULT";
    std::string patch_type = "rxmesh";
    std::string input_matrix = "/media/behrooz/FarazHard/IPC_matrices/MatOnBoard/Matrices/hessian_0_0_last_IPC.mtx";
    std::string input_faces = "/media/behrooz/FarazHard/IPC_matrices/MatOnBoard/SF";
    std::string input_V = "/media/behrooz/FarazHard/IPC_matrices/MatOnBoard/V_rest";
    int patch_size = 512;
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
        app.add_option("-i,--input_matrix", input_matrix, "input matrix");

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

    std::cout << "Loading matrix from: " << args.input_matrix << std::endl;
    std::cout << "Loading faces from: " << args.input_faces << std::endl;
    std::cout << "Loading vertex positions" << args.input_V << std::endl;
    std::cout << "Output csv address: " << args.output_csv_address << std::endl;
    int DIM = 3;
    
    //Load the matrix
    Eigen::SparseMatrix<double> base; // Or Eigen::MatrixXd for dense matrices
    Eigen::loadMarket(base, args.input_matrix);

    //Load the mesh
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    read_face_and_pos(args.input_V, args.input_faces, V, F);
    assert(V.rows() == base.rows() / DIM);

    
    PARTH::ParthAPI parth_compressor;
    Eigen::SparseMatrix<double> mat;
    Eigen::loadMarket(mat, args.input_matrix);
    parth_compressor.setMatrix(mat.rows(), mat.outerIndexPtr(), mat.innerIndexPtr(), DIM);
    int* Gp = parth_compressor.Mp;
    int* Gi = parth_compressor.Mi;
    int G_N = parth_compressor.M_n;

    std::vector<int> node_to_patch;
    std::vector<int> parth_perm, parth_new_labels, parth_sep_ptr;
    // RXMESH_SOLVER::create_patch_with_parth(base.rows(), base.outerIndexPtr(), base.innerIndexPtr(), DIM,
    //  args.patch_size, node_to_patch, parth_perm, parth_new_labels, parth_sep_ptr);
    RXMESH_SOLVER::create_patch_with_metis(base.rows(), base.outerIndexPtr(), base.innerIndexPtr(), DIM, args.patch_size, node_to_patch);
    assert(node_to_patch.size() == base.rows() / DIM);

    //====== Metis ordering ======
    RXMESH_SOLVER::Ordering* metis_ordering;
    metis_ordering = RXMESH_SOLVER::Ordering::create(RXMESH_SOLVER::DEMO_ORDERING_TYPE::METIS);
    metis_ordering->setGraph(Gp, Gi, G_N, G_N);
    std::vector<int> metis_etree, metis_perm;
    metis_ordering->compute_permutation(metis_perm, metis_etree, false);
    std::vector<int> matrix_metis_perm(metis_perm.size() * DIM);
    for(int i = 0; i < metis_perm.size(); i++){
        for(int j = 0; j < DIM; j++){
            matrix_metis_perm[i * DIM + j] = metis_perm[i] * DIM + j;
        }
    }
    double metis_factor_nnz = RXMESH_SOLVER::get_factor_nnz(base.outerIndexPtr(),
                                                base.innerIndexPtr(),
                                                base.valuePtr(),
                                                base.rows(),
                                                base.nonZeros(),
                                                matrix_metis_perm);


    //====== Parth ordering ======
    assert(parth_perm.size() == base.rows());
    double parth_factor_nnz = RXMESH_SOLVER::get_factor_nnz(base.outerIndexPtr(),
                                                base.innerIndexPtr(),
                                                base.valuePtr(),
                                                base.rows(),
                                                base.nonZeros(),
                                                parth_perm);

    //====== Patch ordering ======
    RXMESH_SOLVER::Ordering* patch_ordering = RXMESH_SOLVER::Ordering::create(RXMESH_SOLVER::DEMO_ORDERING_TYPE::PATCH_ORDERING);
    patch_ordering->setGraph(Gp, Gi, G_N, G_N);
    reinterpret_cast<RXMESH_SOLVER::PatchOrdering*>(patch_ordering)->_g_node_to_patch = node_to_patch;
    reinterpret_cast<RXMESH_SOLVER::PatchOrdering*>(patch_ordering)->_cpu_order.init_patches(node_to_patch.size(), node_to_patch, args.binary_level);
    std::vector<int> matrix_perm, matrix_etree;
    auto ordering_start = std::chrono::high_resolution_clock::now();
    std::vector<int> patch_perm, patch_etree;
    patch_ordering->compute_permutation(patch_perm, patch_etree, true);
    //Map to global permutation
    matrix_perm.resize(patch_perm.size() * DIM);
    for(int i1 = 0; i1 < patch_perm.size(); i1++){
        for(int j = 0; j < DIM; j++){
            matrix_perm[i1 * DIM + j] = patch_perm[i1] * DIM + j;
        }
    }
    matrix_etree = patch_etree;
    for(auto& value: matrix_etree){
        value = value * DIM;
    }
    assert(matrix_perm.size() == mat.rows());
    auto ordering_end = std::chrono::high_resolution_clock::now();
    long int ordering_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        ordering_end - ordering_start)
        .count();
            
    //Check for correct perm
    if (!RXMESH_SOLVER::check_valid_permutation(matrix_perm.data(), matrix_perm.size())) {
        spdlog::error("Permutation is not valid!");
    }
    spdlog::info("Ordering time: {} ms",
                ordering_time);


    double patch_factor_nnz = RXMESH_SOLVER::get_factor_nnz(base.outerIndexPtr(),
                        base.innerIndexPtr(),
                        base.valuePtr(),
                        base.rows(),
                        base.nonZeros(),
                        matrix_perm);

    //====== Patch vs Metis vs Parth ======
    spdlog::info("The ratio of patching factor to metis factor is {}",
        (patch_factor_nnz * 1.0 /metis_factor_nnz));
    spdlog::info("The ratio of patching factor to parth factor is {}",
        (patch_factor_nnz * 1.0 /parth_factor_nnz));

    //Show the patches
    RXMESH_SOLVER::show_patches(G_N, Gp, Gi, V, F, node_to_patch);

    // //Show Parth separators
    // RXMESH_SOLVER::show_separator_per_level(V, F, parth_new_labels, parth_sep_ptr);
    //Show Patch separators
    std::vector<int> patch_labels;
    std::vector<int> patch_sep_ptr;
    patch_ordering->getEtree(patch_labels, patch_sep_ptr);
    RXMESH_SOLVER::show_separator_per_level(V, F,patch_labels, patch_sep_ptr);
    delete patch_ordering;
    delete metis_ordering;

    return 0;
}