//
// Created by behrooz on 2025-01-10.
//
// Patch Remeshing Benchmark: Computes ordering on interior vertices,
// maintains node_to_patch across remeshing iterations, and solves
// Laplacian equation with Dirichlet boundary conditions.
//

#include <igl/read_triangle_mesh.h>
#include <igl/slice.h>
#include <igl/setdiff.h>
#include <igl/colon.h>
#include <igl/avg_edge_length.h>
#include <igl/adjacency_list.h>
#include <spdlog/spdlog.h>
#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <chrono>
#include <cmath>
#include <map>
#include <iostream>
#include <filesystem>
#include <unordered_map>

#include "LinSysSolver.hpp"
#include "SPD_cot_matrix.h"
#include "get_factor_nnz.h"
#include "check_valid_permutation.h"
#include "csv_utils.h"
#include "remove_diagonal.h"
#include "ordering.h"
#include "create_patch_with_metis.h"
#include "createPatch.h"
#include "update_perm_with_boundary.h"
#include "remesh_the_patch.h"

// Polyscope for visualization
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

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


// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Identifies boundary vertices based on z-coordinate threshold
 * @param V Vertex matrix
 * @param F Face matrix
 * @param boundary_vertices Output vector of boundary vertex indices
 * @param interior_vertices Output vector of interior vertex indices
 * @param interior_inv Inverse mapping: global vertex -> interior index (-1 if boundary)
 */
void identify_boundary_vertices(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    Eigen::VectorXi& boundary_vertices,
    Eigen::VectorXi& interior_vertices,
    Eigen::VectorXi& interior_inv)
{
    Eigen::VectorXd z_coordinate = V.col(1);  // Using Y as "height"
    double min_Z = z_coordinate.minCoeff();
    double threshold = igl::avg_edge_length(V, F);
    
    std::vector<bool> is_boundary(V.rows(), false);
    int num_boundary = 0;
    
    for (int i = 0; i < z_coordinate.rows(); i++) {
        if (z_coordinate(i) - min_Z < threshold) {
            is_boundary[i] = true;
            num_boundary++;
        }
    }
    
    // Build boundary vertex list
    boundary_vertices.resize(num_boundary);
    int cnt = 0;
    for (size_t i = 0; i < is_boundary.size(); i++) {
        if (is_boundary[i]) {
            boundary_vertices[cnt++] = i;
        }
    }
    
    // Build interior vertex list using setdiff
    Eigen::VectorXi all_vertices, IA;
    igl::colon<int>(0, V.rows() - 1, all_vertices);
    igl::setdiff(all_vertices, boundary_vertices, interior_vertices, IA);
    
    // Build inverse mapping
    interior_inv.resize(V.rows());
    interior_inv.setConstant(-1);
    for (int i = 0; i < interior_vertices.rows(); i++) {
        interior_inv(interior_vertices(i)) = i;
    }
    
    spdlog::info("Identified {} boundary vertices, {} interior vertices", 
                 num_boundary, interior_vertices.rows());
}

/**
 * @brief Gets the most common patch ID among neighbors of a vertex
 * @param vertex_idx Global vertex index
 * @param adjacency Adjacency list
 * @param interior_inv Mapping from global to interior index
 * @param node_to_patch Current node_to_patch for interior vertices
 * @return Most common patch ID among neighbors, or 0 if no valid neighbors
 */
int get_max_neighbor_patch(
    int vertex_idx,
    const int* outer_index,
    const int* inner_index,
    const int* node_to_patch)
{
    std::unordered_map<int, int> patch_counts;
    
    for (int i = outer_index[vertex_idx]; i < outer_index[vertex_idx + 1]; i++) {
        int neighbor = inner_index[i];
        int patch_id = node_to_patch[neighbor];
        if (patch_id >= 0) {
            patch_counts[patch_id]++;
        }
    }
    
    if (patch_counts.empty()) {
        return 0;  // Default patch
    }
    
    // Find max
    int max_patch = 0;
    int max_count = 0;
    for (const auto& [patch_id, count] : patch_counts) {
        if (count > max_count) {
            max_count = count;
            max_patch = patch_id;
        }
    }
    
    return max_patch;
}

/**
 * @brief Updates node_to_patch after remeshing
 * @param old_to_new_dof_map Mapping from old vertex indices to new ones (-1 if deleted)
 * @param old_in Old interior vertex indices
 * @param old_interior_inv Old interior inverse mapping
 * @param old_node_to_patch Old node_to_patch for interior vertices
 * @param new_in New interior vertex indices
 * @param new_interior_inv New interior inverse mapping
 * @param adjacency New mesh adjacency list
 * @param new_node_to_patch Output: new node_to_patch for interior vertices
 */
// void update_node_to_patch(
//     const std::vector<int>& old_to_new_dof_map,
//     const Eigen::VectorXi& old_in,
//     const Eigen::VectorXi& old_interior_inv,
//     const std::vector<int>& old_node_to_patch,
//     const Eigen::VectorXi& new_in,
//     const Eigen::VectorXi& new_interior_inv,
//     const std::vector<std::vector<int>>& adjacency,
//     std::vector<int>& new_node_to_patch)
// {
//     new_node_to_patch.resize(new_in.rows(), -1);
//
//     // 1. Map existing interior vertices
//     for (int old_interior_idx = 0; old_interior_idx < old_in.rows(); old_interior_idx++) {
//         int old_global_v = old_in[old_interior_idx];
//         if (old_global_v < 0 || old_global_v >= static_cast<int>(old_to_new_dof_map.size())) {
//             continue;
//         }
//
//         int new_global_v = old_to_new_dof_map[old_global_v];
//         if (new_global_v != -1 && new_global_v < new_interior_inv.rows()) {
//             int new_interior_idx = new_interior_inv[new_global_v];
//             if (new_interior_idx != -1 && new_interior_idx < static_cast<int>(new_node_to_patch.size())) {
//                 new_node_to_patch[new_interior_idx] = old_node_to_patch[old_interior_idx];
//             }
//         }
//     }
//
//     // 2. Assign new vertices using max neighbor patch
//     for (int i = 0; i < static_cast<int>(new_node_to_patch.size()); i++) {
//         if (new_node_to_patch[i] == -1) {
//             int global_v = new_in[i];
//             new_node_to_patch[i] = get_max_neighbor_patch(global_v, adjacency, new_interior_inv, new_node_to_patch);
//         }
//     }
//
//     spdlog::info("Updated node_to_patch: {} interior vertices", new_node_to_patch.size());
// }


// ============================================================================
// CLI Arguments
// ============================================================================

struct CLIArgs
{
    std::string input_mesh;
    int num_iterations = 5;
    int patch_size = 512;
    int binary_level = 7;
    double patch_percentage = 0.05;  // Fraction of mesh to remesh
    double remesh_target_scale = 1.0;  // Scale factor for target edge length
    int remesh_iters = 3;  // Number of remeshing iterations per step
    std::string output_csv_address = "/home/behrooz/Desktop/Last_Project/gpu_ordering/output/multiple_factorization/patch_remeshing";
    std::string solver_type = "CHOLMOD";
    std::string ordering_type = "PATCH_ORDERING";
    std::string default_ordering_type = "METIS";
    int center_face_id = -1;  // -1 means random
    bool use_gpu = false;
    bool visualize = false;

    CLIArgs(int argc, char* argv[])
    {
        CLI::App app{"Patch Remeshing Benchmark - Multiple factorization with changing sparsity pattern"};
        app.add_option("-i,--input", input_mesh, "Input mesh file path")->required();
        app.add_option("-n,--num_iterations", num_iterations, "Number of remeshing iterations");
        app.add_option("-a,--ordering", ordering_type, "Ordering type: DEFAULT, PATCH_ORDERING");
        app.add_option("-d,--default_ordering_type", default_ordering_type, "Default ordering type for MKL: METIS, AMD");
        app.add_option("-s,--solver", solver_type, "Solver type: CHOLMOD, CUDSS, MKL");
        app.add_option("-o,--output", output_csv_address, "Output CSV file path (without .csv extension)");
        app.add_option("-z,--patch_size", patch_size, "Patch size for node_to_patch computation");
        app.add_option("-b,--binary_level", binary_level, "Binary level for nested dissection tree");
        app.add_option("--patch_percentage", patch_percentage, "Fraction of mesh to remesh (0.0 to 1.0)");
        app.add_option("--remesh_target", remesh_target_scale, "Scale factor for target edge length");
        app.add_option("--remesh_iters", remesh_iters, "Number of remeshing sub-iterations");
        app.add_option("--center_face", center_face_id, "Center face ID for patch selection (-1 for random)");
        app.add_option("-g,--use_gpu", use_gpu, "Use GPU for ordering");
        app.add_option("-v,--visualize", visualize, "Show mesh visualization after benchmark");

        try {
            app.parse(argc, argv);
        } catch (const CLI::ParseError& e) {
            exit(app.exit(e));
        }
    }
};


// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[])
{
    CLIArgs args(argc, argv);

    spdlog::info("=== Patch Remeshing Benchmark ===");
    spdlog::info("Input mesh: {}", args.input_mesh);
    spdlog::info("Output CSV: {}", args.output_csv_address);
    spdlog::info("Solver: {}", args.solver_type);
    spdlog::info("Ordering: {}", args.ordering_type);
    spdlog::info("Num iterations: {}", args.num_iterations);
    spdlog::info("Patch percentage: {}", args.patch_percentage);

    // ========== Load mesh ==========
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    if (!igl::read_triangle_mesh(args.input_mesh, V, F)) {
        spdlog::error("Failed to read mesh from: {}", args.input_mesh);
        return 1;
    }
    spdlog::info("Loaded mesh: {} vertices, {} faces", V.rows(), F.rows());

    // ========== Initialize solver ==========
    RXMESH_SOLVER::LinSysSolver* solver = nullptr;
    if (args.solver_type == "CHOLMOD") {
        solver = RXMESH_SOLVER::LinSysSolver::create(RXMESH_SOLVER::LinSysSolverType::CPU_CHOLMOD);
        spdlog::info("Using CHOLMOD direct solver.");
    } else if (args.solver_type == "CUDSS") {
        solver = RXMESH_SOLVER::LinSysSolver::create(RXMESH_SOLVER::LinSysSolverType::GPU_CUDSS);
        spdlog::info("Using CUDSS direct solver.");
    } else if (args.solver_type == "MKL") {
        solver = RXMESH_SOLVER::LinSysSolver::create(RXMESH_SOLVER::LinSysSolverType::CPU_MKL);
        spdlog::info("Using Intel MKL PARDISO direct solver.");
        solver->ordering_type = args.default_ordering_type;
    } else {
        spdlog::error("Unknown solver type: {}. Supported: CHOLMOD, CUDSS, MKL", args.solver_type);
        return 1;
    }
    assert(solver != nullptr);

    // ========== Prepare CSV output ==========
    std::vector<std::string> header;
    header.emplace_back("mesh_name");
    header.emplace_back("iteration");
    header.emplace_back("ordering_type");
    header.emplace_back("solver_type");
    header.emplace_back("num_vertices");
    header.emplace_back("num_interior");
    header.emplace_back("num_faces");
    header.emplace_back("L_in_in_NNZ");
    header.emplace_back("patch_size");
    header.emplace_back("num_patches");
    header.emplace_back("factor/matrix NNZ ratio");
    header.emplace_back("patching_time");
    header.emplace_back("ordering_time");
    header.emplace_back("node_to_patch_update_time");
    header.emplace_back("analysis_time");
    header.emplace_back("factorization_time");
    header.emplace_back("solve_time");
    header.emplace_back("remesh_time");
    header.emplace_back("residual");

    std::string mesh_name = std::filesystem::path(args.input_mesh).stem().string();
    RXMESH_SOLVER::CSVManager runtime_csv(args.output_csv_address, "patch_remeshing_benchmark", header, false);

    // ========== Main loop ==========
    RXMESH_SOLVER::Ordering* ordering = nullptr;
    std::vector<int> perm;
    std::vector<int> etree;
    std::vector<int> interior_node_to_patch;
    
    // State variables that persist across iterations
    Eigen::VectorXi in, b, in_inv;  // Interior, boundary, and inverse mapping
    Eigen::SparseMatrix<double> L_in_in, L_in_b;
    
    // Center face for patch selection
    int center_face = args.center_face_id;
    if (center_face < 0 || center_face >= F.rows()) {
        center_face = F.rows() / 2;  // Default to middle face
    }
    std::vector<int> new_to_old_map; //Mapping from old vertices to new vertices
    std::vector<int> node_to_patch; //Mapping from interior vertices to patches
    Eigen::VectorXi selected_remesh_patch;
    for (int iter = 0; iter < args.num_iterations; iter++) {
        spdlog::info("========== Iteration {} ==========", iter);

        // Reset solver state when matrix structure changes (after remeshing)
        if (iter > 0) {
            solver->resetSolver();
        }

        // ========== Identify boundary vertices ==========
        Eigen::VectorXi old_in = in;
        Eigen::VectorXi old_in_inv = in_inv;
        
        identify_boundary_vertices(V, F, b, in, in_inv);
        
        if (b.rows() == 0) {
            spdlog::warn("No boundary found, using first 10% of vertices as boundary");
            int num_boundary = std::max(1, static_cast<int>(V.rows() * 0.1));
            b.resize(num_boundary);
            for (int i = 0; i < num_boundary; i++) {
                b[i] = i;
            }
            Eigen::VectorXi all_vertices, IA;
            igl::colon<int>(0, V.rows() - 1, all_vertices);
            igl::setdiff(all_vertices, b, in, IA);
            in_inv.resize(V.rows());
            in_inv.setConstant(-1);
            for (int i = 0; i < in.rows(); i++) {
                in_inv(in(i)) = i;
            }
        }

        // ========== Build interior Laplacian L_in_in ==========
        // Use SPD cotangent matrix (guaranteed positive definite via element-wise projection)
        Eigen::SparseMatrix<double> L;
        RXMESH_SOLVER::computeSPD_cot_matrix(V, F, L);
        igl::slice(L, in, in, L_in_in);
        igl::slice(L, in, b, L_in_b);
        
        // Already positive definite from computeSPD_cot_matrix (no negation needed)
        Eigen::SparseMatrix<double> A_in_in = L_in_in;
        
        spdlog::info("L_in_in size: {} x {}, NNZ: {}", L_in_in.rows(), L_in_in.cols(), L_in_in.nonZeros());

        // ========== Compute or update node_to_patch ==========
        auto node_to_patch_start = std::chrono::high_resolution_clock::now();
        long int patching_time = 0;
        int num_patches = 0;
        // if(args.ordering_type == "PATCH_ORDERING") {
        //     if (iter == 0) {
        //         // Initial computation using METIS
        //         RXMESH_SOLVER::create_patch_with_metis(
        //             L.rows(),
        //             const_cast<int*>(L.outerIndexPtr()),
        //             const_cast<int*>(L.innerIndexPtr()),
        //             1,  // DIM=1 for scalar Laplacian
        //             args.patch_size,
        //             node_to_patch
        //         );
        //
        //         //Update the node_to_patch with the boundary
        //         //Conver b from VectorXi to vector<int>
        //         std::vector<int> b_vec(b.rows());
        //         for(int i = 0; i < b.rows(); i++){
        //             b_vec[i] = b(i);
        //         }
        //         RXMESH_SOLVER::update_perm_with_boundary(node_to_patch, b_vec);
        //         assert(node_to_patch.size() == in.rows());
        //         // Count unique patches
        //         std::unordered_set<int> unique_patches(node_to_patch.begin(), node_to_patch.end());
        //         num_patches = unique_patches.size();
        //
        //         auto node_to_patch_end = std::chrono::high_resolution_clock::now();
        //         patching_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        //             node_to_patch_end - node_to_patch_start).count();
        //         spdlog::info("Initial node_to_patch computed: {} patches in {} ms", num_patches, patching_time);
        //     } else {
        //         //Create new to old mapping
        //         std::vector<int> new_to_old_map(L_in_in.rows(), -1);
        //         for(int i = 0; i < old_to_new_dof_map.size(); i++){
        //             if(old_to_new_dof_map[i] == -1){
        //                 continue;
        //             }
        //             int new_vertex_id = old_to_new_dof_map[i];
        //             new_to_old_map[new_vertex_id] = i;
        //         }
        //         //Update the node_to_patch
        //         std::vector<int> new_node_to_patch(L_in_in.rows(), -1);
        //         //Compute new_to_old_map
        //         for(int i = 0; i < new_to_old_map.size(); i++){
        //             if(new_to_old_map[i] == -1){
        //                 //Update with max neighbor patch
        //                 int max_neighbor_patch = get_max_neighbor_patch(i, L_in_in.outerIndexPtr(), in_inv, node_to_patch);
        //                 new_node_to_patch[i] = max_neighbor_patch;
        //             }
        //             int old_vertex_id = new_to_old_map[i];
        //             new_node_to_patch[i] = node_to_patch[old_vertex_id];
        //         }
        //         std::unordered_set<int> unique_patches(new_node_to_patch.begin(), new_node_to_patch.end());
        //         num_patches = unique_patches.size();
        //
        //         auto node_to_patch_end = std::chrono::high_resolution_clock::now();
        //         patching_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        //             node_to_patch_end - node_to_patch_start).count();
        //         spdlog::info("Updated node_to_patch: {} patches in {} ms", num_patches, patching_time);
        //     }
        // }
        // ========== Create/update ordering ==========
        long int ordering_time = 0;
        
        if (args.ordering_type == "PATCH_ORDERING") {
            // Clean up previous ordering
            if (ordering != nullptr) {
                delete ordering;
            }
            
            ordering = RXMESH_SOLVER::Ordering::create(RXMESH_SOLVER::DEMO_ORDERING_TYPE::PATCH_ORDERING);
            ordering->setOptions({
                {"use_gpu", args.use_gpu ? "1" : "0"},
                {"patch_type", "reuse_patch"},  // We're providing our own patches
                {"patch_size", std::to_string(args.patch_size)},
                {"binary_level", std::to_string(args.binary_level)}
            });
            
            // Create graph for ordering (remove diagonal)
            std::vector<int> Gp, Gi;
            RXMESH_SOLVER::remove_diagonal(A_in_in.rows(), A_in_in.outerIndexPtr(), A_in_in.innerIndexPtr(), Gp, Gi);
            ordering->setGraph(Gp.data(), Gi.data(), A_in_in.rows(), Gi.size());
            
            // Set our computed node_to_patch
            ordering->setPatch(interior_node_to_patch);
            
            // Compute permutation
            auto ordering_start = std::chrono::high_resolution_clock::now();
            bool compute_etree = (args.solver_type == "CUDSS");
            ordering->compute_permutation(perm, etree, compute_etree);
            auto ordering_end = std::chrono::high_resolution_clock::now();
            ordering_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                ordering_end - ordering_start).count();
            
            // Validate permutation
            if (!RXMESH_SOLVER::check_valid_permutation(perm.data(), perm.size())) {
                spdlog::error("Permutation is not valid!");
                delete solver;
                delete ordering;
                return 1;
            }
            
            spdlog::info("Ordering computed in {} ms, perm size: {}", ordering_time, perm.size());
            solver->ordering_name = ordering->typeStr();
        } else {
            // Default ordering - solver handles it
            perm.clear();
            etree.clear();
            spdlog::info("Using DEFAULT ordering (solver's built-in).");
        }

        // ========== Set matrix and analyze ==========
        // Eigen::SparseMatrix<double> lower_A;
        // if (args.solver_type == "MKL") {
        //     lower_A = A_in_in.triangularView<Eigen::Lower>();
        //     solver->setMatrix(lower_A.outerIndexPtr(),
        //                       lower_A.innerIndexPtr(),
        //                       lower_A.valuePtr(),
        //                       lower_A.rows(),
        //                       lower_A.nonZeros());
        // } else {
        //     solver->setMatrix(A_in_in.outerIndexPtr(),
        //                       A_in_in.innerIndexPtr(),
        //                       A_in_in.valuePtr(),
        //                       A_in_in.rows(),
        //                       A_in_in.nonZeros());
        // }
        //
        // // Set ordering
        // solver->ordering(perm, etree);
        //
        // // Symbolic analysis
        // auto analysis_start = std::chrono::high_resolution_clock::now();
        // solver->analyze_pattern(perm, etree);
        // if (args.solver_type == "CUDSS") {
        //     CUDA_SYNC_CHECK();
        // }
        // auto analysis_end = std::chrono::high_resolution_clock::now();
        // long int analysis_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        //     analysis_end - analysis_start).count();
        // spdlog::info("Symbolic analysis time: {} ms", analysis_time);
        //
        // // ========== Factorize ==========
        // auto factor_start = std::chrono::high_resolution_clock::now();
        // solver->factorize();
        // if (args.solver_type == "CUDSS") {
        //     CUDA_SYNC_CHECK();
        // }
        // auto factor_end = std::chrono::high_resolution_clock::now();
        // long int factorization_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        //     factor_end - factor_start).count();
        // spdlog::info("Factorization time: {} ms", factorization_time);
        //
        // // ========== Solve Laplacian with Dirichlet BC ==========
        // // BC: boundary vertices get constant value
        // Eigen::VectorXd Z = V.col(2);  // Use z-coordinate as solution template
        // Z.setConstant(5.0);  // Constant boundary value
        // Eigen::VectorXd bc = Z(b);
        //
        // Eigen::VectorXd rhs = L_in_b * bc;
        // Eigen::VectorXd sol;
        //
        // auto solve_start = std::chrono::high_resolution_clock::now();
        // solver->solve(rhs, sol);
        // if (args.solver_type == "CUDSS") {
        //     CUDA_SYNC_CHECK();
        // }
        // auto solve_end = std::chrono::high_resolution_clock::now();
        // long int solve_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        //     solve_end - solve_start).count();
        // spdlog::info("Solve time: {} ms", solve_time);
        //
        // Update solution
        // Z(in) = sol;

        // Compute residual
        // double residual = (rhs - A_in_in * sol).norm();
        // spdlog::info("Residual: {}", residual);

        // ========== Record to CSV ==========
        long int factor_nnz = solver->getFactorNNZ();
        double nnz_ratio = factor_nnz > 0 ? factor_nnz * 1.0 / A_in_in.nonZeros() : -1;
        
        runtime_csv.addElementToRecord(mesh_name, "mesh_name");
        runtime_csv.addElementToRecord(iter, "iteration");
        runtime_csv.addElementToRecord(args.ordering_type, "ordering_type");
        runtime_csv.addElementToRecord(args.solver_type, "solver_type");
        runtime_csv.addElementToRecord(static_cast<int>(V.rows()), "num_vertices");
        runtime_csv.addElementToRecord(static_cast<int>(in.rows()), "num_interior");
        runtime_csv.addElementToRecord(static_cast<int>(F.rows()), "num_faces");
        runtime_csv.addElementToRecord(static_cast<int>(A_in_in.nonZeros()), "L_in_in_NNZ");
        runtime_csv.addElementToRecord(args.patch_size, "patch_size");
        runtime_csv.addElementToRecord(num_patches, "num_patches");
        runtime_csv.addElementToRecord(nnz_ratio, "factor/matrix NNZ ratio");
        runtime_csv.addElementToRecord(patching_time, "patching_time");
        runtime_csv.addElementToRecord(ordering_time, "ordering_time");
        runtime_csv.addElementToRecord(0, "node_to_patch_update_time");  // Included in patching_time
        // runtime_csv.addElementToRecord(analysis_time, "analysis_time");
        // runtime_csv.addElementToRecord(factorization_time, "factorization_time");
        // runtime_csv.addElementToRecord(solve_time, "solve_time");

        // ========== Remeshing (if not last iteration) ==========
        long int remesh_time = 0;
        if (iter < args.num_iterations - 1) {
            //Select a random patch for remeshing
            std::default_random_engine generator (iter);
            std::uniform_real_distribution<double> distribution (0, F.rows());
            int fid = distribution(generator);
            RXMESH_SOLVER::createPatch(fid, 0.01, selected_remesh_patch, F, V);
            Eigen::MatrixXi F_up;
            Eigen::MatrixXd V_up;
    
            selected_remesh_patch = RXMESH_SOLVER::remesh(selected_remesh_patch, F, V,
                F_up, V_up, args.remesh_target_scale, new_to_old_map);

            V = V_up;
            F = F_up;
        }
        
        runtime_csv.addElementToRecord(remesh_time, "remesh_time");
        // runtime_csv.addElementToRecord(residual, "residual");
        runtime_csv.addRecord();
    }

    spdlog::info("=== Patch Remeshing Benchmark Complete ===");
    spdlog::info("Results saved to: {}", args.output_csv_address);

    // ========== Visualization (optional) ==========
    if (args.visualize) {
        spdlog::info("Launching visualization...");
        polyscope::init();
        
        std::vector<std::array<double, 3>> vertices(V.rows());
        for (int i = 0; i < V.rows(); i++) {
            vertices[i] = {V(i, 0), V(i, 1), V(i, 2)};
        }
        
        std::vector<std::array<int, 3>> faces(F.rows());
        for (int i = 0; i < F.rows(); i++) {
            faces[i] = {F(i, 0), F(i, 1), F(i, 2)};
        }
        
        polyscope::registerSurfaceMesh("remeshed_mesh", vertices, faces);
        polyscope::show();
    }

    // Cleanup
    delete solver;
    if (ordering != nullptr) {
        delete ordering;
    }

    return 0;
}
