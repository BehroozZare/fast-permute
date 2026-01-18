//
// Created by behrooz on 2025-01-10.
//
// Mesh Smoothing Benchmark: Computes ordering once, performs symbolic analysis once,
// then loops through iterations for repeated factorization and solve.
// This is a multiple factorization benchmark with constant sparsity pattern.
//

#include <igl/read_triangle_mesh.h>
#include <igl/writeOBJ.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/doublearea.h>
#include <igl/barycenter.h>
#include <spdlog/spdlog.h>
#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include <chrono>
#include <cmath>
#include <map>
#include <iostream>
#include <filesystem>

#include "LinSysSolver.hpp"
#include "get_factor_nnz.h"
#include "check_valid_permutation.h"
#include "csv_utils.h"
#include "remove_diagonal.h"
#include "ordering.h"

// Polyscope for visualization
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/view.h"
#include "imgui.h"
#include "glm/glm.hpp"

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


// ============== Visualization: Global state for polyscope callback ==============
static std::vector<Eigen::MatrixXd>* g_vertex_history = nullptr;
static Eigen::MatrixXi* g_faces = nullptr;
static polyscope::SurfaceMesh* g_smoothing_mesh = nullptr;
static int g_current_iteration = 0;
static int g_total_iterations = 0;
static bool g_auto_play = false;
static float g_auto_play_speed = 0.5f;  // seconds per frame
static float g_last_update_time = 0.0f;

// Update mesh vertex positions for current iteration
inline void update_mesh_positions() {
    if (g_smoothing_mesh == nullptr || g_vertex_history == nullptr) return;
    if (g_current_iteration < 0 || g_current_iteration >= static_cast<int>(g_vertex_history->size())) return;

    const Eigen::MatrixXd& V = (*g_vertex_history)[g_current_iteration];
    
    // Convert to std::vector format for polyscope
    std::vector<std::array<double, 3>> vertices(V.rows());
    for (int i = 0; i < V.rows(); i++) {
        vertices[i] = {V(i, 0), V(i, 1), V(i, 2)};
    }
    
    g_smoothing_mesh->updateVertexPositions(vertices);
    
    // Compute mesh bounding box center
    Eigen::Vector3d minV = V.colwise().minCoeff();
    Eigen::Vector3d maxV = V.colwise().maxCoeff();
    Eigen::Vector3d center = (minV + maxV) / 2.0;
    
    // Compute a reasonable camera distance based on mesh size
    double meshSize = (maxV - minV).norm();
    double cameraDist = meshSize * 2.0;
    
    // Set camera to look at mesh center from a position along the Z axis
    glm::vec3 cameraPos(center.x(), center.y(), center.z() + cameraDist);
    glm::vec3 target(center.x(), center.y(), center.z());
    polyscope::view::lookAt(cameraPos, target);
}

// ImGui callback for smoothing viewer controls
inline void smoothing_viewer_callback() {
    // Set initial window position and size
    ImGui::SetNextWindowPos(ImVec2(20, 20), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(350, 250), ImGuiCond_FirstUseEver);
    
    ImGui::Begin("Mesh Smoothing Viewer", nullptr, 
                 ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize);
    
    ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.4f, 1.0f), "Smoothing Iteration Controls");
    ImGui::Separator();
    
    // Display current iteration
    ImGui::Text("Iteration: %d / %d", g_current_iteration, g_total_iterations - 1);
    
    // Slider for iteration selection
    bool changed = false;
    if (ImGui::SliderInt("##iteration", &g_current_iteration, 0, g_total_iterations - 1)) {
        changed = true;
    }
    
    ImGui::Separator();
    
    // Navigation buttons
    if (ImGui::Button("<< First")) {
        g_current_iteration = 0;
        changed = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("< Prev")) {
        if (g_current_iteration > 0) {
            g_current_iteration--;
            changed = true;
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Next >")) {
        if (g_current_iteration < g_total_iterations - 1) {
            g_current_iteration++;
            changed = true;
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Last >>")) {
        g_current_iteration = g_total_iterations - 1;
        changed = true;
    }
    
    ImGui::Separator();
    
    // Auto-play controls
    if (ImGui::Checkbox("Auto-play", &g_auto_play)) {
        g_last_update_time = static_cast<float>(ImGui::GetTime());
    }
    ImGui::SameLine();
    ImGui::SliderFloat("Speed", &g_auto_play_speed, 0.05f, 2.0f, "%.2f s");
    
    // Handle auto-play
    if (g_auto_play) {
        float current_time = static_cast<float>(ImGui::GetTime());
        if (current_time - g_last_update_time >= g_auto_play_speed) {
            g_current_iteration++;
            if (g_current_iteration >= g_total_iterations) {
                g_current_iteration = 0;  // Loop back to start
            }
            g_last_update_time = current_time;
            changed = true;
        }
    }
    
    // Update mesh if iteration changed
    if (changed) {
        update_mesh_positions();
    }
    
    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Keys: Left/Right arrows, Space to toggle play");
    
    ImGui::End();
    
    // Handle keyboard input
    if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow)) {
        if (g_current_iteration > 0) {
            g_current_iteration--;
            update_mesh_positions();
        }
    }
    if (ImGui::IsKeyPressed(ImGuiKey_RightArrow)) {
        if (g_current_iteration < g_total_iterations - 1) {
            g_current_iteration++;
            update_mesh_positions();
        }
    }
    if (ImGui::IsKeyPressed(ImGuiKey_Space)) {
        g_auto_play = !g_auto_play;
        g_last_update_time = static_cast<float>(ImGui::GetTime());
    }
}

// Main visualization function
inline void show_smoothing_visualization(std::vector<Eigen::MatrixXd>& history, 
                                          Eigen::MatrixXi& F) {
    if (history.empty()) {
        spdlog::warn("No vertex history to visualize.");
        return;
    }
    
    // Set global state
    g_vertex_history = &history;
    g_faces = &F;
    g_current_iteration = 0;
    g_total_iterations = static_cast<int>(history.size());
    g_auto_play = false;
    
    spdlog::info("Launching visualization with {} iterations...", g_total_iterations);
    
    // Initialize polyscope
    polyscope::init();
    
    // Convert initial vertices to std::vector format
    const Eigen::MatrixXd& V0 = history[0];
    std::vector<std::array<double, 3>> vertices(V0.rows());
    for (int i = 0; i < V0.rows(); i++) {
        vertices[i] = {V0(i, 0), V0(i, 1), V0(i, 2)};
    }
    
    // Convert faces to std::vector format
    std::vector<std::array<int, 3>> faces(F.rows());
    for (int i = 0; i < F.rows(); i++) {
        faces[i] = {F(i, 0), F(i, 1), F(i, 2)};
    }
    
    // Register the surface mesh
    g_smoothing_mesh = polyscope::registerSurfaceMesh("smoothed_mesh", vertices, faces);
    
    // Set the user callback
    polyscope::state::userCallback = smoothing_viewer_callback;
    
    // Show the polyscope GUI
    polyscope::show();
    
    // Cleanup: reset global state
    polyscope::state::userCallback = nullptr;
    g_vertex_history = nullptr;
    g_faces = nullptr;
    g_smoothing_mesh = nullptr;
}


struct CLIArgs
{
    std::string input_mesh;
    int num_iterations = 10;
    double delta = 0.001;
    int binary_level = 7;
    std::string output_csv_address = "/home/behrooz/Desktop/Last_Project/gpu_ordering/output/Smoothing/smoothing";
    std::string solver_type = "CHOLMOD";
    std::string ordering_type = "DEFAULT";
    std::string default_ordering_type = "METIS";
    std::string patch_type = "rxmesh";
    int patch_size = 512;
    bool use_gpu = false;
    bool visualize = false;
    bool save_meshes = true;
    std::string mesh_output_dir = "/media/behrooz/FarazHard/Last_Project/Smoothing_benchmark/smoothing_meshes";

    CLIArgs(int argc, char* argv[])
    {
        CLI::App app{"Mesh Smoothing Benchmark - Multiple factorization with constant sparsity pattern"};
        app.add_option("-i,--input", input_mesh, "Input mesh file path")->required();
        app.add_option("-n,--num_iterations", num_iterations, "Number of smoothing iterations");
        app.add_option("--delta", delta, "Smoothing step size (default: 0.001)");
        app.add_option("-a,--ordering", ordering_type, "Ordering type: DEFAULT, PATCH_ORDERING, PARTH, METIS, NEUTRAL");
        app.add_option("-d,--default_ordering_type", default_ordering_type, "Default ordering type for MKL: METIS, AMD, ParMETIS");
        app.add_option("-s,--solver", solver_type, "Solver type: CHOLMOD, CUDSS, MKL");
        app.add_option("-o,--output", output_csv_address, "Output CSV file path (without .csv extension)");
        app.add_option("-p,--patch_type", patch_type, "Patch type for PATCH_ORDERING: rxmesh, metis");
        app.add_option("-z,--patch_size", patch_size, "Patch size");
        app.add_option("-b,--binary_level", binary_level, "Binary level for nested dissection tree");
        app.add_option("-g,--use_gpu", use_gpu, "Use GPU for ordering");
        app.add_option("-v,--visualize", visualize, "Show mesh smoothing visualization after benchmark");
        app.add_option("--save_meshes", save_meshes, "Save mesh at each iteration as OBJ files");
        app.add_option("--mesh_output_dir", mesh_output_dir, "Directory to save mesh OBJ files");

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

    spdlog::info("=== Mesh Smoothing Benchmark ===");
    spdlog::info("Input mesh: {}", args.input_mesh);
    spdlog::info("Output CSV: {}", args.output_csv_address);
    spdlog::info("Solver: {}", args.solver_type);
    spdlog::info("Ordering: {}", args.ordering_type);
    spdlog::info("Num iterations: {}", args.num_iterations);
    spdlog::info("Delta (smoothing step): {}", args.delta);

    // ========== Load mesh ==========
    Eigen::MatrixXd V, U;
    Eigen::MatrixXi F;
    if (!igl::read_triangle_mesh(args.input_mesh, V, F)) {
        spdlog::error("Failed to read mesh from: {}", args.input_mesh);
        return 1;
    }
    spdlog::info("Loaded mesh: {} vertices, {} faces", V.rows(), F.rows());

    // Initialize smoothed positions
    U = V;

    // ========== Compute Laplacian L (once, constant) ==========
    Eigen::SparseMatrix<double> L;
    igl::cotmatrix(V, F, L);
    spdlog::info("Laplacian size: {} x {}, NNZ: {}", L.rows(), L.cols(), L.nonZeros());

    // ========== Compute initial mass matrix and system matrix ==========
    // This gives us the sparsity pattern for ordering and analysis
    Eigen::SparseMatrix<double> M;
    igl::massmatrix(U, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);
    Eigen::SparseMatrix<double> S = M - args.delta * L;
    spdlog::info("System matrix S size: {} x {}, NNZ: {}", S.rows(), S.cols(), S.nonZeros());

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
        if (args.default_ordering_type == "METIS") {
            solver->ordering_type = "METIS";
        } else if (args.default_ordering_type == "AMD") {
            solver->ordering_type = "AMD";
        } else if (args.default_ordering_type == "ParMETIS") {
            solver->ordering_type = "ParMETIS";
        } else {
            spdlog::error("Unknown default ordering type for MKL.");
            return 1;
        }
    } else {
        spdlog::error("Unknown solver type: {}. Supported: CHOLMOD, CUDSS, MKL", args.solver_type);
        return 1;
    }
    assert(solver != nullptr);

    // ========== Create graph for ordering ==========
    std::vector<int> Gp, Gi;
    RXMESH_SOLVER::remove_diagonal(S.rows(), S.outerIndexPtr(), S.innerIndexPtr(), Gp, Gi);

    // ========== Compute ordering (once, outside the loop) ==========
    std::vector<int> perm;
    std::vector<int> etree;
    long int ordering_init_time = -1;
    long int ordering_time = -1;
    RXMESH_SOLVER::Ordering* ordering = nullptr;
    std::string mesh_name = std::filesystem::path(args.input_mesh).stem().string();

    if (args.ordering_type == "DEFAULT") {
        spdlog::info("Using DEFAULT ordering (solver's built-in).");
    } else if (args.ordering_type == "METIS") {
        spdlog::info("Using METIS ordering.");
        ordering = RXMESH_SOLVER::Ordering::create(RXMESH_SOLVER::DEMO_ORDERING_TYPE::METIS);
        if (args.solver_type == "CUDSS") {
            spdlog::error("METIS ordering is not supported with CUDSS solver.");
            delete solver;
            return 1;
        }
    } else if (args.ordering_type == "PATCH_ORDERING") {
        spdlog::info("Using PATCH_ORDERING.");
        ordering = RXMESH_SOLVER::Ordering::create(RXMESH_SOLVER::DEMO_ORDERING_TYPE::PATCH_ORDERING);
        ordering->setOptions({
            {"use_gpu", args.use_gpu ? "1" : "0"},
            {"patch_type", args.patch_type},
            {"patch_size", std::to_string(args.patch_size)},
            {"binary_level", std::to_string(args.binary_level)}
        });
    } else if (args.ordering_type == "PARTH") {
        spdlog::info("Using PARTH ordering.");
        ordering = RXMESH_SOLVER::Ordering::create(RXMESH_SOLVER::DEMO_ORDERING_TYPE::PARTH);
        ordering->setOptions({{"binary_level", std::to_string(args.binary_level)}});
    } else if (args.ordering_type == "NEUTRAL") {
        spdlog::info("Using NEUTRAL ordering.");
        ordering = RXMESH_SOLVER::Ordering::create(RXMESH_SOLVER::DEMO_ORDERING_TYPE::NEUTRAL);
    } else {
        spdlog::error("Unknown ordering type: {}", args.ordering_type);
        delete solver;
        return 1;
    }

    // Compute ordering if using custom ordering
    if (ordering != nullptr) {
        // Provide mesh data if the ordering needs it
        if (ordering->needsMesh()) {
            ordering->setMesh(V.data(), V.rows(), V.cols(),
                              F.data(), F.rows(), F.cols());
        }
        ordering->setGraph(Gp.data(), Gi.data(), S.rows(), Gi.size());

        // Initialize ordering (timed)
        auto ordering_init_start = std::chrono::high_resolution_clock::now();
        ordering->init();
        auto ordering_init_end = std::chrono::high_resolution_clock::now();
        ordering_init_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            ordering_init_end - ordering_init_start).count();
        spdlog::info("Prep time: {} ms", ordering_init_time);

        // Compute permutation (timed)
        bool compute_etree = (args.solver_type == "CUDSS");
        auto ordering_start = std::chrono::high_resolution_clock::now();
        ordering->compute_permutation(perm, etree, compute_etree);
        auto ordering_end = std::chrono::high_resolution_clock::now();
        ordering_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            ordering_end - ordering_start).count();
        spdlog::info("Ordering time: {} ms", ordering_time);

        // Validate permutation
        if (!RXMESH_SOLVER::check_valid_permutation(perm.data(), perm.size())) {
            spdlog::error("Permutation is not valid!");
            delete solver;
            delete ordering;
            return 1;
        }
        spdlog::info("Permutation validated: size = {}", perm.size());

        solver->ordering_name = ordering->typeStr();
    }

    // ========== Compute factor NNZ for custom ordering ==========
    long int factor_nnz = -1;
    if (!perm.empty()) {
        factor_nnz = RXMESH_SOLVER::get_factor_nnz(S.outerIndexPtr(),
                                                   S.innerIndexPtr(),
                                                   S.valuePtr(),
                                                   S.rows(),
                                                   S.nonZeros(),
                                                   perm);
        spdlog::info("Factor NNZ ratio: {:.4f}", factor_nnz * 1.0 / S.nonZeros());
    }

    // ========== Set matrix and perform symbolic analysis (once) ==========
    Eigen::SparseMatrix<double> lower_S;
    if (args.solver_type == "MKL") {
        lower_S = S.triangularView<Eigen::Lower>();
        solver->setMatrix(lower_S.outerIndexPtr(),
                          lower_S.innerIndexPtr(),
                          lower_S.valuePtr(),
                          lower_S.rows(),
                          lower_S.nonZeros());
    } else {
        solver->setMatrix(S.outerIndexPtr(),
                          S.innerIndexPtr(),
                          S.valuePtr(),
                          S.rows(),
                          S.nonZeros());
    }

    // Set ordering
    auto ordering_integration_start = std::chrono::high_resolution_clock::now();
    solver->ordering(perm, etree);
    auto ordering_integration_end = std::chrono::high_resolution_clock::now();
    long int ordering_integration_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        ordering_integration_end - ordering_integration_start).count();
    spdlog::info("Ordering integration time: {} ms", ordering_integration_time);

    // Symbolic analysis (once)
    auto analysis_start = std::chrono::high_resolution_clock::now();
    solver->analyze_pattern(perm, etree);
    if (args.solver_type == "CUDSS") {
        CUDA_SYNC_CHECK();
    }
    auto analysis_end = std::chrono::high_resolution_clock::now();
    long int analysis_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        analysis_end - analysis_start).count();
    spdlog::info("Symbolic analysis time: {} ms", analysis_time);

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
    header.emplace_back("prep_time");
    header.emplace_back("ordering_time");
    header.emplace_back("ordering_integration_time");
    header.emplace_back("analysis_time");
    header.emplace_back("factorization_time");
    header.emplace_back("solve_time");
    header.emplace_back("residual");

    RXMESH_SOLVER::CSVManager runtime_csv(args.output_csv_address, "mesh_smoothing_benchmark", header, false);

    // ========== Store vertex positions for visualization ==========
    std::vector<Eigen::MatrixXd> vertex_history;
    if (args.visualize) {
        vertex_history.reserve(args.num_iterations + 1);
        vertex_history.push_back(V);  // Store initial positions
    }

    // ========== Save meshes to OBJ files (optional) ==========
    if (args.save_meshes) {
        // Create output directory if it doesn't exist
        std::filesystem::create_directories(args.mesh_output_dir);
        
        // Save initial mesh
        std::string initial_mesh_path = args.mesh_output_dir + "/" + mesh_name + "_0.obj";
        if (igl::writeOBJ(initial_mesh_path, V, F)) {
            spdlog::info("Saved initial mesh to: {}", initial_mesh_path);
        } else {
            spdlog::warn("Failed to save initial mesh to: {}", initial_mesh_path);
        }
    }

    // ========== Smoothing loop ==========
    spdlog::info("Starting smoothing loop with {} iterations...", args.num_iterations);

    for (int iter = 0; iter < args.num_iterations; iter++) {
        spdlog::info("--- Iteration {} ---", iter);

        // Recompute mass matrix based on current vertex positions
        igl::massmatrix(U, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);

        // Form system matrix S = M - delta*L
        S = M - args.delta * L;

        // Form RHS = M * U (use .eval() to force evaluation in release mode)
        Eigen::MatrixXd RHS = (M * U).eval();

        // Update matrix values in solver (same sparsity pattern)
        if (args.solver_type == "MKL") {
            lower_S = S.triangularView<Eigen::Lower>();
            solver->setMatrix(lower_S.outerIndexPtr(),
                              lower_S.innerIndexPtr(),
                              lower_S.valuePtr(),
                              lower_S.rows(),
                              lower_S.nonZeros());
        } else {
            solver->setMatrix(S.outerIndexPtr(),
                              S.innerIndexPtr(),
                              S.valuePtr(),
                              S.rows(),
                              S.nonZeros());
        }

        // Factorize (timed)
        auto factor_start = std::chrono::high_resolution_clock::now();
        solver->factorize();
        if (args.solver_type == "CUDSS") {
            CUDA_SYNC_CHECK();
        }
        auto factor_end = std::chrono::high_resolution_clock::now();
        long int factorization_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            factor_end - factor_start).count();
        spdlog::info("Factorization time: {} ms", factorization_time);

        // Solve for all coordinates at once (multi-RHS)
        // Verify RHS dimensions are correct
        if (RHS.rows() != S.rows() || RHS.cols() != 3) {
            spdlog::error("RHS dimensions incorrect: {} x {} (expected {} x 3)", 
                          RHS.rows(), RHS.cols(), S.rows());
            return 1;
        }
        Eigen::MatrixXd U_new(U.rows(), U.cols());
        auto solve_start = std::chrono::high_resolution_clock::now();
        solver->solve(RHS, U_new);
        if (args.solver_type == "CUDSS") {
            CUDA_SYNC_CHECK();
        }
        auto solve_end = std::chrono::high_resolution_clock::now();
        long int solve_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            solve_end - solve_start).count();
        spdlog::info("Solve time: {} ms", solve_time);

        U = U_new;

        // Compute centroid and subtract (important for numerics)
        Eigen::VectorXd dblA;
        igl::doublearea(U, F, dblA);
        double area = 0.5 * dblA.sum();
        Eigen::MatrixXd BC;
        igl::barycenter(U, F, BC);
        Eigen::RowVector3d centroid(0, 0, 0);
        for (int i = 0; i < BC.rows(); i++) {
            centroid += 0.5 * dblA(i) / area * BC.row(i);
        }
        U.rowwise() -= centroid;
        // Normalize to unit surface area (important for numerics)
        U.array() /= sqrt(area);

        // Store vertex positions for visualization
        if (args.visualize) {
            vertex_history.push_back(U);
        }

        // Save mesh to OBJ file
        if (args.save_meshes) {
            std::string mesh_path = args.mesh_output_dir + "/" + mesh_name + "_" + std::to_string(iter + 1) + ".obj";
            if (igl::writeOBJ(mesh_path, U, F)) {
                spdlog::info("Saved mesh iteration {} to: {}", iter + 1, mesh_path);
            } else {
                spdlog::warn("Failed to save mesh iteration {} to: {}", iter + 1, mesh_path);
            }
        }

        // Compute residual (for the first coordinate as representative)
        Eigen::VectorXd rhs_check = RHS.col(0);
        double residual = (rhs_check - S * U.col(0)).norm();
        spdlog::info("Residual (coord 0): {}", residual);

        // Record to CSV
        runtime_csv.addElementToRecord(mesh_name, "mesh_name");
        runtime_csv.addElementToRecord(iter, "iteration");
        runtime_csv.addElementToRecord(args.ordering_type, "ordering_type");
        runtime_csv.addElementToRecord(args.solver_type, "solver_type");
        runtime_csv.addElementToRecord(static_cast<int>(S.rows()), "G_N");
        runtime_csv.addElementToRecord(static_cast<int>(S.nonZeros()), "G_NNZ");

        // ND levels from etree
        int nd_levels = etree.empty() ? 0 : static_cast<int>(std::log2(etree.size() + 1));
        runtime_csv.addElementToRecord(nd_levels, "nd_levels");

        // Patch statistics
        if (args.ordering_type == "PATCH_ORDERING" && ordering != nullptr && iter == 0) {
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
            runtime_csv.addElementToRecord(factor_nnz * 1.0 / S.nonZeros(), "factor/matrix NNZ ratio");
        } else {
            runtime_csv.addElementToRecord(solver->getFactorNNZ() * 1.0 / S.nonZeros(), "factor/matrix NNZ ratio");
        }

        // Timing (ordering times are only meaningful for first iteration)
        runtime_csv.addElementToRecord(ordering_init_time, "prep_time");
        runtime_csv.addElementToRecord(ordering_time, "ordering_time");
        runtime_csv.addElementToRecord(ordering_integration_time, "ordering_integration_time");
        runtime_csv.addElementToRecord(analysis_time, "analysis_time");
        runtime_csv.addElementToRecord(factorization_time, "factorization_time");
        runtime_csv.addElementToRecord(solve_time, "solve_time");
        runtime_csv.addElementToRecord(residual, "residual");
        runtime_csv.addRecord();

        ordering_init_time = 0;
        ordering_time = 0;
        ordering_integration_time = 0;
        analysis_time = 0;
        factorization_time = 0;
        solve_time = 0;
    }

    spdlog::info("=== Mesh Smoothing Benchmark Complete ===");
    spdlog::info("Results saved to: {}", args.output_csv_address);

    // ========== Visualization (optional) ==========
    if (args.visualize && !vertex_history.empty()) {
        spdlog::info("Launching visualization with {} frames...", vertex_history.size());
        show_smoothing_visualization(vertex_history, F);
    }

    // Cleanup
    delete solver;
    if (ordering != nullptr) {
        delete ordering;
    }

    return 0;
}
