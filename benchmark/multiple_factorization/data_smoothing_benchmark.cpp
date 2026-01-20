#include <igl/read_triangle_mesh.h>
#include <igl/hessian_energy.h>
#include <igl/curved_hessian_energy.h>
#include <igl/massmatrix.h>
#include <igl/cotmatrix.h>
#include <igl/vertex_components.h>
#include <igl/remove_unreferenced.h>
#include <igl/heat_geodesics.h>

#include <CLI/CLI.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCholesky>

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "imgui.h"

#include <spdlog/spdlog.h>

#include "LinSysSolver.hpp"
#include "ordering.h"
#include "remove_diagonal.h"
#include "csv_utils.h"
#include "get_factor_nnz.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <set>
#include <limits>
#include <stdlib.h>



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



// Global state for polyscope visualization
static polyscope::SurfaceMesh* g_mesh = nullptr;
static int g_current_mode = 0;  // Mode for noisy function section
static int g_current_section = 0; // 0 = noisy function, 1 = step function

// Noisy function data
static Eigen::VectorXd* g_zexact = nullptr;
static Eigen::VectorXd* g_znoisy = nullptr;
static Eigen::VectorXd* g_zl = nullptr;
static Eigen::VectorXd* g_zh = nullptr;
static Eigen::VectorXd* g_zch = nullptr;

// Update visualization based on current mode and section
inline void updateVisualization() {
    const Eigen::VectorXd* z = nullptr;
    std::string name;
    

    // Noisy function section
    switch (g_current_mode) {
        case 0:
            z = g_zexact;
            name = "exact";
            break;
        case 1:
            z = g_znoisy;
            name = "noisy";
            break;
        case 2:
            z = g_zl;
            name = "laplacian_smoothed";
            break;
        case 3:
            z = g_zh;
            name = "hessian_smoothed";
            break;
        case 4:
            z = g_zch;
            name = "curved_hessian_smoothed";
            break;
    }
    
    if (z && g_mesh) {
        std::vector<double> scalarData(z->rows());
        for (int i = 0; i < z->rows(); i++) {
            scalarData[i] = (*z)(i);
        }
        auto q = g_mesh->addVertexScalarQuantity("scalar_field", scalarData);
        q->setEnabled(true);
        q->setMapRange(std::make_pair(z->minCoeff(), z->maxCoeff()));
    }
}

// ImGui callback for mode selection
inline void dataSmoothingCallback() {
    ImGui::SetNextWindowPos(ImVec2(20, 20), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(350, 350), ImGuiCond_FirstUseEver);
    
    ImGui::Begin("Data Smoothing");
    
    bool changed = false;
    
    ImGui::Separator();
    ImGui::Text("Select visualization mode:");
    ImGui::Separator();
    
    // Noisy function modes
    if (ImGui::RadioButton("1. Original (exact)", g_current_mode == 0)) {
        g_current_mode = 0;
        changed = true;
    }
    if (ImGui::RadioButton("2. Noisy", g_current_mode == 1)) {
        g_current_mode = 1;
        changed = true;
    }
    if (ImGui::RadioButton("3. Laplacian smoothed (zero Neumann)", g_current_mode == 2)) {
        g_current_mode = 2;
        changed = true;
    }
    if (ImGui::RadioButton("4. Hessian smoothed (natural planar)", g_current_mode == 3)) {
        g_current_mode = 3;
        changed = true;
    }
    if (ImGui::RadioButton("5. Curved Hessian smoothed (natural curved)", g_current_mode == 4)) {
        g_current_mode = 4;
        changed = true;
    }
    
    if (changed) {
        updateVisualization();
    }
    
    ImGui::End();
}

struct CLIArgs {
    std::string mesh_path;
    std::string ordering_type = "DEFAULT";
    std::string solver_type = "CUDSS";
    std::string patch_type = "rxmesh";
    int patch_size = 512;
    int binary_level = 7;
    bool use_gpu = false;
    std::string output_csv_address="/home/behrooz/Desktop/Last_Project/gpu_ordering/output/NoiseSmoothing/NoiseSmoothing";//Include absolute path with csv file name without .csv extension
    
    CLIArgs(int argc, char* argv[]) {
        CLI::App app{"Data smoothing benchmark"};
        app.add_option("-m,--mesh", mesh_path, "Path to input mesh")->required();
        app.add_option("-a,--ordering", ordering_type, "Ordering type: DEFAULT, PARTH, PATCH_ORDERING");
        app.add_option("-s,--solver", solver_type, "Solver type: CUDSS, MKL");
        app.add_option("-p,--patch_type", patch_type, "Patch type: rxmesh or metis");
        app.add_option("-z,--patch_size", patch_size, "Patch size for PATCH_ORDERING");
        app.add_option("-b,--binary_level", binary_level, "Binary level for nested dissection");
        app.add_option("-g,--use_gpu", use_gpu, "Use GPU for ordering");
        try {
            app.parse(argc, argv);
        } catch (const CLI::ParseError& e) {
            exit(app.exit(e));
        }
    }
};

// Create ordering based on CLI arguments
RXMESH_SOLVER::Ordering* createOrdering(const CLIArgs& args) {
    RXMESH_SOLVER::Ordering* ordering = nullptr;
    
    if (args.ordering_type == "DEFAULT") {
        spdlog::info("Using DEFAULT ordering (solver's internal ordering).");
        ordering = nullptr;
    } else if (args.ordering_type == "PATCH_ORDERING") {
        spdlog::info("Using PATCH_ORDERING ordering.");
        ordering = RXMESH_SOLVER::Ordering::create(
            RXMESH_SOLVER::DEMO_ORDERING_TYPE::PATCH_ORDERING);
        ordering->setOptions(
            {{"use_gpu", args.use_gpu ? "1" : "0"},
             {"patch_type", args.patch_type},
             {"patch_size", std::to_string(args.patch_size)},
             {"use_patch_separator", "1"},
             {"patch_ordering_local_permute_method", "amd"},
             {"binary_level", std::to_string(args.binary_level)}});
    } else if (args.ordering_type == "PARTH") {
        spdlog::info("Using PARTH ordering.");
        ordering = RXMESH_SOLVER::Ordering::create(
            RXMESH_SOLVER::DEMO_ORDERING_TYPE::PARTH);
        ordering->setOptions({{"binary_level", std::to_string(args.binary_level)}});
    } else {
        spdlog::error("Unknown ordering type: {}. Using DEFAULT.", args.ordering_type);
        ordering = nullptr;
    }
    
    return ordering;
}

// Struct to hold solve results and timing data for CSV export
struct SolveResult {
    Eigen::VectorXd solution;
    long int ordering_init_time = 0;
    long int ordering_patch_time = 0;
    long int ordering_compute_permutation_time = 0;
    long int ordering_integration_time = 0;
    long int analysis_time = 0;
    long int factorization_time = 0;
    long int solve_time = 0;
    double residual = 0.0;
    std::string ordering_type = "DEFAULT";
    int matrix_rows = 0;
    int matrix_nnz = 0;
    // New fields to match scp_benchmark.cpp
    int nd_levels = 0;
    std::string patch_type = "";
    int patch_size = 0;
    double factor_nnz_ratio = 0.0;
};

// Solve a linear system using CUDSS with optional custom ordering
SolveResult solveCUDSS(
    const Eigen::SparseMatrix<double>& A,
    const Eigen::VectorXd& rhs,
    const CLIArgs& args,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::string& solver_name)
{
    spdlog::info("=== {} Solver (CUDSS) ===", solver_name);
    
    // Drop any existing RXMesh logger to avoid "logger already exists" error
    spdlog::drop("RXMesh");
    
    // Create CUDSS solver
    RXMESH_SOLVER::LinSysSolver* solver = RXMESH_SOLVER::LinSysSolver::create(
        RXMESH_SOLVER::LinSysSolverType::GPU_CUDSS);
    
    // Create ordering
    RXMESH_SOLVER::Ordering* ordering = createOrdering(args);
    
    // Prepare permutation and etree vectors
    std::vector<int> perm;
    std::vector<int> etree;
    
    // Timing variables
    long int ordering_init_time = 0;
    long int ordering_patch_time = 0;
    long int ordering_compute_permutation_time = 0;
    long int ordering_integration_time = 0;
    long int analysis_time = 0;
    long int factorization_time = 0;
    long int solve_time = 0;
    
    // If using custom ordering, compute the permutation
    if (ordering != nullptr) {
        // Remove diagonal to get graph for ordering
        std::vector<int> Gp;
        std::vector<int> Gi;
        RXMESH_SOLVER::remove_diagonal(
            A.rows(), 
            const_cast<int*>(A.outerIndexPtr()), 
            const_cast<int*>(A.innerIndexPtr()), 
            Gp, Gi);
        
        // Check if the matrix is diagonal (no off-diagonal entries)
        // In this case, skip custom ordering as it's not needed
        if (Gi.empty()) {
            spdlog::info("{} - Matrix is diagonal, skipping custom ordering", solver_name);
            delete ordering;
            ordering = nullptr;
        } else {
            // Provide mesh data if the ordering needs it (e.g., PATCH_ORDERING)
            if (ordering->needsMesh()) {
                ordering->setMesh(V.data(), V.rows(), V.cols(),
                                  F.data(), F.rows(), F.cols());
            }
            
            ordering->setGraph(Gp.data(), Gi.data(), A.rows(), Gi.size());
            
            auto ordering_init_start = std::chrono::high_resolution_clock::now();
            ordering->init();
            CUDA_SYNC_CHECK();
            auto ordering_init_end = std::chrono::high_resolution_clock::now();
            ordering_init_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                ordering_init_end - ordering_init_start).count();
            spdlog::info("{} - Ordering initialization time: {} ms", solver_name, ordering_init_time);
            auto ordering_compute_permutation_start = std::chrono::high_resolution_clock::now();
            ordering->compute_permutation(perm, etree, true);  // true for CUDSS etree format
            CUDA_SYNC_CHECK();
            auto ordering_compute_permutation_end = std::chrono::high_resolution_clock::now();
            ordering_compute_permutation_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                ordering_compute_permutation_end - ordering_compute_permutation_start).count();
            spdlog::info("{} - Ordering compute permutation time: {} ms", solver_name, ordering_compute_permutation_time);
            solver->ordering_name = ordering->typeStr();
        }
    }
    
    // Set matrix
    solver->setMatrix(
        const_cast<int*>(A.outerIndexPtr()),
        const_cast<int*>(A.innerIndexPtr()),
        const_cast<double*>(A.valuePtr()),
        A.rows(),
        A.nonZeros());
    
    // Ordering integration
    auto ordering_integration_start = std::chrono::high_resolution_clock::now();
    solver->ordering(perm, etree);
    CUDA_SYNC_CHECK();
    auto ordering_integration_end = std::chrono::high_resolution_clock::now();
    ordering_integration_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        ordering_integration_end - ordering_integration_start).count();
    spdlog::info("{} - Ordering integration time: {} ms", solver_name, ordering_integration_time);
    // Analyze pattern
    auto start = std::chrono::high_resolution_clock::now();
    solver->analyze_pattern(perm, etree);
    CUDA_SYNC_CHECK();
    auto end = std::chrono::high_resolution_clock::now();
    analysis_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    spdlog::info("{} - Analysis time: {} ms", solver_name, analysis_time);
    
    // Factorize
    start = std::chrono::high_resolution_clock::now();
    solver->factorize();
    CUDA_SYNC_CHECK();
    end = std::chrono::high_resolution_clock::now();
    factorization_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    spdlog::info("{} - Factorization time: {} ms", solver_name, factorization_time);
    
    // Solve
    Eigen::VectorXd result;
    start = std::chrono::high_resolution_clock::now();
    solver->solve(const_cast<Eigen::VectorXd&>(rhs), result);
    CUDA_SYNC_CHECK();
    end = std::chrono::high_resolution_clock::now();
    solve_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    spdlog::info("{} - Solve time: {} ms", solver_name, solve_time);
    
    // Compute residual
    double residual = (rhs - A * result).norm();
    spdlog::info("{} - Residual: {}", solver_name, residual);
    
    // Get ordering statistics (patch_time, patch_size, patch_type)
    ordering_patch_time = 0;
    int patch_size = 0;
    std::string patch_type_str = "";
    if(ordering != nullptr && args.ordering_type == "PATCH_ORDERING") {
        std::map<std::string, double> stat;
        ordering->getStatistics(stat);
        ordering_patch_time = stat["patching_time"];
        patch_size = static_cast<int>(stat["patch_size"]);
        patch_type_str = args.patch_type;
    }
    
    // Compute nd_levels from etree
    int nd_levels = etree.empty() ? 0 : static_cast<int>(std::log2(etree.size() + 1));
    
    // Compute factor NNZ ratio
    double factor_nnz_ratio = 0.0;
    // if (!perm.empty()) {
    //     long int factor_nnz = RXMESH_SOLVER::get_factor_nnz(
    //         A.outerIndexPtr(),
    //         A.innerIndexPtr(),
    //         A.valuePtr(),
    //         A.rows(),
    //         A.nonZeros(),
    //         perm);
    //     factor_nnz_ratio = factor_nnz * 1.0 / A.nonZeros();
    //     spdlog::info("{} - Factor NNZ ratio: {:.4f}", solver_name, factor_nnz_ratio);
    // } else {
    //     // Use solver's internal factor NNZ if available
    //     factor_nnz_ratio = solver->getFactorNNZ() * 1.0 / A.nonZeros();
    // }
    
    // Build result struct
    SolveResult solve_result;
    solve_result.solution = result;
    solve_result.ordering_init_time = ordering_init_time;
    solve_result.ordering_patch_time = ordering_patch_time;
    solve_result.ordering_compute_permutation_time = ordering_compute_permutation_time;
    solve_result.ordering_integration_time = ordering_integration_time;
    solve_result.analysis_time = analysis_time;
    solve_result.factorization_time = factorization_time;
    solve_result.solve_time = solve_time;
    solve_result.residual = residual;
    solve_result.ordering_type = (ordering != nullptr) ? ordering->typeStr() : "DEFAULT";
    solve_result.matrix_rows = A.rows();
    solve_result.matrix_nnz = A.nonZeros();
    // New fields
    solve_result.nd_levels = nd_levels;
    solve_result.patch_type = patch_type_str;
    solve_result.patch_size = patch_size;
    solve_result.factor_nnz_ratio = factor_nnz_ratio;
    
    // Cleanup
    delete solver;
    delete ordering;
    
    return solve_result;
}

int main(int argc, char * argv[])
{
  CLIArgs args(argc, argv);
  
  typedef Eigen::SparseMatrix<double> SparseMat;
  srand(57);
  
  //Read our mesh
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  if(!igl::read_triangle_mesh(args.mesh_path, V, F)) {
    spdlog::error("Failed to load mesh: {}", args.mesh_path);
    return 1;
  }
  spdlog::info("The mesh is read with V size of {}", V.rows());
  //Constructing an exact function to smooth
  igl::HeatGeodesicsData<double> hgData;
  igl::heat_geodesics_precompute(V, F, hgData);
  Eigen::VectorXd heatDist;
  Eigen::VectorXi gamma(1);
  gamma << std::min(1947, (int)V.rows() - 1); // This is the number of vertices in the mesh
  igl::heat_geodesics_solve(hgData, gamma, heatDist);
  Eigen::VectorXd zexact = 0.1*(heatDist.array() 
  + (-heatDist.maxCoeff())).pow(2) 
  + 3*V.block(0,1,V.rows(),1).array().cos();
  
  //Make the exact function noisy
  const double s = 0.1*(zexact.maxCoeff() - zexact.minCoeff());
  Eigen::VectorXd znoisy = zexact + s*Eigen::VectorXd::Random(zexact.size());
  
  //Constructing the squared Laplacian and squared Hessian energy
    spdlog::info("Preprocessing...");
  SparseMat L, M;
  igl::cotmatrix(V, F, L);
  igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);
  // Solve M * MinvL = L using CUDSS column-by-column
  // SparseMat MinvL = solveMatrixCUDSS(M, L, args, V, F);
  // SparseMat QL = L.transpose()*MinvL;
  SparseMat QH;
  igl::hessian_energy(V, F, QH);
  SparseMat QcH;
  igl::curved_hessian_energy(V, F, QcH);
    spdlog::info("Preprocessing is done.");
  
  // Log matrix sizes
  spdlog::info("V rows: {}", V.rows());
  spdlog::info("F rows: {}", F.rows());
  spdlog::info("L rows: {}", L.rows());
  spdlog::info("M rows: {}, Q NNZ: {}", M.rows(), M.nonZeros());
  // spdlog::info("QL rows: {}, QL NNZ: {}", QL.rows(), QL.nonZeros());
  spdlog::info("QH rows: {}, QH NNZ: {}", QH.rows(), QH.nonZeros());
  spdlog::info("QcH rows: {}, QcH NNZ: {}", QcH.rows(), QcH.nonZeros());
  
  //Solve to find Laplacian-smoothed Hessian-smoothed, and
  // curved-Hessian-smoothed solutions using CUDSS
  spdlog::info("Using ordering type: {}", args.ordering_type);
  
  // Build the system matrices
  // const double al = 3e-7;
  // SparseMat lapMatrix = al*QL + (1.-al)*M;
  // Eigen::VectorXd lapRhs = (1.-al)*M*znoisy;
  
  const double ah = 2e-7;
  SparseMat hessMatrix = ah*QH + (1.-ah)*M;
  Eigen::VectorXd hessRhs = (1.-ah)*M*znoisy;
  
  const double ach = 3e-7;
  SparseMat curvedHessMatrix = ach*QcH + (1.-ach)*M;
  Eigen::VectorXd curvedHessRhs = (1.-ach)*M*znoisy;
  
  // Solve using CUDSS with the selected ordering
  // SolveResult lapResult = solveCUDSS(lapMatrix, lapRhs, args, V, F, "Laplacian");
  SolveResult hessResult = solveCUDSS(hessMatrix, hessRhs, args, V, F, "Hessian");
  SolveResult curvedHessResult = solveCUDSS(curvedHessMatrix, curvedHessRhs, args, V, F, "CurvedHessian");
  
  // Extract solutions
  // Eigen::VectorXd zl = lapResult.solution;
  Eigen::VectorXd zh = hessResult.solution;
  Eigen::VectorXd zch = curvedHessResult.solution;
  
  // Export timing data to CSV
  std::string mesh_name = std::filesystem::path(args.mesh_path).stem().string();
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
  
  RXMESH_SOLVER::CSVManager runtime_csv(args.output_csv_address, "output", header, false);
  
  // Record for Laplacian solver
  // runtime_csv.addElementToRecord(mesh_name, "mesh_name");
  // runtime_csv.addElementToRecord("Laplacian", "solver_name");
  // runtime_csv.addElementToRecord(lapResult.matrix_rows, "G_N");
  // runtime_csv.addElementToRecord(lapResult.matrix_nnz, "G_NNZ");
  // runtime_csv.addElementToRecord(lapResult.ordering_type, "ordering_type");
  // runtime_csv.addElementToRecord(lapResult.ordering_time, "ordering_time");
  // runtime_csv.addElementToRecord(lapResult.analysis_time, "analysis_time");
  // runtime_csv.addElementToRecord(lapResult.factorization_time, "factorization_time");
  // runtime_csv.addElementToRecord(lapResult.solve_time, "solve_time");
  // runtime_csv.addElementToRecord(lapResult.ordering_time + lapResult.analysis_time +
  //                                 lapResult.factorization_time + lapResult.solve_time, "total_time");
  // runtime_csv.addElementToRecord(lapResult.residual, "residual");
  // runtime_csv.addRecord();
  
  // Record for Hessian solver (iteration 0)
  runtime_csv.addElementToRecord(mesh_name, "mesh_name");
  runtime_csv.addElementToRecord(0, "iteration");
  runtime_csv.addElementToRecord(hessResult.ordering_type, "ordering_type");
  runtime_csv.addElementToRecord(args.solver_type, "solver_type");
  runtime_csv.addElementToRecord(hessResult.matrix_rows, "G_N");
  runtime_csv.addElementToRecord(hessResult.matrix_nnz, "G_NNZ");
  runtime_csv.addElementToRecord(hessResult.nd_levels, "nd_levels");
  runtime_csv.addElementToRecord(hessResult.patch_type, "patch_type");
  runtime_csv.addElementToRecord(hessResult.patch_size, "patch_size");
  runtime_csv.addElementToRecord(hessResult.ordering_patch_time, "patch_time");
  runtime_csv.addElementToRecord(hessResult.factor_nnz_ratio, "factor/matrix NNZ ratio");
  runtime_csv.addElementToRecord(hessResult.ordering_init_time, "ordering_init_time");
  runtime_csv.addElementToRecord(hessResult.ordering_compute_permutation_time, "ordering_time");
  runtime_csv.addElementToRecord(hessResult.ordering_integration_time, "ordering_integration_time");
  runtime_csv.addElementToRecord(hessResult.analysis_time, "analysis_time");
  runtime_csv.addElementToRecord(hessResult.factorization_time, "factorization_time");
  runtime_csv.addElementToRecord(hessResult.solve_time, "solve_time");
  runtime_csv.addElementToRecord(hessResult.residual, "residual");
  runtime_csv.addRecord();
  
  // Record for Curved Hessian solver (iteration 1)
  runtime_csv.addElementToRecord(mesh_name, "mesh_name");
  runtime_csv.addElementToRecord(1, "iteration");
  runtime_csv.addElementToRecord(curvedHessResult.ordering_type, "ordering_type");
  runtime_csv.addElementToRecord(args.solver_type, "solver_type");
  runtime_csv.addElementToRecord(curvedHessResult.matrix_rows, "G_N");
  runtime_csv.addElementToRecord(curvedHessResult.matrix_nnz, "G_NNZ");
  runtime_csv.addElementToRecord(curvedHessResult.nd_levels, "nd_levels");
  runtime_csv.addElementToRecord(curvedHessResult.patch_type, "patch_type");
  runtime_csv.addElementToRecord(curvedHessResult.patch_size, "patch_size");
  runtime_csv.addElementToRecord(curvedHessResult.ordering_patch_time, "patch_time");
  runtime_csv.addElementToRecord(curvedHessResult.factor_nnz_ratio, "factor/matrix NNZ ratio");
  runtime_csv.addElementToRecord(curvedHessResult.ordering_init_time, "ordering_init_time");
  runtime_csv.addElementToRecord(curvedHessResult.ordering_compute_permutation_time, "ordering_time");
  runtime_csv.addElementToRecord(curvedHessResult.ordering_integration_time, "ordering_integration_time");
  runtime_csv.addElementToRecord(curvedHessResult.analysis_time, "analysis_time");
  runtime_csv.addElementToRecord(curvedHessResult.factorization_time, "factorization_time");
  runtime_csv.addElementToRecord(curvedHessResult.solve_time, "solve_time");
  runtime_csv.addElementToRecord(curvedHessResult.residual, "residual");
  runtime_csv.addRecord();
  
  spdlog::info("CSV data exported to: {}.csv", args.output_csv_address);
  
  // Set up global pointers for polyscope callback
  g_zexact = &zexact;
  g_znoisy = &znoisy;
  // g_zl = &zl;
  g_zh = &zh;
  g_zch = &zch;
  
  // Log value ranges for debugging
  spdlog::info("Exact value range: [{}, {}]", zexact.minCoeff(), zexact.maxCoeff());
  spdlog::info("Noisy value range: [{}, {}]", znoisy.minCoeff(), znoisy.maxCoeff());
  // spdlog::info("Laplacian smoothed range: [{}, {}]", zl.minCoeff(), zl.maxCoeff());
  spdlog::info("Hessian smoothed range: [{}, {}]", zh.minCoeff(), zh.maxCoeff());
  spdlog::info("Curved Hessian smoothed range: [{}, {}]", zch.minCoeff(), zch.maxCoeff());
  
  // Initialize polyscope
  polyscope::init();
  
  // Register the surface mesh
  std::vector<std::array<double, 3>> vertices(V.rows());
  for (int i = 0; i < V.rows(); i++) {
      vertices[i] = {V(i, 0), V(i, 1), V(i, 2)};
  }
  std::vector<std::array<int, 3>> faces(F.rows());
  for (int i = 0; i < F.rows(); i++) {
      faces[i] = {F(i, 0), F(i, 1), F(i, 2)};
  }
  g_mesh = polyscope::registerSurfaceMesh("mesh", vertices, faces);
  g_mesh->setEdgeWidth(0.0);  // Hide mesh edges
  
  // Set the user callback for mode selection
  polyscope::state::userCallback = dataSmoothingCallback;
  
  // Initial visualization (show noisy data)
  g_current_section = 0;
  g_current_mode = 1;
  updateVisualization();
  
  std::cout << "Use the GUI to switch between different smoothing modes.\n";
  
  // Show the polyscope GUI
  polyscope::show();
  
  // Cleanup
  polyscope::state::userCallback = nullptr;
  g_zexact = nullptr;
  g_znoisy = nullptr;
  g_zl = nullptr;
  g_zh = nullptr;
  g_zch = nullptr;
  g_mesh = nullptr;

  return 0;
}
