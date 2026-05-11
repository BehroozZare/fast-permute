//
// Inverse Rendering Benchmark: For each row of counts.csv in the dataset
// folder, if n_vertices exceeds the threshold, load the mesh, scalar system
// matrix, and dense fwd/bwd RHS blocks, compute ordering -> symbolic ->
// factorization once, then repeat the fwd multi-RHS solve fwd_count times
// and the bwd multi-RHS solve bwd_count times. One CSV row is written per
// direction per entry.
//
// The problem is scalar (DIM=1): the graph used for ordering is simply the
// sparsity pattern of the Hessian, so no ParthAPI-style DIM compression is
// needed.
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
#include "remove_diagonal.h"
#include "SPD_cot_matrix.h"
#include "utils/inverse_rendering_prep_benchmark.h"
#include "csv_utils.h"
#include "save_vector.h"
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
    int min_vertices = 50000;
    std::string output_csv_address =
        "/home/behrooz/Desktop/Last_Project/gpu_ordering/output/Apps/inverse_rendering";
    std::string solver_type         = "CUDSS";
    std::string ordering_type       = "DEFAULT";
    std::string patch_type          = "rxmesh";
    std::string check_point_address =
        "/media/behrooz/FarazHard/Last_Project/Apps/inverse_rendering/nefertiti";
    bool use_gpu = false;

    CLIArgs(int argc, char* argv[])
    {
        CLI::App app{"Inverse Rendering Benchmark - Per-entry scalar factorization "
                     "with fwd/bwd multi-RHS solve repeats"};
        app.add_option("-a,--ordering", ordering_type,
                       "Ordering type: DEFAULT, PATCH_ORDERING, PARTH");
        app.add_option("-s,--solver", solver_type, "Solver type: CUDSS, MKL");
        app.add_option("-o,--output", output_csv_address,
                       "Output CSV file path (without .csv extension)");
        app.add_option("-p,--patch_type", patch_type,
                       "Patch type for PATCH_ORDERING: rxmesh, metis");
        app.add_option("-b,--binary_level", binary_level,
                       "Binary level for nested dissection tree");
        app.add_option("-k,--check_point_address", check_point_address,
                       "Folder containing inverse rendering data (with counts.csv)");
        app.add_option("-m,--min_vertices", min_vertices,
                       "Skip entries whose n_vertices is <= this threshold");
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
    std::filesystem::create_directories(
        std::filesystem::path(args.output_csv_address).parent_path());

    spdlog::info("=== Inverse Rendering Benchmark ===");
    spdlog::info("Loading benchmark data from: {}", args.check_point_address);
    spdlog::info("Output CSV address: {}", args.output_csv_address);
    spdlog::info("Solver: {}", args.solver_type);
    spdlog::info("Ordering: {}", args.ordering_type);
    spdlog::info("Min n_vertices (strict >): {}", args.min_vertices);

    // ========== Load counts.csv entries ==========
    std::vector<homa::InverseRenderingEntry> entries;
    homa::prepare_inverse_rendering_benchmark_data(
        args.check_point_address, entries);
    if (entries.empty()) {
        spdlog::error("No entries found. Exiting.");
        return 1;
    }
    spdlog::info("Parsed {} entries from counts.csv", entries.size());

    // Dataset name = folder name (e.g. "nefertiti", "dragon")
    std::string dataset_name =
        std::filesystem::path(args.check_point_address).filename().string();

    // ========== Prepare CSV output (slim schema + direction / rhs_count) ==========
    std::vector<std::string> header;
    header.emplace_back("mesh_name");
    header.emplace_back("iteration");
    header.emplace_back("direction");
    header.emplace_back("rhs_count");
    header.emplace_back("ordering_type");
    header.emplace_back("solver_type");
    header.emplace_back("G_N");
    header.emplace_back("G_NNZ");
    header.emplace_back("nd_levels");
    header.emplace_back("patch_type");
    header.emplace_back("patch_size");
    header.emplace_back("patch_time");
    header.emplace_back("node_to_patch_time");
    header.emplace_back("decompose_time");
    header.emplace_back("local_permute_time");
    header.emplace_back("assemble_time");
    header.emplace_back("factor/matrix NNZ ratio");
    header.emplace_back("ordering_init_time");
    header.emplace_back("ordering_time");
    header.emplace_back("ordering_integration_time");
    header.emplace_back("analysis_time");
    header.emplace_back("factorization_time");
    header.emplace_back("solve_time");
    header.emplace_back("residual");

    homa::CSVManager runtime_csv(args.output_csv_address,
                                          "inverse_rendering_benchmark",
                                          header, false);

    // ========== Iterate entries ==========
    for (const auto& entry : entries) {
        if (entry.n_vertices <= args.min_vertices) {
            spdlog::info("Skipping entry idx={} (n_vertices={} <= {})",
                         entry.idx, entry.n_vertices, args.min_vertices);
            continue;
        }

        std::string mesh_stem = std::filesystem::path(entry.mesh_file).stem().string();
        std::string mesh_name = dataset_name + "/" + mesh_stem;

        spdlog::info("=== Processing entry idx={} mesh={} (n_vertices={}, "
                     "fwd_count={}, bwd_count={}) ===",
                     entry.idx, mesh_name, entry.n_vertices,
                     entry.fwd_count, entry.bwd_count);

        // ----- Load mesh -----
        Eigen::MatrixXd OV;
        Eigen::MatrixXi OF;
        if (!igl::read_triangle_mesh(entry.mesh_file, OV, OF)) {
            spdlog::error("Failed to read mesh from: {}", entry.mesh_file);
            continue;
        }
        spdlog::info("Loaded mesh: {} vertices, {} faces", OV.rows(), OF.rows());

        // ----- Load matrix -----
        Eigen::SparseMatrix<double> hessian;
        if (!Eigen::loadMarket(hessian, entry.matrix_file)) {
            spdlog::error("Failed to load matrix: {}", entry.matrix_file);
            continue;
        }
        spdlog::info("Matrix: {} x {}, NNZ: {}",
                     hessian.rows(), hessian.cols(), hessian.nonZeros());

        // ----- Load fwd/bwd dense RHS blocks -----
        Eigen::MatrixXd rhs_fwd, rhs_bwd;
        if (!Eigen::loadMarketDense(rhs_fwd, entry.fwd_rhs_file)) {
            spdlog::error("Failed to load fwd RHS: {}", entry.fwd_rhs_file);
            continue;
        }
        if (!Eigen::loadMarketDense(rhs_bwd, entry.bwd_rhs_file)) {
            spdlog::error("Failed to load bwd RHS: {}", entry.bwd_rhs_file);
            continue;
        }
        spdlog::info("RHS fwd: {} x {}, bwd: {} x {}",
                     rhs_fwd.rows(), rhs_fwd.cols(),
                     rhs_bwd.rows(), rhs_bwd.cols());
        if (rhs_fwd.rows() != hessian.rows() || rhs_bwd.rows() != hessian.rows()) {
            spdlog::error("RHS row count does not match matrix size; skipping entry.");
            continue;
        }

        // ----- Diagnostic: compare hessian sparsity pattern vs cotangent Laplacian -----
        Eigen::SparseMatrix<double> cot_L;
        homa::computeSPD_cot_matrix(OV, OF, cot_L);
        if (cot_L.rows() != hessian.rows()) {
            spdlog::warn("cot_L rows ({}) != hessian rows ({}); skipping "
                         "sparsity comparison.",
                         cot_L.rows(), hessian.rows());
        } else {
            bool patterns_match = homa::compare_sparsity_no_diagonal(
                static_cast<int>(hessian.rows()),
                hessian.outerIndexPtr(), hessian.innerIndexPtr(),
                cot_L.outerIndexPtr(), cot_L.innerIndexPtr(),
                "hessian", "cot_L");
            spdlog::info("Sparsity pattern identical (hessian vs cot_L): {}",
                         patterns_match);
        }

        // TEST: replace the loaded hessian with the cotangent Laplacian so
        // the rest of the pipeline (ordering, factorization, solve, residual)
        // runs against cot_L instead of the input matrix.
        if (cot_L.rows() == hessian.rows()) {
            spdlog::warn("TEST OVERRIDE: using cot_L in place of loaded hessian "
                         "for ordering/factor/solve.");
            hessian = cot_L;
        }

        // ----- Initialize solver -----
        homa::LinSysSolver* solver = nullptr;
        if (args.solver_type == "CUDSS") {
            solver = homa::LinSysSolver::create(
                homa::LinSysSolverType::GPU_CUDSS);
        } else if (args.solver_type == "MKL") {
            solver = homa::LinSysSolver::create(
                homa::LinSysSolverType::CPU_MKL);
        } else {
            spdlog::error("Unknown solver type: {}", args.solver_type);
            return 1;
        }
        assert(solver != nullptr);

        // ----- Graph for ordering (scalar: graph == hessian sparsity) -----
        int* Gp    = hessian.outerIndexPtr();
        int* Gi    = hessian.innerIndexPtr();
        int  G_N   = static_cast<int>(hessian.rows());
        int  G_NNZ = static_cast<int>(hessian.nonZeros());

        // ----- Compute ordering -----
        std::vector<int> matrix_perm;
        std::vector<int> matrix_etree;
        long int ordering_time      = -1;
        long int ordering_init_time = -1;
        homa::Ordering* ordering = nullptr;
        bool entry_failed = false;

        if (args.ordering_type == "PATCH_ORDERING") {
            // RXMesh registers a global spdlog logger named "RXMesh" on
            // construction; dropping it first allows repeated creation in
            // the same process without throwing.
            spdlog::drop("RXMesh");
            ordering = homa::Ordering::create(
                homa::DEMO_ORDERING_TYPE::PATCH_ORDERING);
            ordering->setOptions({
                {"use_gpu", args.use_gpu ? "1" : "0"},
                {"patch_type", args.patch_type},
                {"binary_level", std::to_string(args.binary_level)}
            });
            if (ordering->needsMesh()) {
                if (OV.rows() == 0 || OF.rows() == 0) {
                    spdlog::error("PATCH_ORDERING needs mesh but mesh is empty.");
                    entry_failed = true;
                } else {
                    ordering->setMesh(OV.data(), OV.rows(), OV.cols(),
                                      OF.data(), OF.rows(), OF.cols());
                }
            }
            if (!entry_failed) {
                ordering->setGraph(Gp, Gi, G_N, G_NNZ);

                auto ordering_init_start = std::chrono::high_resolution_clock::now();
                ordering->init();
                auto ordering_init_end = std::chrono::high_resolution_clock::now();
                ordering_init_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    ordering_init_end - ordering_init_start).count();
                spdlog::info("Ordering init time: {} ms", ordering_init_time);

                bool compute_etree = (args.solver_type == "CUDSS");
                auto ordering_start = std::chrono::high_resolution_clock::now();
                ordering->compute_permutation(matrix_perm, matrix_etree, compute_etree);
                auto ordering_end = std::chrono::high_resolution_clock::now();
                ordering_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    ordering_end - ordering_start).count();
                spdlog::info("Ordering time: {} ms", ordering_time);
            }
        } else if (args.ordering_type == "PARTH") {
            ordering = homa::Ordering::create(
                homa::DEMO_ORDERING_TYPE::PARTH);
            ordering->setOptions({{"binary_level", std::to_string(args.binary_level)}});
            ordering->setGraph(Gp, Gi, G_N, G_NNZ);

            auto ordering_init_start = std::chrono::high_resolution_clock::now();
            ordering->init();
            auto ordering_init_end = std::chrono::high_resolution_clock::now();
            ordering_init_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                ordering_init_end - ordering_init_start).count();
            spdlog::info("Ordering init time: {} ms", ordering_init_time);

            bool compute_etree = (args.solver_type == "CUDSS");
            auto ordering_start = std::chrono::high_resolution_clock::now();
            ordering->compute_permutation(matrix_perm, matrix_etree, compute_etree);
            auto ordering_end = std::chrono::high_resolution_clock::now();
            ordering_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                ordering_end - ordering_start).count();
            spdlog::info("Ordering time: {} ms", ordering_time);
        } else if (args.ordering_type == "DEFAULT") {
            spdlog::info("Using DEFAULT ordering (solver's built-in).");
        } else {
            spdlog::error("Unknown ordering type: {}", args.ordering_type);
            entry_failed = true;
        }

        if (!entry_failed && !matrix_perm.empty()) {
            if (!homa::check_valid_permutation(matrix_perm.data(),
                                                       matrix_perm.size())) {
                spdlog::error("Permutation invalid; skipping entry.");
                entry_failed = true;
            } else {
                spdlog::info("Permutation validated: size = {}", matrix_perm.size());
            }
        }

        if (entry_failed) {
            if (ordering != nullptr) delete ordering;
            delete solver;
            continue;
        }

        // ----- Factor NNZ (symbolic, once) -----
        long int factor_nnz = -1;
        if (!matrix_perm.empty()) {
            factor_nnz = homa::get_factor_nnz(hessian.outerIndexPtr(),
                                                       hessian.innerIndexPtr(),
                                                       hessian.valuePtr(),
                                                       hessian.rows(),
                                                       hessian.nonZeros(),
                                                       matrix_perm);
            spdlog::info("Factor NNZ ratio: {:.4f}",
                         factor_nnz * 1.0 / hessian.nonZeros());
            if (ordering != nullptr) {
                solver->ordering_name = ordering->typeStr();
            }
        }

        // ----- setMatrix + ordering + analyze_pattern + factorize (once) -----
        Eigen::SparseMatrix<double> lower_hessian;
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

        auto ordering_integration_start = std::chrono::high_resolution_clock::now();
        solver->ordering(matrix_perm, matrix_etree);
        auto ordering_integration_end = std::chrono::high_resolution_clock::now();
        long int ordering_integration_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                ordering_integration_end - ordering_integration_start).count();
        spdlog::info("Ordering integration time: {} ms", ordering_integration_time);

        auto analysis_start = std::chrono::high_resolution_clock::now();
        solver->analyze_pattern(matrix_perm, matrix_etree);
        if (args.solver_type == "CUDSS") CUDA_SYNC_CHECK();
        auto analysis_end = std::chrono::high_resolution_clock::now();
        long int analysis_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            analysis_end - analysis_start).count();
        spdlog::info("Symbolic analysis time: {} ms", analysis_time);

        auto factor_start = std::chrono::high_resolution_clock::now();
        solver->factorize();
        if (args.solver_type == "CUDSS") CUDA_SYNC_CHECK();
        auto factor_end = std::chrono::high_resolution_clock::now();
        long int factorization_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            factor_end - factor_start).count();
        spdlog::info("Factorization time: {} ms", factorization_time);

        // ----- Full hessian for residual computation -----
        Eigen::SparseMatrix<double> hessian_full =
            hessian.selfadjointView<Eigen::Lower>();

        // Helper lambda: run N repeated multi-RHS solves, accumulate total time
        // and return (accumulated_solve_ms, final_residual).
        auto run_direction = [&](Eigen::MatrixXd& rhs_block, int count)
            -> std::pair<long int, double>
        {
            long int accumulated_ms = 0;
            Eigen::MatrixXd result;
            for (int r = 0; r < count; ++r) {
                auto solve_start = std::chrono::high_resolution_clock::now();
                solver->solve(rhs_block, result);
                if (args.solver_type == "CUDSS") CUDA_SYNC_CHECK();
                auto solve_end = std::chrono::high_resolution_clock::now();
                accumulated_ms +=
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        solve_end - solve_start).count();
            }
            double residual = 0.0;
            if (count > 0) {
                double rhs_norm = rhs_block.norm();
                double num = (rhs_block - hessian_full * result).norm();
                residual = (rhs_norm > 0.0) ? (num / rhs_norm) : num;
            }
            return {accumulated_ms, residual};
        };

        spdlog::info("Running fwd direction: {} multi-RHS solves", entry.fwd_count);
        auto [fwd_solve_ms, fwd_residual] = run_direction(rhs_fwd, entry.fwd_count);
        spdlog::info("fwd total solve time: {} ms, residual: {}",
                     fwd_solve_ms, fwd_residual);

        spdlog::info("Running bwd direction: {} multi-RHS solves", entry.bwd_count);
        auto [bwd_solve_ms, bwd_residual] = run_direction(rhs_bwd, entry.bwd_count);
        spdlog::info("bwd total solve time: {} ms, residual: {}",
                     bwd_solve_ms, bwd_residual);

        // ----- Pre-gather shared fields -----
        int nd_levels = matrix_etree.empty()
                            ? 0
                            : static_cast<int>(std::log2(matrix_etree.size() + 1));

        std::string patch_type_field = "";
        double patch_size_field = 0.0;
        double patch_time_field = 0.0;
        double node_to_patch_field = 0.0;
        double decompose_field = 0.0;
        double local_permute_field = 0.0;
        double assemble_field = 0.0;
        if (args.ordering_type == "PATCH_ORDERING" && ordering != nullptr) {
            std::map<std::string, double> stat;
            ordering->getStatistics(stat);
            patch_type_field     = args.patch_type;
            patch_size_field     = stat["patch_size"];
            patch_time_field     = stat["patching_time"];
            node_to_patch_field  = stat["node_to_patch_time"];
            decompose_field      = stat["decompose_time"];
            local_permute_field  = stat["local_permute_time"];
            assemble_field       = stat["assemble_time"];
        }

        double factor_ratio = (factor_nnz > 0)
            ? (factor_nnz * 1.0 / hessian.nonZeros())
            : (solver->getFactorNNZ() * 1.0 / hessian.nonZeros());

        auto write_row = [&](const std::string& direction,
                             int                 rhs_count,
                             long int            solve_time_ms,
                             double              residual,
                             long int            factorization_time_row,
                             long int            ordering_init_time_row,
                             long int            ordering_time_row,
                             long int            ordering_integration_time_row,
                             long int            analysis_time_row)
        {
            runtime_csv.addElementToRecord(mesh_name, "mesh_name");
            runtime_csv.addElementToRecord(entry.idx, "iteration");
            runtime_csv.addElementToRecord(direction, "direction");
            runtime_csv.addElementToRecord(rhs_count, "rhs_count");
            runtime_csv.addElementToRecord(args.ordering_type, "ordering_type");
            runtime_csv.addElementToRecord(args.solver_type, "solver_type");
            runtime_csv.addElementToRecord(static_cast<int>(hessian.rows()), "G_N");
            runtime_csv.addElementToRecord(static_cast<int>(hessian.nonZeros()), "G_NNZ");
            runtime_csv.addElementToRecord(nd_levels, "nd_levels");
            runtime_csv.addElementToRecord(patch_type_field, "patch_type");
            runtime_csv.addElementToRecord(patch_size_field, "patch_size");
            runtime_csv.addElementToRecord(patch_time_field, "patch_time");
            runtime_csv.addElementToRecord(node_to_patch_field, "node_to_patch_time");
            runtime_csv.addElementToRecord(decompose_field, "decompose_time");
            runtime_csv.addElementToRecord(local_permute_field, "local_permute_time");
            runtime_csv.addElementToRecord(assemble_field, "assemble_time");
            runtime_csv.addElementToRecord(factor_ratio, "factor/matrix NNZ ratio");
            runtime_csv.addElementToRecord(ordering_init_time_row, "ordering_init_time");
            runtime_csv.addElementToRecord(ordering_time_row, "ordering_time");
            runtime_csv.addElementToRecord(ordering_integration_time_row,
                                           "ordering_integration_time");
            runtime_csv.addElementToRecord(analysis_time_row, "analysis_time");
            runtime_csv.addElementToRecord(factorization_time_row, "factorization_time");
            runtime_csv.addElementToRecord(solve_time_ms, "solve_time");
            runtime_csv.addElementToRecord(residual, "residual");
            runtime_csv.addRecord();
        };

        // fwd row carries the ordering/analysis/factorization costs.
        write_row("fwd", entry.fwd_count, fwd_solve_ms, fwd_residual,
                  factorization_time,
                  ordering_init_time, ordering_time,
                  ordering_integration_time, analysis_time);

        // bwd row zeros out the one-time setup costs so they are not double-counted.
        write_row("bwd", entry.bwd_count, bwd_solve_ms, bwd_residual,
                  0, 0, 0, 0, 0);

        // ----- Cleanup entry-scoped resources -----
        if (ordering != nullptr) delete ordering;
        delete solver;
    }

    spdlog::info("=== Inverse Rendering Benchmark Complete ===");
    spdlog::info("Results saved to: {}", args.output_csv_address);
    return 0;
}
