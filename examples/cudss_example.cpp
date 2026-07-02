#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <cuda_runtime.h>

#include "homa/utils/SPD_cot_matrix.h"
#include <homa/solvers/LinSysSolver.h>
#include "homa/homa.h"
#include "homa/types.h"
#include "homa/utils/remove_diagonal.h"
#include "homa/utils/check_valid_permutation.h"
#include "homa/utils/cuda_error_handler.h"
#include <cuda_runtime.h>
#include <igl/read_triangle_mesh.h>
#include <spdlog/spdlog.h>

#include "util.h"

int main(int argc, char* argv[])
{
    std::string input_mesh;
    int         patch_size       = 512;
    int         runs             = 1;
    std::string separator_method = "auto";

    CLI::App app{"Homa cuDSS example"};
    app.add_option("-i,--input", input_mesh, "Input mesh (.obj)")->required();
    app.add_option("-p,--patch_size", patch_size, "Patch size (default: 512)");
    app.add_option("-r,--runs", runs, "Number of timed runs (default: 1)")
        ->check(CLI::PositiveNumber);
    app.add_option("--separator-method",
                   separator_method,
                   "Separator strategy: auto (heuristic, default), quotient, "
                   "or direct (METIS)")
        ->transform(CLI::IsMember({"auto", "quotient", "direct", "metis"},
                                  CLI::ignore_case));
    CLI11_PARSE(app, argc, argv);

    const homa::Options::SeparatorMethod sep_method =
        separator_method_from_name(separator_method);

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    if (!igl::read_triangle_mesh(input_mesh, V, F)) {
        std::cerr << "Failed to read mesh: " << input_mesh << "\n";
        return 1;
    }

    Eigen::SparseMatrix<double> L;
    homa::computeSPD_cot_matrix(V, F, L);
    spdlog::info(
        "Matrix: {}x{}, {} non-zeros", L.rows(), L.cols(), L.nonZeros());

    int              n = static_cast<int>(L.rows());
    std::vector<int> Gp, Gi;
    homa::remove_diagonal(n, L.outerIndexPtr(), L.innerIndexPtr(), Gp, Gi);

    Eigen::VectorXd rhs = Eigen::VectorXd::Random(n);

    using Clock = std::chrono::high_resolution_clock;
    auto cpu_ms = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    GpuTimer gpu_timer;

    // --- Warm-up: pay CUDA/cuDSS one-time init costs before any timed run ---
    {
        std::unique_ptr<homa::LinSysSolverD> solver(
            homa::LinSysSolverD::create(homa::LinSysSolverType::GPU_CUDSS));
        solver->setMatrix(L.outerIndexPtr(),
                          L.innerIndexPtr(),
                          L.valuePtr(),
                          n,
                          L.nonZeros());
        std::vector<int> empty;
        solver->ordering(empty, empty);
        solver->analyze_pattern();
        solver->factorize();
        Eigen::VectorXd sol;
        solver->solve(rhs, sol);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // --- Solver-default path (cuDSS internal reordering + symbolic
    // factorization) ---
    const int total_runs     = std::max(1, runs);
    float     def_reorder_ms = 0.0f, def_symbolic_ms = 0.0f;
    float     def_factorize_ms = 0.0f, def_solve_ms = 0.0f;
    double    def_residual = 0.0;
    for (int r = 0; r < total_runs; ++r) {
        std::unique_ptr<homa::LinSysSolverD> solver(
            homa::LinSysSolverD::create(homa::LinSysSolverType::GPU_CUDSS));
        solver->setMatrix(L.outerIndexPtr(),
                          L.innerIndexPtr(),
                          L.valuePtr(),
                          n,
                          L.nonZeros());

        std::vector<int> empty;

        gpu_timer.start();
        solver->ordering(empty, empty);  // cuDSS REORDERING phase (default)
        def_reorder_ms += gpu_timer.stop_ms();

        gpu_timer.start();
        solver->analyze_pattern();  // cuDSS SYMBOLIC_FACTORIZATION phase
        def_symbolic_ms += gpu_timer.stop_ms();

        gpu_timer.start();
        solver->factorize();
        def_factorize_ms += gpu_timer.stop_ms();

        Eigen::VectorXd sol;
        gpu_timer.start();
        solver->solve(rhs, sol);
        def_solve_ms += gpu_timer.stop_ms();
        def_residual = (rhs - L * sol).norm();
    }
    def_reorder_ms /= total_runs;
    def_symbolic_ms /= total_runs;
    def_factorize_ms /= total_runs;
    def_solve_ms /= total_runs;

    // --- HOMA path ---
    double homa_ordering_ms = 0.0;
    float  homa_reorder_ms = 0.0f, homa_symbolic_ms = 0.0f;
    float  homa_factorize_ms = 0.0f, homa_solve_ms = 0.0f;
    double homa_residual = 0.0;
    {
        homa::Options opts;
        opts.use_gpu    = true;
        opts.patch_size = patch_size;
        opts.compute_etree =
            true;  // cuDSS requires etree alongside permutation
        opts.local_method     = homa::Options::LocalMethod::AMD;
        opts.separator_method = sep_method;

        homa::OrderingResult ord;
        for (int r = 0; r < total_runs; ++r) {
            auto t0 = Clock::now();
            ord     = homa::compute_ordering(n, Gp.data(), Gi.data(), opts);
            homa_ordering_ms += cpu_ms(t0, Clock::now());
        }
        homa_ordering_ms /= total_runs;

        if (!homa::check_valid_permutation(ord.perm.data(), ord.perm.size())) {
            spdlog::error("HOMA permutation is invalid!");
        }

        for (int r = 0; r < total_runs; ++r) {
            std::unique_ptr<homa::LinSysSolverD> solver(
                homa::LinSysSolverD::create(homa::LinSysSolverType::GPU_CUDSS));
            solver->setMatrix(L.outerIndexPtr(),
                              L.innerIndexPtr(),
                              L.valuePtr(),
                              n,
                              L.nonZeros());

            gpu_timer.start();
            solver->ordering(ord.perm, ord.etree);
            homa_reorder_ms += gpu_timer.stop_ms();

            gpu_timer.start();
            solver->analyze_pattern();  // cuDSS SYMBOLIC_FACTORIZATION phase
            homa_symbolic_ms += gpu_timer.stop_ms();

            gpu_timer.start();
            solver->factorize();
            homa_factorize_ms += gpu_timer.stop_ms();

            Eigen::VectorXd sol;
            gpu_timer.start();
            solver->solve(rhs, sol);
            homa_solve_ms += gpu_timer.stop_ms();
            homa_residual = (rhs - L * sol).norm();
        }
        homa_reorder_ms /= total_runs;
        homa_symbolic_ms /= total_runs;
        homa_factorize_ms /= total_runs;
        homa_solve_ms /= total_runs;
    }

    std::cout << "\n=== cuDSS Results ===\n";
    float def_total_ms =
        def_reorder_ms + def_symbolic_ms + def_factorize_ms + def_solve_ms;
    float homa_total_ms = homa_ordering_ms + homa_reorder_ms +
                          homa_symbolic_ms + homa_factorize_ms + homa_solve_ms;
    std::cout << std::left << std::fixed << std::setprecision(3)
              << std::setw(18) << "" << std::setw(16) << "Solver-default"
              << "HOMA\n"
              << std::setw(18) << "Ordering (ms) :" << std::setw(16) << "---"
              << homa_ordering_ms << "\n"
              << std::setw(18) << "Reorder  (ms) :" << std::setw(16)
              << def_reorder_ms << homa_reorder_ms << "\n"
              << std::setw(18) << "Symbolic (ms) :" << std::setw(16)
              << def_symbolic_ms << homa_symbolic_ms << "\n"
              << std::setw(18) << "Factorize (ms):" << std::setw(16)
              << def_factorize_ms << homa_factorize_ms << "\n"
              << std::setw(18) << "Solve (ms)    :" << std::setw(16)
              << def_solve_ms << homa_solve_ms << "\n"
              << std::setw(18) << "Total (ms)    :" << std::setw(16)
              << def_total_ms << homa_total_ms
              << " (speedup =" << def_total_ms / homa_total_ms << ")\n"
              << std::setw(18) << "Residual      :" << std::setw(16)
              << def_residual << homa_residual << "\n";

    return 0;
}
