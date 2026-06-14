#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "homa/solvers/LinSysSolver.h"
#include "homa/utils/check_valid_permutation.h"
#include "homa/ordering.h"
#include "homa/types.h"
#include "homa/utils/remove_diagonal.h"

#include "util.h"

#include <spdlog/spdlog.h>

namespace {

using SparseMatrix = Eigen::SparseMatrix<double>;

SparseMatrix expand_symmetric_storage(const SparseMatrix& raw)
{
    Eigen::Index lower = 0;
    Eigen::Index upper = 0;

    for (int c = 0; c < raw.outerSize(); ++c) {
        for (SparseMatrix::InnerIterator it(raw, c); it; ++it) {
            if (it.row() > it.col()) {
                ++lower;
            } else if (it.row() < it.col()) {
                ++upper;
            }
        }
    }

    SparseMatrix expanded =
        (lower >= upper) ? SparseMatrix(raw.selfadjointView<Eigen::Lower>())
                         : SparseMatrix(raw.selfadjointView<Eigen::Upper>());
    expanded.makeCompressed();
    return expanded;
}

SparseMatrix solver_matrix_for(homa::LinSysSolverType solver_type, const SparseMatrix& A)
{
    if (solver_type == homa::LinSysSolverType::CPU_MKL) {
        SparseMatrix lower = A.triangularView<Eigen::Lower>();
        lower.makeCompressed();
        return lower;
    }

    SparseMatrix solver_matrix = A;
    solver_matrix.makeCompressed();
    return solver_matrix;
}

template <typename Fn>
double time_step(homa::LinSysSolverType solver_type, Fn&& fn)
{
#ifdef USE_CUDSS
    if (solver_type == homa::LinSysSolverType::GPU_CUDSS) {
        GpuTimer timer;
        timer.start();
        fn();
        return timer.stop_ms();
    }
#endif

    using Clock = std::chrono::high_resolution_clock;
    const auto t0 = Clock::now();
    fn();
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

void warm_up_if_needed(homa::LinSysSolverType solver_type,
                       SparseMatrix&          solver_matrix,
                       Eigen::VectorXd&       rhs)
{
    if (solver_type != homa::LinSysSolverType::GPU_CUDSS) {
        return;
    }

#ifdef USE_CUDSS    
    std::unique_ptr<homa::LinSysSolver> solver(homa::LinSysSolver::create(solver_type));
    solver->setMatrix(solver_matrix.outerIndexPtr(),
                      solver_matrix.innerIndexPtr(),
                      solver_matrix.valuePtr(),
                      static_cast<int>(solver_matrix.rows()),
                      static_cast<int>(solver_matrix.nonZeros()));

    std::vector<int> empty;
    solver->ordering(empty, empty);
    solver->analyze_pattern();
    solver->factorize();
    Eigen::VectorXd sol;
    solver->solve(rhs, sol);
    cudaDeviceSynchronize();
#endif
}

StageTimes run_solver_path(homa::LinSysSolverType solver_type,
                           SparseMatrix&          solver_matrix,
                           const SparseMatrix&    residual_matrix,
                           Eigen::VectorXd&       rhs,
                           std::vector<int>&      perm,
                           std::vector<int>&      etree,
                           bool                   use_homa_ordering)
{
    StageTimes times;
        
    std::unique_ptr<homa::LinSysSolver> solver(homa::LinSysSolver::create(solver_type));
    solver->setMatrix(solver_matrix.outerIndexPtr(),
                      solver_matrix.innerIndexPtr(),
                      solver_matrix.valuePtr(),
                      static_cast<int>(solver_matrix.rows()),
                      static_cast<int>(solver_matrix.nonZeros()));

    std::vector<int> empty;
    if (solver_type == homa::LinSysSolverType::GPU_CUDSS) {
        times.reorder_ms = time_step(solver_type, [&]() {
            solver->ordering(use_homa_ordering ? perm : empty,
                             use_homa_ordering ? etree : empty);
        });
        times.analysis_ms = time_step(solver_type, [&]() {
            solver->analyze_pattern();
        });
    } else {
        times.analysis_ms = time_step(solver_type, [&]() {
            solver->analyze_pattern(use_homa_ordering ? perm : empty,
                                    use_homa_ordering ? etree : empty);
        });
    }

    times.factorize_ms = time_step(solver_type, [&]() { solver->factorize(); });

    Eigen::VectorXd sol;
    times.solve_ms = time_step(solver_type, [&]() { solver->solve(rhs, sol); });
    times.residual = (rhs - residual_matrix * sol).norm();

    return times;
}

} // namespace

int main(int argc, char* argv[])
{
    std::string input_matrix;
    std::string solver_name = "cholmod";
    int         patch_size  = 512;
    std::string output_json;

    CLI::App app{"Homa Matrix Market linear solver example"};
    app.add_option("-i,--input", input_matrix, "Input matrix (.mtx)")->required();
    app.add_option("-s,--solver", solver_name, "Solver backend: cholmod, mkl, cudss");
    app.add_option("-p,--patch_size", patch_size, "Patch size (default: 512)");
    app.add_option("-o,--out", output_json,
        "Optional JSON file to write benchmark results to (no output if empty)");
    CLI11_PARSE(app, argc, argv);

    SparseMatrix raw;
    if (!Eigen::loadMarket(raw, input_matrix)) {
        std::cerr << "Failed to read Matrix Market file: " << input_matrix << "\n";
        return 1;
    }

    if (raw.rows() != raw.cols()) {
        std::cerr << "Matrix must be square: " << raw.rows() << "x" << raw.cols() << "\n";
        return 1;
    }

    SparseMatrix A = is_matrix_market_symmetric(input_matrix) ? expand_symmetric_storage(raw) : raw;
    A.makeCompressed();

    homa::LinSysSolverType solver_type = homa::LinSysSolverType::CPU_CHOLMOD;
    try {
        solver_type = solver_type_from_name(solver_name);
    } catch (const std::invalid_argument& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }

    SparseMatrix solver_matrix = solver_matrix_for(solver_type, A);

    spdlog::info("Matrix: {}x{}, {} non-zeros", A.rows(), A.cols(), A.nonZeros());
    if (solver_matrix.nonZeros() != A.nonZeros()) {
        spdlog::info("Solver matrix: {} non-zeros", solver_matrix.nonZeros());
    }

    const int n = static_cast<int>(A.rows());
    std::vector<int> Gp, Gi;
    homa::remove_diagonal(n, A.outerIndexPtr(), A.innerIndexPtr(), Gp, Gi);

    Eigen::VectorXd rhs = Eigen::VectorXd::Random(n);

    warm_up_if_needed(solver_type, solver_matrix, rhs);

    std::vector<int> empty_perm, empty_etree;
    StageTimes       def = run_solver_path(
        solver_type, solver_matrix, A, rhs, empty_perm, empty_etree, false);

    using Clock = std::chrono::high_resolution_clock;

    homa::Options opts;
    opts.use_gpu             = true;
    opts.patch_size          = patch_size;
    opts.use_patch_separator = true;
    opts.compute_etree       = (solver_type == homa::LinSysSolverType::GPU_CUDSS);
    opts.local_method        = homa::Options::LocalMethod::AMD;

    const auto t0 = Clock::now();
    homa::OrderingResult ord = homa::compute_ordering(n, Gp.data(), Gi.data(), opts);
    const double homa_ordering_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    if (!homa::check_valid_permutation(ord.perm.data(), ord.perm.size())) {
        spdlog::error("HOMA permutation is invalid!");
    }

    StageTimes homa =
        run_solver_path(solver_type, solver_matrix, A, rhs, ord.perm, ord.etree, true);
    homa.ordering_ms = homa_ordering_ms;

    double def_total_ms  = def.ordering_ms  + def.factorize_ms  + def.solve_ms;
    double homa_total_ms = homa.ordering_ms + homa.factorize_ms + homa.solve_ms;
    if (solver_type == homa::LinSysSolverType::GPU_CUDSS) {
        def_total_ms  += def.reorder_ms  + def.analysis_ms;
        homa_total_ms += homa.reorder_ms + homa.analysis_ms;
    } else {
        def_total_ms  += def.analysis_ms;
        homa_total_ms += homa.analysis_ms;
    }

    std::cout << "\n=== " << solver_display_name(solver_type) << " Matrix Results ===\n";
    std::cout << std::left << std::fixed << std::setprecision(3) << std::setw(18) << "" << std::setw(16) << "Solver-default" << "HOMA\n"
              << std::setw(18) << "Ordering (ms) :" << std::setw(16) << "---" << homa_ordering_ms << "\n";

    if (solver_type == homa::LinSysSolverType::GPU_CUDSS) {
        std::cout << std::setw(18) << "Reorder  (ms) :" << std::setw(16) << def.reorder_ms << homa.reorder_ms << "\n"
                  << std::setw(18) << "Symbolic (ms) :" << std::setw(16) << def.analysis_ms << homa.analysis_ms << "\n";
    } else {
        std::cout << std::setw(18) << "Analysis (ms) :" << std::setw(16) << def.analysis_ms << homa.analysis_ms << "\n";
    }

    std::cout << std::setw(18) << "Factorize (ms):" << std::setw(16) << def.factorize_ms << homa.factorize_ms << "\n"
              << std::setw(18) << "Solve (ms)    :" << std::setw(16) << def.solve_ms << homa.solve_ms << "\n"
              << std::setw(18) << "Total (ms)    :" << std::setw(16) << def_total_ms << homa_total_ms << " (speedup =" << def_total_ms / homa_total_ms << ")\n"
              << std::setw(18) << "Residual      :" << std::setw(16) << def.residual << homa.residual << "\n";

    if (!output_json.empty()) {
        BenchmarkRecord rec;
        rec.matrix_path      = input_matrix;
        rec.solver_name      = solver_display_name(solver_type);
        rec.n                = static_cast<int>(A.rows());
        rec.nnz              = static_cast<long long>(A.nonZeros());
        rec.patch_size       = patch_size;
        rec.default_times    = def;
        rec.homa_times       = homa;
        rec.default_total_ms = def_total_ms;
        rec.homa_total_ms    = homa_total_ms;
        write_results_json(output_json, rec);
    }

    return 0;
}
