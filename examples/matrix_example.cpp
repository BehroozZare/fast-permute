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
#include <type_traits>
#include <utility>
#include <vector>

#include "homa/solvers/LinSysSolver.h"
#include "homa/utils/check_valid_permutation.h"
#include "homa/homa.h"
#include "homa/types.h"
#include "homa/utils/remove_diagonal.h"

#include "util.h"

#include <spdlog/spdlog.h>

namespace {

template <class Scalar>
using SparseMat = Eigen::SparseMatrix<Scalar>;

template <class Scalar>
using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

template <class Scalar>
SparseMat<Scalar> expand_symmetric_storage(const SparseMat<Scalar>& raw)
{
    Eigen::Index lower = 0;
    Eigen::Index upper = 0;

    for (int c = 0; c < raw.outerSize(); ++c) {
        for (typename SparseMat<Scalar>::InnerIterator it(raw, c); it; ++it) {
            if (it.row() > it.col()) {
                ++lower;
            } else if (it.row() < it.col()) {
                ++upper;
            }
        }
    }

    SparseMat<Scalar> expanded =
        (lower >= upper) ?
            SparseMat<Scalar>(raw.template selfadjointView<Eigen::Lower>()) :
            SparseMat<Scalar>(raw.template selfadjointView<Eigen::Upper>());
    expanded.makeCompressed();
    return expanded;
}

template <class Scalar>
SparseMat<Scalar> make_spd_from_pattern(const SparseMat<Scalar>& raw)
{
    const int                        n = static_cast<int>(raw.rows());
    std::vector<std::pair<int, int>> edges;

    for (int c = 0; c < raw.outerSize(); ++c) {
        for (typename SparseMat<Scalar>::InnerIterator it(raw, c); it; ++it) {
            const int r = static_cast<int>(it.row());
            if (r == c) {
                continue;
            }
            edges.emplace_back(r, c);
            edges.emplace_back(c, r);
        }
    }

    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    std::vector<int> degree(n, 0);
    for (const auto& edge : edges) {
        ++degree[edge.first];
    }

    std::vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(edges.size() + static_cast<size_t>(n));
    for (const auto& edge : edges) {
        triplets.emplace_back(edge.first, edge.second, Scalar(-1));
    }
    for (int i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, static_cast<Scalar>(degree[i] + 1));
    }

    SparseMat<Scalar> spd(n, n);
    spd.setFromTriplets(triplets.begin(), triplets.end());
    spd.makeCompressed();
    return spd;
}

template <class Scalar>
SparseMat<Scalar> solver_matrix_for(homa::LinSysSolverType   solver_type,
                                    const SparseMat<Scalar>& A)
{
    if (solver_type == homa::LinSysSolverType::CPU_MKL) {
        SparseMat<Scalar> lower = A.template triangularView<Eigen::Lower>();
        lower.makeCompressed();
        return lower;
    }

    SparseMat<Scalar> solver_matrix = A;
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

    using Clock   = std::chrono::high_resolution_clock;
    const auto t0 = Clock::now();
    fn();
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

template <class Scalar>
void warm_up_if_needed(homa::LinSysSolverType solver_type,
                       SparseMat<Scalar>&     solver_matrix,
                       Vec<Scalar>&           rhs)
{
    if (solver_type != homa::LinSysSolverType::GPU_CUDSS) {
        return;
    }

#ifdef USE_CUDSS
    std::unique_ptr<homa::LinSysSolver<Scalar>> solver(
        homa::LinSysSolver<Scalar>::create(solver_type));
    solver->setMatrix(solver_matrix.outerIndexPtr(),
                      solver_matrix.innerIndexPtr(),
                      solver_matrix.valuePtr(),
                      static_cast<int>(solver_matrix.rows()),
                      static_cast<int>(solver_matrix.nonZeros()));

    std::vector<int> empty;
    solver->ordering(empty, empty);
    solver->analyze_pattern();
    solver->factorize();
    Vec<Scalar> sol;
    solver->solve(rhs, sol);
    cudaDeviceSynchronize();
#endif
}

template <class Scalar>
StageTimes run_solver_path(homa::LinSysSolverType   solver_type,
                           SparseMat<Scalar>&       solver_matrix,
                           const SparseMat<Scalar>& residual_matrix,
                           Vec<Scalar>&             rhs,
                           std::vector<int>&        perm,
                           std::vector<int>&        etree,
                           bool                     use_homa_ordering)
{
    StageTimes times;

    std::unique_ptr<homa::LinSysSolver<Scalar>> solver(
        homa::LinSysSolver<Scalar>::create(solver_type));
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
        times.analysis_ms =
            time_step(solver_type, [&]() { solver->analyze_pattern(); });
    } else {
        times.analysis_ms = time_step(solver_type, [&]() {
            solver->analyze_pattern(use_homa_ordering ? perm : empty,
                                    use_homa_ordering ? etree : empty);
        });
    }

    times.factorize_ms = time_step(solver_type, [&]() { solver->factorize(); });

    Vec<Scalar> sol;
    times.solve_ms = time_step(solver_type, [&]() { solver->solve(rhs, sol); });
    times.residual = static_cast<double>((rhs - residual_matrix * sol).norm());

    return times;
}

template <class Scalar>
const char* precision_label()
{
    if constexpr (std::is_same_v<Scalar, float>)
        return "float";
    if constexpr (std::is_same_v<Scalar, double>)
        return "double";
    return "unknown";
}

template <class Scalar>
int run_benchmark(const std::string&             input_matrix,
                  homa::LinSysSolverType         solver_type,
                  int                            patch_size,
                  const std::string&             output_json,
                  bool                           make_spd_pattern,
                  int                            runs,
                  homa::Options::SeparatorMethod separator_method)
{
    SparseMat<Scalar> raw;
    if (!Eigen::loadMarket(raw, input_matrix)) {
        std::cerr << "Failed to read Matrix Market file: " << input_matrix
                  << "\n";
        return 1;
    }

    if (raw.rows() != raw.cols()) {
        std::cerr << "Matrix must be square: " << raw.rows() << "x"
                  << raw.cols() << "\n";
        return 1;
    }

    SparseMat<Scalar> A = make_spd_pattern ?
                              make_spd_from_pattern<Scalar>(raw) :
                              (is_matrix_market_symmetric(input_matrix) ?
                                   expand_symmetric_storage<Scalar>(raw) :
                                   raw);
    A.makeCompressed();

    SparseMat<Scalar> solver_matrix = solver_matrix_for<Scalar>(solver_type, A);

    spdlog::info("Matrix: {}x{}, {} non-zeros, precision={}",
                 A.rows(),
                 A.cols(),
                 A.nonZeros(),
                 precision_label<Scalar>());
    if (solver_matrix.nonZeros() != A.nonZeros()) {
        spdlog::info("Solver matrix: {} non-zeros", solver_matrix.nonZeros());
    }

    const int        n = static_cast<int>(A.rows());
    std::vector<int> Gp, Gi;
    homa::remove_diagonal(n, A.outerIndexPtr(), A.innerIndexPtr(), Gp, Gi);

    Vec<Scalar> rhs = Vec<Scalar>::Random(n);

    warm_up_if_needed<Scalar>(solver_type, solver_matrix, rhs);

    using Clock = std::chrono::high_resolution_clock;

    // Helper to accumulate timings across multiple runs.
    // keep the last residual for the JSON record but average all stage times
    const int total_runs      = std::max(1, runs);
    auto      avg_stage_times = [&](auto produce) {
        StageTimes sum;
        StageTimes last;
        for (int r = 0; r < total_runs; ++r) {
            last = produce();
            sum.ordering_ms += last.ordering_ms;
            sum.reorder_ms += last.reorder_ms;
            sum.analysis_ms += last.analysis_ms;
            sum.factorize_ms += last.factorize_ms;
            sum.solve_ms += last.solve_ms;
        }
        StageTimes avg;
        avg.ordering_ms  = sum.ordering_ms / total_runs;
        avg.reorder_ms   = sum.reorder_ms / total_runs;
        avg.analysis_ms  = sum.analysis_ms / total_runs;
        avg.factorize_ms = sum.factorize_ms / total_runs;
        avg.solve_ms     = sum.solve_ms / total_runs;
        avg.residual     = last.residual;
        return avg;
    };

    std::vector<int> empty_perm, empty_etree;
    StageTimes       def = avg_stage_times([&]() {
        return run_solver_path<Scalar>(
            solver_type, solver_matrix, A, rhs, empty_perm, empty_etree, false);
    });

    homa::Options opts;
    opts.use_gpu          = true;
    opts.patch_size       = patch_size;
    opts.compute_etree    = (solver_type == homa::LinSysSolverType::GPU_CUDSS);
    opts.local_method     = homa::Options::LocalMethod::AMD;
    opts.separator_method = separator_method;

    double               homa_ordering_ms_sum = 0.0;
    homa::OrderingResult ord;
    for (int r = 0; r < total_runs; ++r) {
        const auto t0 = Clock::now();
        ord           = homa::compute_ordering(n, Gp.data(), Gi.data(), opts);
        homa_ordering_ms_sum +=
            std::chrono::duration<double, std::milli>(Clock::now() - t0)
                .count();
    }
    const double homa_ordering_ms = homa_ordering_ms_sum / total_runs;

    if (!homa::check_valid_permutation(ord.perm.data(), ord.perm.size())) {
        spdlog::error("HOMA permutation is invalid!");
    }

    StageTimes homa  = avg_stage_times([&]() {
        return run_solver_path<Scalar>(
            solver_type, solver_matrix, A, rhs, ord.perm, ord.etree, true);
    });
    homa.ordering_ms = homa_ordering_ms;

    double def_total_ms  = def.ordering_ms + def.factorize_ms + def.solve_ms;
    double homa_total_ms = homa.ordering_ms + homa.factorize_ms + homa.solve_ms;
    if (solver_type == homa::LinSysSolverType::GPU_CUDSS) {
        def_total_ms += def.reorder_ms + def.analysis_ms;
        homa_total_ms += homa.reorder_ms + homa.analysis_ms;
    } else {
        def_total_ms += def.analysis_ms;
        homa_total_ms += homa.analysis_ms;
    }

    std::cout << "\n=== " << solver_display_name(solver_type)
              << " Matrix Results (" << precision_label<Scalar>() << ") ===\n";
    std::cout << std::left << std::fixed << std::setprecision(3)
              << std::setw(18) << "" << std::setw(16) << "Solver-default"
              << "HOMA\n"
              << std::setw(18) << "Ordering (ms) :" << std::setw(16) << "---"
              << homa_ordering_ms << "\n";

    if (solver_type == homa::LinSysSolverType::GPU_CUDSS) {
        std::cout << std::setw(18) << "Reorder  (ms) :" << std::setw(16)
                  << def.reorder_ms << homa.reorder_ms << "\n"
                  << std::setw(18) << "Symbolic (ms) :" << std::setw(16)
                  << def.analysis_ms << homa.analysis_ms << "\n";
    } else {
        std::cout << std::setw(18) << "Analysis (ms) :" << std::setw(16)
                  << def.analysis_ms << homa.analysis_ms << "\n";
    }

    std::cout << std::setw(18) << "Factorize (ms):" << std::setw(16)
              << def.factorize_ms << homa.factorize_ms << "\n"
              << std::setw(18) << "Solve (ms)    :" << std::setw(16)
              << def.solve_ms << homa.solve_ms << "\n"
              << std::setw(18) << "Total (ms)    :" << std::setw(16)
              << def_total_ms << homa_total_ms
              << " (speedup =" << def_total_ms / homa_total_ms << ")\n"
              << std::setw(18) << "Residual      :" << std::setw(16)
              << def.residual << homa.residual << "\n";

    if (!output_json.empty()) {
        BenchmarkRecord rec;
        rec.matrix_path      = input_matrix;
        rec.solver_name      = solver_display_name(solver_type);
        rec.precision        = precision_label<Scalar>();
        rec.n                = static_cast<int>(A.rows());
        rec.nnz              = static_cast<long long>(A.nonZeros());
        rec.patch_size       = patch_size;
        rec.separator_method = separator_method;
        rec.default_times    = def;
        rec.homa_times       = homa;
        rec.default_total_ms = def_total_ms;
        rec.homa_total_ms    = homa_total_ms;
        write_results_json(output_json, rec);
    }

    return 0;
}

}  // namespace

int main(int argc, char* argv[])
{
    std::string input_matrix;
    std::string solver_name = "cholmod";
    std::string precision   = "double";
    int         patch_size  = 512;
    std::string output_json;
    bool        make_spd_pattern = false;
    int         runs             = 1;
    std::string separator_method = "auto";

    CLI::App app{"Homa Matrix Market linear solver example"};
    app.add_option("-i,--input", input_matrix, "Input matrix (.mtx)")
        ->required();
    app.add_option(
        "-s,--solver", solver_name, "Solver backend: cholmod, mkl, cudss");
    app.add_option("-p,--patch_size", patch_size, "Patch size (default: 512)");
    app.add_option("--precision",
                   precision,
                   "Floating-point precision: double (default) or float");
    app.add_option("-o,--out",
                   output_json,
                   "Optional JSON file to write benchmark results to (no "
                   "output if empty)");
    app.add_flag(
        "--make-spd-from-pattern",
        make_spd_pattern,
        "Build a guaranteed SPD matrix from the input sparsity pattern");
    app.add_option(
           "-r,--runs",
           runs,
           "Number of timed runs (default: 1). Stage times are averaged")
        ->check(CLI::PositiveNumber);
    app.add_option("--separator-method",
                   separator_method,
                   "Separator strategy: auto (heuristic, default), quotient, "
                   "or direct (METIS)")
        ->check(CLI::IsMember({"auto", "quotient", "direct", "metis"},
                              CLI::ignore_case));
    CLI11_PARSE(app, argc, argv);

    homa::LinSysSolverType solver_type = homa::LinSysSolverType::CPU_CHOLMOD;
    try {
        solver_type = solver_type_from_name(solver_name);
    } catch (const std::invalid_argument& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }

    const homa::Options::SeparatorMethod sep_method =
        separator_method_from_name(separator_method);

    const std::string prec = to_lower(precision);
    if (prec == "double" || prec == "fp64" || prec == "f64") {
        return run_benchmark<double>(input_matrix,
                                     solver_type,
                                     patch_size,
                                     output_json,
                                     make_spd_pattern,
                                     runs,
                                     sep_method);
    }
    if (prec == "float" || prec == "fp32" || prec == "f32" ||
        prec == "single") {
        return run_benchmark<float>(input_matrix,
                                    solver_type,
                                    patch_size,
                                    output_json,
                                    make_spd_pattern,
                                    runs,
                                    sep_method);
    }

    std::cerr << "Unknown precision '" << precision
              << "' (expected 'double' or 'float')\n";
    return 1;
}
