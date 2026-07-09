#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>
#include <igl/read_triangle_mesh.h>
#include <spdlog/spdlog.h>

#include "homa/homa.h"
#include "homa/types.h"
#include "homa/utils/SPD_cot_matrix.h"
#include "homa/utils/check_valid_permutation.h"
#include "homa/utils/cuda_error_handler.h"
#include "homa/utils/remove_diagonal.h"
#include <homa/solvers/LinSysSolver.h>

#include "util.h"
#include "cleanup_mesh.h"

template <class Scalar>
using SparseMat = Eigen::SparseMatrix<Scalar>;

template <class Scalar>
using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

template <class Scalar>
const char* precision_label()
{
    if constexpr (std::is_same_v<Scalar, float>) {
        return "float";
    }
    if constexpr (std::is_same_v<Scalar, double>) {
        return "double";
    }
    return "unknown";
}

template <typename Fn>
double time_cudss_step(Fn&& fn)
{
    GpuTimer timer;
    timer.start();
    fn();
    return timer.stop_ms();
}

template <class Scalar>
void warm_up(SparseMat<Scalar>& L, Vec<Scalar>& rhs)
{
    std::unique_ptr<homa::LinSysSolver<Scalar>> solver(
        homa::LinSysSolver<Scalar>::create(homa::LinSysSolverType::GPU_CUDSS));
    solver->setMatrix(L.outerIndexPtr(),
                      L.innerIndexPtr(),
                      L.valuePtr(),
                      static_cast<int>(L.rows()),
                      static_cast<int>(L.nonZeros()));

    std::vector<int> empty;
    solver->ordering(empty, empty);
    solver->analyze_pattern();
    solver->factorize();
    Vec<Scalar> sol;
    solver->solve(rhs, sol);
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <class Scalar>
StageTimes run_cudss_path(SparseMat<Scalar>& L,
                          Vec<Scalar>&       rhs,
                          std::vector<int>&  perm,
                          std::vector<int>&  etree,
                          bool               use_homa_ordering)
{
    StageTimes times;

    std::unique_ptr<homa::LinSysSolver<Scalar>> solver(
        homa::LinSysSolver<Scalar>::create(homa::LinSysSolverType::GPU_CUDSS));
    solver->setMatrix(L.outerIndexPtr(),
                      L.innerIndexPtr(),
                      L.valuePtr(),
                      static_cast<int>(L.rows()),
                      static_cast<int>(L.nonZeros()));

    std::vector<int> empty;
    times.reorder_ms   = time_cudss_step([&]() {
        solver->ordering(use_homa_ordering ? perm : empty,
                         use_homa_ordering ? etree : empty);
    });
    times.analysis_ms  = time_cudss_step([&]() { solver->analyze_pattern(); });
    times.factorize_ms = time_cudss_step([&]() { solver->factorize(); });

    Vec<Scalar> sol;
    times.solve_ms = time_cudss_step([&]() { solver->solve(rhs, sol); });
    times.residual = static_cast<double>((rhs - L * sol).norm());

    return times;
}

template <class Scalar>
int run_benchmark(const std::string&             input_mesh,
                  int                            patch_size,
                  const std::string&             output_json,
                  int                            runs,
                  homa::Options::SeparatorMethod separator_method)
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    if (!igl::read_triangle_mesh(input_mesh, V, F)) {
        spdlog::error("Failed to read mesh: {}", input_mesh);
        return 1;
    }
    if (!cleanup_mesh(V, F)) {
        spdlog::error("Mesh cleanup produced an empty mesh: {}", input_mesh);
        return 1;
    }

    Eigen::SparseMatrix<double> L_double;
    homa::computeSPD_cot_matrix(V, F, L_double);
    L_double.makeCompressed();

    SparseMat<Scalar> L = L_double.template cast<Scalar>();
    L.makeCompressed();

    spdlog::info("Matrix: {}x{}, {} non-zeros, precision={}",
                 L.rows(),
                 L.cols(),
                 L.nonZeros(),
                 precision_label<Scalar>());

    const int        n = static_cast<int>(L.rows());
    std::vector<int> Gp, Gi;
    homa::remove_diagonal(n, L.outerIndexPtr(), L.innerIndexPtr(), Gp, Gi);

    Vec<Scalar> rhs = Vec<Scalar>::Random(n);

    warm_up<Scalar>(L, rhs);

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
        return run_cudss_path<Scalar>(L, rhs, empty_perm, empty_etree, false);
    });

    homa::Options opts;
    opts.use_gpu          = true;
    opts.patch_size       = patch_size;
    opts.compute_etree    = true;
    opts.local_method     = homa::Options::LocalMethod::AMD;
    opts.separator_method = separator_method;

    using Clock = std::chrono::high_resolution_clock;

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
        return run_cudss_path<Scalar>(L, rhs, ord.perm, ord.etree, true);
    });
    homa.ordering_ms = homa_ordering_ms;

    const double def_total_ms =
        def.reorder_ms + def.analysis_ms + def.factorize_ms + def.solve_ms;
    const double homa_total_ms = homa.ordering_ms + homa.reorder_ms +
                                 homa.analysis_ms + homa.factorize_ms +
                                 homa.solve_ms;

    std::cout << "\n=== cuDSS Mesh Results (" << precision_label<Scalar>()
              << ") ===\n";
    std::cout << std::left << std::fixed << std::setprecision(3)
              << std::setw(18) << "" << std::setw(16) << "Solver-default"
              << "HOMA\n"
              << std::setw(18) << "Ordering (ms) :" << std::setw(16) << "---"
              << homa_ordering_ms << "\n"
              << std::setw(18) << "Reorder  (ms) :" << std::setw(16)
              << def.reorder_ms << homa.reorder_ms << "\n"
              << std::setw(18) << "Symbolic (ms) :" << std::setw(16)
              << def.analysis_ms << homa.analysis_ms << "\n"
              << std::setw(18) << "Factorize (ms):" << std::setw(16)
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
        rec.matrix_path = input_mesh;
        rec.solver_name =
            solver_display_name(homa::LinSysSolverType::GPU_CUDSS);
        rec.precision        = precision_label<Scalar>();
        rec.n                = static_cast<int>(L.rows());
        rec.nnz              = static_cast<long long>(L.nonZeros());
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

int main(int argc, char* argv[])
{
    std::string input_mesh;
    std::string precision  = "double";
    int         patch_size = 512;
    std::string output_json;
    int         runs             = 1;
    std::string separator_method = "quotient";

    CLI::App app{"Homa cuDSS mesh example"};
    app.add_option("-i,--input", input_mesh, "Input mesh")->required();
    app.add_option("-p,--patch_size", patch_size, "Patch size (default: 512)");
    app.add_option("--precision",
                   precision,
                   "Floating-point precision: double (default) or float");
    app.add_option("-o,--out",
                   output_json,
                   "Optional JSON file to write benchmark results to (no "
                   "output if empty)");
    app.add_option("-r,--runs", runs, "Number of timed runs (default: 1)")
        ->check(CLI::PositiveNumber);
    app.add_option("--separator-method",
                   separator_method,
                   "Separator strategy: auto (heuristic, default), quotient, "
                   "or direct (METIS)")
        ->check(CLI::IsMember({"auto", "quotient", "direct", "metis"},
                              CLI::ignore_case));
    CLI11_PARSE(app, argc, argv);

    const homa::Options::SeparatorMethod sep_method =
        separator_method_from_name(separator_method);

    const std::string prec = to_lower(precision);
    if (prec == "double" || prec == "fp64" || prec == "f64") {
        return run_benchmark<double>(
            input_mesh, patch_size, output_json, runs, sep_method);
    }
    if (prec == "float" || prec == "fp32" || prec == "f32" ||
        prec == "single") {
        return run_benchmark<float>(
            input_mesh, patch_size, output_json, runs, sep_method);
    }
    spdlog::error("Unknown precision '{}' (expected 'double' or 'float')",
                  precision);
    return 1;
}
