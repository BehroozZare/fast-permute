#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include <chrono>
#include <iomanip>
#include <iostream>

#include "SPD_cot_matrix.h"
#include "LinSysSolver.hpp"
#include "homa/ordering.h"
#include "homa/types.h"
#include "remove_diagonal.h"
#include "check_valid_permutation.h"
#include <igl/read_triangle_mesh.h>
#include <spdlog/spdlog.h>

int main(int argc, char* argv[])
{
    std::string input_mesh;
    int         patch_size = 512;

    CLI::App app{"Homa CHOLMOD example"};
    app.add_option("-i,--input",      input_mesh, "Input mesh (.obj)")->required();
    app.add_option("-p,--patch_size", patch_size, "Patch size (default: 512)");
    CLI11_PARSE(app, argc, argv);

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    if (!igl::read_triangle_mesh(input_mesh, V, F)) {
        std::cerr << "Failed to read mesh: " << input_mesh << "\n";
        return 1;
    }

    Eigen::SparseMatrix<double> L;
    homa::computeSPD_cot_matrix(V, F, L);
    spdlog::info("Matrix: {}x{}, {} non-zeros", L.rows(), L.cols(), L.nonZeros());

    int n = static_cast<int>(L.rows());
    std::vector<int> Gp, Gi;
    homa::remove_diagonal(n, L.outerIndexPtr(), L.innerIndexPtr(), Gp, Gi);

    Eigen::VectorXd rhs = Eigen::VectorXd::Random(n);

    using Clock = std::chrono::high_resolution_clock;
    auto elapsed_ms = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
    };

    // --- Solver-default path (CHOLMOD internal METIS ordering) ---
    long def_analysis_ms = 0, def_factorize_ms = 0, def_solve_ms = 0;
    double def_residual = 0.0;
    {
        std::unique_ptr<homa::LinSysSolver> solver(
            homa::LinSysSolver::create(homa::LinSysSolverType::CPU_CHOLMOD));
        solver->setMatrix(L.outerIndexPtr(), L.innerIndexPtr(), L.valuePtr(), n, L.nonZeros());

        std::vector<int> empty;
        auto t0 = Clock::now();
        solver->analyze_pattern(empty, empty);
        def_analysis_ms = elapsed_ms(t0, Clock::now());

        t0 = Clock::now();
        solver->factorize();
        def_factorize_ms = elapsed_ms(t0, Clock::now());

        Eigen::VectorXd sol;
        t0 = Clock::now();
        solver->solve(rhs, sol);
        def_solve_ms = elapsed_ms(t0, Clock::now());
        def_residual = (rhs - L * sol).norm();
    }

    // --- HOMA path ---
    long homa_ordering_ms = 0, homa_analysis_ms = 0, homa_factorize_ms = 0, homa_solve_ms = 0;
    double homa_residual = 0.0;
    {
        homa::Options opts;
        opts.patch_size          = patch_size;
        opts.use_patch_separator = true;
        opts.local_method        = homa::Options::LocalMethod::AMD;

        auto t0 = Clock::now();
        homa::OrderingResult ord = homa::compute_ordering(n, Gp.data(), Gi.data(), opts);
        homa_ordering_ms = elapsed_ms(t0, Clock::now());

        if (!homa::check_valid_permutation(ord.perm.data(), ord.perm.size()))
            spdlog::error("HOMA permutation is invalid!");

        std::unique_ptr<homa::LinSysSolver> solver(
            homa::LinSysSolver::create(homa::LinSysSolverType::CPU_CHOLMOD));
        solver->setMatrix(L.outerIndexPtr(), L.innerIndexPtr(), L.valuePtr(), n, L.nonZeros());

        t0 = Clock::now();
        solver->analyze_pattern(ord.perm, ord.etree);
        homa_analysis_ms = elapsed_ms(t0, Clock::now());

        t0 = Clock::now();
        solver->factorize();
        homa_factorize_ms = elapsed_ms(t0, Clock::now());

        Eigen::VectorXd sol;
        t0 = Clock::now();
        solver->solve(rhs, sol);
        homa_solve_ms = elapsed_ms(t0, Clock::now());
        homa_residual = (rhs - L * sol).norm();
    }

    std::cout << "\n=== CHOLMOD Results ===\n";
    std::cout << std::left
              << std::setw(18) << ""                << std::setw(16) << "Solver-default" << "HOMA\n"
              << std::setw(18) << "Ordering (ms) :" << std::setw(16) << "---"            << homa_ordering_ms  << "\n"
              << std::setw(18) << "Analysis (ms) :" << std::setw(16) << def_analysis_ms  << homa_analysis_ms  << "\n"
              << std::setw(18) << "Factorize (ms):" << std::setw(16) << def_factorize_ms << homa_factorize_ms << "\n"
              << std::setw(18) << "Solve (ms)    :" << std::setw(16) << def_solve_ms     << homa_solve_ms     << "\n"
              << std::setw(18) << "Residual      :" << std::setw(16) << def_residual     << homa_residual     << "\n";

    return 0;
}
