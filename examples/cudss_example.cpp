#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include <chrono>
#include <iostream>

#include "SPD_cot_matrix.h"
#include "LinSysSolver.hpp"
#include "ordering_factory.h"
#include "remove_diagonal.h"
#include "check_valid_permutation.h"
#include <igl/read_triangle_mesh.h>
#include <spdlog/spdlog.h>

int main(int argc, char* argv[])
{
    std::string input_mesh;
    int         patch_size = 512;
    int         use_gpu    = 1;

    CLI::App app{"Homa cuDSS example"};
    app.add_option("-i,--input",      input_mesh, "Input mesh (.obj)")->required();
    app.add_option("-p,--patch_size", patch_size, "Patch size (default: 512)");
    app.add_option("-g,--use_gpu",    use_gpu,    "Use GPU for ordering (0|1, default: 1)");
    CLI11_PARSE(app, argc, argv);

    // Load mesh
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    if (!igl::read_triangle_mesh(input_mesh, V, F)) {
        std::cerr << "Failed to read mesh: " << input_mesh << "\n";
        return 1;
    }

    // Build SPD cotangent Laplacian
    Eigen::SparseMatrix<double> L;
    homa::computeSPD_cot_matrix(V, F, L);
    spdlog::info("Matrix: {}x{}, {} non-zeros", L.rows(), L.cols(), L.nonZeros());

    // Build graph (CSR without diagonal)
    std::vector<int> Gp, Gi;
    homa::remove_diagonal(L.rows(), L.outerIndexPtr(), L.innerIndexPtr(), Gp, Gi);

    // Ordering
    std::vector<int> perm, etree;
    homa::Ordering* ordering = homa::Ordering::create(homa::DEMO_ORDERING_TYPE::PATCH_ORDERING);
    {
        homa::Options opts;
        opts.use_gpu             = (use_gpu != 0);
        opts.patch_size          = patch_size;
        opts.use_patch_separator = true;
        opts.local_method        = homa::Options::LocalMethod::AMD;
        ordering->applyOptions(opts);
        ordering->setOptions({{"patch_type", "metis"}});
    }
    ordering->setGraph(Gp.data(), Gi.data(), L.rows(), Gi.size());
    ordering->init();

    auto t0 = std::chrono::high_resolution_clock::now();
    // cuDSS requires the permutation in device-friendly (1-based) format
    ordering->compute_permutation(perm, etree, /*for_gpu=*/true);
    auto t1 = std::chrono::high_resolution_clock::now();
    long ordering_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    if (!homa::check_valid_permutation(perm.data(), perm.size()))
        spdlog::error("Permutation is invalid!");

    // Solver
    homa::LinSysSolver* solver = homa::LinSysSolver::create(homa::LinSysSolverType::GPU_CUDSS);
    solver->setMatrix(L.outerIndexPtr(), L.innerIndexPtr(), L.valuePtr(), L.rows(), L.nonZeros());

    t0 = std::chrono::high_resolution_clock::now();
    solver->ordering(perm, etree);
    t1 = std::chrono::high_resolution_clock::now();
    long integration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    solver->analyze_pattern(perm, etree);
    t1 = std::chrono::high_resolution_clock::now();
    long analysis_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    solver->factorize();
    t1 = std::chrono::high_resolution_clock::now();
    long factorize_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    Eigen::VectorXd rhs = Eigen::VectorXd::Random(L.rows());
    Eigen::VectorXd result;
    t0 = std::chrono::high_resolution_clock::now();
    solver->solve(rhs, result);
    t1 = std::chrono::high_resolution_clock::now();
    long solve_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    double residual = (rhs - L * result).norm();

    std::cout << "\n=== cuDSS Results ===\n";
    std::cout << "Ordering  : " << ordering_ms   << " ms\n";
    std::cout << "Analysis  : " << analysis_ms   << " ms\n";
    std::cout << "Factorize : " << factorize_ms  << " ms\n";
    std::cout << "Solve     : " << solve_ms       << " ms\n";
    std::cout << "Residual  : " << residual       << "\n";

    delete solver;
    delete ordering;
    return 0;
}
