//
// Created by behrooz on 2026-04-20.
//
// Benchmark that mirrors the full `tri_mesh_laplace_benchmark` pipeline
// and additionally exposes perm / etree / node_to_patch / node_to_etree_mapping
// arrays in main, then derives graph-clustering quality metrics:
//   - edge cut
//   - intra-cluster connectivity (density + conductance stats)
//   - separator sizes per etree level
//   - patch size balance (min / max / avg / std)
//   - boundary-vertex ratio
//
// All metrics are written to the same CSV row alongside ordering/factorize/solve
// times so a single run links patch quality to fill-in and runtime.

#include <igl/read_triangle_mesh.h>
#include <spdlog/spdlog.h>
#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include <chrono>
#include <unsupported/Eigen/SparseExtra>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <unordered_set>

#include "SPD_cot_matrix.h"
#include "LinSysSolver.hpp"
#include "get_factor_nnz.h"
#include "check_valid_permutation.h"
#include "ordering_factory.h"
#include "remove_diagonal.h"
#include "csv_utils.h"
#include "save_vector.h"


struct CLIArgs
{
    std::string input_mesh;
    int         binary_level = 7;
    std::string output_csv_address =
        "/home/behrooz/Desktop/Last_Project/gpu_ordering/output/patch_metrics/"
        "compare_patching";
    std::string solver_type                      = "CHOLMOD";
    std::string ordering_type                    = "PATCH_ORDERING";
    std::string default_ordering_type            = "METIS";
    int         use_patch_separator              = 1;
    std::string patch_ordering_local_permute_method = "amd";

    std::string patch_type = "rxmesh";
    int         patch_size = 512;
    bool        use_gpu    = false;

    CLIArgs(int argc, char* argv[])
    {
        CLI::App app{"Patch-quality / separator / fill-in comparison"};
        app.add_option("-a,--ordering", ordering_type, "ordering type");
        app.add_option("-d,--default_ordering_type",
                       default_ordering_type,
                       "default ordering type");
        app.add_option("-s,--solver", solver_type, "solver type");
        app.add_option("-o,--output", output_csv_address, "output folder name");
        app.add_option("-i,--input", input_mesh, "input mesh name");
        app.add_option("-g,--use_gpu", use_gpu, "use gpu");
        app.add_option("-p,--patch_type", patch_type,
                       "how to patch the graph/mesh");
        app.add_option("-z,--patch_size", patch_size, "patch size");
        app.add_option("-b,--binary_level", binary_level,
                       "binary level for binary tree ordering");
        app.add_option("-u,--use_patch_separator", use_patch_separator,
                       "use patch separator");
        app.add_option("-m,--patch_ordering_local_permute_method",
                       patch_ordering_local_permute_method,
                       "patch ordering local permute method");
        try {
            app.parse(argc, argv);
        } catch (const CLI::ParseError& e) {
            exit(app.exit(e));
        }
    }
};


// -----------------------------------------------------------------------------
// Clustering-quality metrics.
// All operate on the undirected graph described by (Gp, Gi) with the diagonal
// already removed, together with `node_to_patch` (one patch id per node) and
// `node_to_etree_mapping` (per-node -> etree/HMD index).
// -----------------------------------------------------------------------------
struct PatchMetrics
{
    // Patch balance.
    int    num_patches           = 0;
    int    min_patch_size        = 0;
    int    max_patch_size        = 0;
    double avg_patch_size        = 0.0;
    double std_patch_size        = 0.0;

    // Edge cut.
    long long num_undirected_edges = 0;
    long long edge_cut             = 0;  // undirected inter-patch edges
    double    edge_cut_ratio       = 0.0;

    // Intra-cluster connectivity.
    double avg_intra_density    = 0.0;  // unweighted edge density per patch
    double min_conductance      = 0.0;
    double median_conductance   = 0.0;
    double max_conductance      = 0.0;
    double mean_conductance     = 0.0;

    // Boundary.
    double boundary_ratio = 0.0;  // fraction of nodes with >=1 neighbor in a different patch

    // Separator sizes per etree (binary heap-order) level.
    // sep_level_sizes[lvl] = number of vertices in separators at that level
    // (i.e. all non-leaf HMD nodes at that level).
    std::vector<long long> sep_level_sizes;
    long long              total_separator_vertices = 0;
    int                    max_sep_level_populated  = -1;
};


static void computePatchBalance(const std::vector<int>& node_to_patch,
                                PatchMetrics&           m)
{
    if (node_to_patch.empty())
        return;
    int max_pid = *std::max_element(node_to_patch.begin(), node_to_patch.end());
    if (max_pid < 0) {
        m.num_patches = 0;
        return;
    }
    std::vector<int> patch_size(max_pid + 1, 0);
    for (int pid : node_to_patch) {
        if (pid >= 0)
            patch_size[pid]++;
    }

    int    count = 0;
    int    mn    = std::numeric_limits<int>::max();
    int    mx    = 0;
    double sum   = 0.0;
    for (int s : patch_size) {
        if (s <= 0)
            continue;
        count++;
        if (s < mn)
            mn = s;
        if (s > mx)
            mx = s;
        sum += s;
    }
    if (count == 0) {
        m.num_patches = 0;
        return;
    }
    double mean = sum / count;
    double var  = 0.0;
    for (int s : patch_size) {
        if (s <= 0)
            continue;
        double d = s - mean;
        var += d * d;
    }
    var /= count;

    m.num_patches    = count;
    m.min_patch_size = mn;
    m.max_patch_size = mx;
    m.avg_patch_size = mean;
    m.std_patch_size = std::sqrt(var);
}


static void computeEdgeAndConductanceMetrics(
    const std::vector<int>& Gp,
    const std::vector<int>& Gi,
    const std::vector<int>& node_to_patch,
    PatchMetrics&           m)
{
    const int G_N = static_cast<int>(Gp.size()) - 1;
    if (G_N <= 0 || node_to_patch.empty())
        return;

    int max_pid =
        *std::max_element(node_to_patch.begin(), node_to_patch.end());
    if (max_pid < 0)
        return;

    const int        P = max_pid + 1;
    std::vector<long long> patch_size(P, 0);
    std::vector<long long> patch_vol(P, 0);       // sum of degrees
    std::vector<long long> patch_internal(P, 0);  // 2x internal edges (directed)
    std::vector<long long> patch_cut(P, 0);       // directed cross-patch arcs

    long long total_directed_edges = 0;
    long long directed_cut         = 0;
    long long boundary_nodes       = 0;

    for (int i = 0; i < G_N; ++i) {
        int pi = node_to_patch[i];
        if (pi >= 0)
            patch_size[pi]++;
        bool is_boundary = false;
        for (int k = Gp[i]; k < Gp[i + 1]; ++k) {
            int j = Gi[k];
            if (j == i)
                continue;
            total_directed_edges++;
            int pj = (j >= 0 && j < G_N) ? node_to_patch[j] : -1;
            if (pi >= 0)
                patch_vol[pi]++;
            if (pi == pj && pi >= 0) {
                patch_internal[pi]++;
            } else {
                if (pi >= 0)
                    patch_cut[pi]++;
                directed_cut++;
                is_boundary = true;
            }
        }
        if (is_boundary)
            boundary_nodes++;
    }

    m.num_undirected_edges = total_directed_edges / 2;
    m.edge_cut             = directed_cut / 2;
    m.edge_cut_ratio =
        (m.num_undirected_edges > 0)
            ? static_cast<double>(m.edge_cut) / m.num_undirected_edges
            : 0.0;
    m.boundary_ratio = static_cast<double>(boundary_nodes) / G_N;

    // Intra-cluster edge density.
    long long total_vol = std::accumulate(patch_vol.begin(), patch_vol.end(),
                                          static_cast<long long>(0));

    double density_sum   = 0.0;
    int    density_count = 0;
    std::vector<double> conductances;
    conductances.reserve(P);

    for (int p = 0; p < P; ++p) {
        long long sz = patch_size[p];
        if (sz <= 1)
            continue;
        double max_edges = static_cast<double>(sz) * (sz - 1);  // directed
        double density   = patch_internal[p] / max_edges;
        density_sum += density;
        density_count++;

        long long vol_p       = patch_vol[p];
        long long vol_rest    = total_vol - vol_p;
        long long vol_min     = std::min(vol_p, vol_rest);
        if (vol_min > 0) {
            double cond = static_cast<double>(patch_cut[p]) / vol_min;
            conductances.push_back(cond);
        }
    }

    if (density_count > 0)
        m.avg_intra_density = density_sum / density_count;

    if (!conductances.empty()) {
        std::sort(conductances.begin(), conductances.end());
        m.min_conductance = conductances.front();
        m.max_conductance = conductances.back();
        m.median_conductance =
            conductances[conductances.size() / 2];
        m.mean_conductance =
            std::accumulate(conductances.begin(), conductances.end(), 0.0) /
            conductances.size();
    }
}


// Separator sizes per binary-heap level.
// In both PARTH and PATCH_ORDERING the decomposition tree is stored in heap
// order (root at index 0, children at 2*i+1 / 2*i+2). The level of a node with
// index h is floor(log2(h + 1)).
static void computeSeparatorLevelSizes(
    const std::vector<std::pair<int, int>>& node_to_etree_mapping,
    PatchMetrics&                           m)
{
    if (node_to_etree_mapping.empty())
        return;

    int max_etree_idx = 0;
    for (const auto& [g_node, e_idx] : node_to_etree_mapping) {
        if (e_idx > max_etree_idx)
            max_etree_idx = e_idx;
    }
    const int tree_size = max_etree_idx + 1;

    std::vector<long long> etree_node_size(tree_size, 0);
    for (const auto& [g_node, e_idx] : node_to_etree_mapping) {
        if (e_idx >= 0 && e_idx < tree_size)
            etree_node_size[e_idx]++;
    }

    auto level_of = [](int h) {
        int lvl = 0;
        int x   = h + 1;
        while (x > 1) {
            x >>= 1;
            lvl++;
        }
        return lvl;
    };

    int max_level = level_of(tree_size - 1);
    m.sep_level_sizes.assign(max_level + 1, 0);

    // A node is a separator if it has at least one child inside the tree AND
    // that child actually contains DOFs (non-empty).
    for (int h = 0; h < tree_size; ++h) {
        int left  = 2 * h + 1;
        int right = 2 * h + 2;
        bool has_child = false;
        if (left < tree_size && etree_node_size[left] > 0)
            has_child = true;
        if (right < tree_size && etree_node_size[right] > 0)
            has_child = true;
        if (has_child) {
            int lvl = level_of(h);
            m.sep_level_sizes[lvl] += etree_node_size[h];
            m.total_separator_vertices += etree_node_size[h];
            if (lvl > m.max_sep_level_populated)
                m.max_sep_level_populated = lvl;
        }
    }
}


static std::string separatorLevelsToString(
    const std::vector<long long>& sep_level_sizes)
{
    std::ostringstream oss;
    for (size_t i = 0; i < sep_level_sizes.size(); ++i) {
        if (i)
            oss << '|';
        oss << sep_level_sizes[i];
    }
    return oss.str();
}


int main(int argc, char* argv[])
{
    CLIArgs args(argc, argv);

    if (args.input_mesh.empty()) {
        std::cerr << "Error: Input mesh file not specified. Use -i or --input "
                     "to specify the mesh file."
                  << std::endl;
        return 1;
    }

    std::cout << "Loading mesh from: " << args.input_mesh << std::endl;
    std::cout << "Output CSV: " << args.output_csv_address << std::endl;

    Eigen::MatrixXd OV;
    Eigen::MatrixXi OF;
    if (!igl::read_triangle_mesh(args.input_mesh, OV, OF)) {
        std::cerr << "Failed to read the mesh: " << args.input_mesh
                  << std::endl;
        return 1;
    }

    Eigen::SparseMatrix<double> OL;
    homa::computeSPD_cot_matrix(OV, OF, OL);

    spdlog::info("Number of rows: {}", OL.rows());
    spdlog::info("Number of non-zeros: {}", OL.nonZeros());
    spdlog::info(
        "Sparsity: {:.2f}%",
        (1 - (OL.nonZeros() / static_cast<double>(OL.rows() * OL.rows()))) *
            100);

    Eigen::VectorXd rhs = Eigen::VectorXd::Random(OL.rows());
    Eigen::VectorXd result;

    std::vector<int> perm;
    std::vector<int> etree;
    std::vector<int> node_to_patch;
    std::vector<std::pair<int, int>> node_to_etree_mapping;

    homa::Ordering* ordering = nullptr;
    std::string              mesh_name =
        std::filesystem::path(args.input_mesh).stem().string();

    if (args.ordering_type == "DEFAULT") {
        spdlog::info("Using default ordering (default for each solver).");
        ordering = nullptr;
    } else if (args.ordering_type == "METIS") {
        spdlog::info("Using METIS ordering.");
        ordering = homa::Ordering::create(
            homa::DEMO_ORDERING_TYPE::METIS);
        if (args.solver_type == "CUDSS") {
            std::cerr
                << "METIS ordering is not supported with CUDSS solver."
                << std::endl;
            return 1;
        }
    } else if (args.ordering_type == "RXMESH_ND") {
        spdlog::info("Using RXMESH ordering.");
        ordering = homa::Ordering::create(
            homa::DEMO_ORDERING_TYPE::RXMESH_ND);
    } else if (args.ordering_type == "PATCH_ORDERING") {
        spdlog::info("Using PATCH_ORDERING ordering.");
        ordering = homa::Ordering::create(
            homa::DEMO_ORDERING_TYPE::PATCH_ORDERING);
        {
            homa::Options opts;
            opts.use_gpu             = args.use_gpu;
            opts.patch_size          = args.patch_size;
            opts.use_patch_separator = args.use_patch_separator != 0;
            opts.nd_levels           = args.binary_level;
            if (args.patch_ordering_local_permute_method == "metis")
                opts.local_method = homa::Options::LocalMethod::METIS;
            else if (args.patch_ordering_local_permute_method == "unity")
                opts.local_method = homa::Options::LocalMethod::NONE;
            else
                opts.local_method = homa::Options::LocalMethod::AMD;
            ordering->applyOptions(opts);
            ordering->setOptions({{"patch_type", args.patch_type}});
        }
    } else if (args.ordering_type == "PARTH") {
        ordering = homa::Ordering::create(
            homa::DEMO_ORDERING_TYPE::PARTH);
        ordering->setOptions(
            {{"binary_level", std::to_string(args.binary_level)}});
    } else if (args.ordering_type == "NEUTRAL") {
        spdlog::info("Using NEUTRAL ordering.");
        ordering = homa::Ordering::create(
            homa::DEMO_ORDERING_TYPE::NEUTRAL);
    } else {
        spdlog::error("Unknown Ordering type.");
    }

    homa::LinSysSolver* solver = nullptr;
    if (args.solver_type == "CHOLMOD") {
        solver = homa::LinSysSolver::create(
            homa::LinSysSolverType::CPU_CHOLMOD);
        spdlog::info("Using CHOLMOD direct solver.");
    } else if (args.solver_type == "CUDSS") {
        solver = homa::LinSysSolver::create(
            homa::LinSysSolverType::GPU_CUDSS);
        spdlog::info("Using CUDSS direct solver.");
    } else if (args.solver_type == "PARTH_SOLVER") {
        solver = homa::LinSysSolver::create(
            homa::LinSysSolverType::PARTH_SOLVER);
        spdlog::info("Using PARTH direct solver.");
    } else if (args.solver_type == "STRUMPACK") {
        solver = homa::LinSysSolver::create(
            homa::LinSysSolverType::GPU_STRUMPACK);
    } else if (args.solver_type == "MKL") {
        solver = homa::LinSysSolver::create(
            homa::LinSysSolverType::CPU_MKL);
        spdlog::info("Using Intel MKL PARDISO direct solver.");
        if (args.default_ordering_type == "METIS") {
            solver->ordering_type = "METIS";
        } else if (args.default_ordering_type == "AMD") {
            solver->ordering_type = "AMD";
        } else if (args.default_ordering_type == "ParMETIS") {
            solver->ordering_type = "ParMETIS";
        } else {
            spdlog::error("Unknown default ordering type.");
            return 1;
        }
    } else {
        spdlog::error("Unknown solver type.");
    }
    assert(solver != nullptr);

    std::vector<int> Gp;
    std::vector<int> Gi;
    homa::remove_diagonal(
        OL.rows(), OL.outerIndexPtr(), OL.innerIndexPtr(), Gp, Gi);

    double   residual                = 0;
    long int ordering_init_time      = -1;
    long int ordering_time           = -1;
    long int ordering_integration_time = -1;
    long int analysis_time           = -1;
    long int factorization_time      = -1;
    long int solve_time              = -1;
    long int factor_nnz              = -1;

    if (ordering != nullptr) {
        spdlog::info("Start Customized Ordering ...");
        if (ordering->needsMesh()) {
            ordering->setMesh(OV.data(), OV.rows(), OV.cols(),
                              OF.data(), OF.rows(), OF.cols());
        }
        ordering->setGraph(Gp.data(), Gi.data(), OL.rows(), Gi.size());

        auto ordering_init_start = std::chrono::high_resolution_clock::now();
        ordering->init();
        auto ordering_init_end = std::chrono::high_resolution_clock::now();
        ordering_init_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                ordering_init_end - ordering_init_start)
                .count();
        spdlog::info("Ordering initialization time: {} ms",
                     ordering_init_time);

        auto ordering_start = std::chrono::high_resolution_clock::now();
        if (args.solver_type == "CUDSS") {
            ordering->compute_permutation(perm, etree, true);
        } else {
            ordering->compute_permutation(perm, etree, false);
        }
        auto ordering_end = std::chrono::high_resolution_clock::now();
        ordering_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                            ordering_end - ordering_start)
                            .count();

        if (!homa::check_valid_permutation(perm.data(),
                                                    perm.size())) {
            spdlog::error("Permutation is not valid!");
        }
        spdlog::info("Ordering time: {} ms", ordering_time);
        assert(perm.size() == OL.rows());

        factor_nnz = homa::get_factor_nnz(OL.outerIndexPtr(),
                                                   OL.innerIndexPtr(),
                                                   OL.valuePtr(),
                                                   OL.rows(),
                                                   OL.nonZeros(),
                                                   perm);
        spdlog::info(
            "The ratio of factor non-zeros to matrix non-zeros given custom "
            "reordering: {}",
            (factor_nnz * 1.0 / OL.nonZeros()));
        solver->ordering_name = ordering->typeStr();

        ordering->getPatch(node_to_patch);
        ordering->getNodeToEtreeMapping(node_to_etree_mapping);
        spdlog::info("node_to_patch size: {}", node_to_patch.size());
        spdlog::info("node_to_etree_mapping size: {}",
                     node_to_etree_mapping.size());

        spdlog::info("Customize Ordering is done.");
    }

    Eigen::SparseMatrix<double> lower_OL;
    if (args.solver_type == "MKL") {
        lower_OL = OL.triangularView<Eigen::Lower>();
        solver->setMatrix(lower_OL.outerIndexPtr(),
                          lower_OL.innerIndexPtr(),
                          lower_OL.valuePtr(),
                          lower_OL.rows(),
                          lower_OL.nonZeros());
    } else {
        solver->setMatrix(OL.outerIndexPtr(),
                          OL.innerIndexPtr(),
                          OL.valuePtr(),
                          OL.rows(),
                          OL.nonZeros());
    }

    auto start = std::chrono::high_resolution_clock::now();
    solver->ordering(perm, etree);
    auto end = std::chrono::high_resolution_clock::now();
    ordering_integration_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    spdlog::info("Ordering integration time: {} ms",
                 ordering_integration_time);

    start = std::chrono::high_resolution_clock::now();
    solver->analyze_pattern(perm, etree);
    end   = std::chrono::high_resolution_clock::now();
    analysis_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    spdlog::info("Analysis time: {} ms", analysis_time);

    start = std::chrono::high_resolution_clock::now();
    solver->factorize();
    end   = std::chrono::high_resolution_clock::now();
    factorization_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    spdlog::info("Factorization time: {} ms", factorization_time);

    start = std::chrono::high_resolution_clock::now();
    solver->solve(rhs, result);
    end   = std::chrono::high_resolution_clock::now();
    solve_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    spdlog::info("Solve time: {} ms", solve_time);

    assert(OL.rows() == OL.cols());
    residual = (rhs - OL * result).norm();
    spdlog::info("Residual: {}", residual);
    spdlog::info("Final factor/matrix NNZ ratio: {}",
                 solver->getFactorNNZ() * 1.0 / OL.nonZeros());

    // -------------------------------------------------------------------------
    // Clustering-quality metrics
    // -------------------------------------------------------------------------
    PatchMetrics metrics;
    if (!node_to_patch.empty() &&
        static_cast<int>(node_to_patch.size()) == OL.rows()) {
        computePatchBalance(node_to_patch, metrics);
        computeEdgeAndConductanceMetrics(Gp, Gi, node_to_patch, metrics);
        spdlog::info(
            "Patches: {}  size[min/avg/max/std] = {}/{:.1f}/{}/{:.1f}",
            metrics.num_patches,
            metrics.min_patch_size,
            metrics.avg_patch_size,
            metrics.max_patch_size,
            metrics.std_patch_size);
        spdlog::info(
            "Edges: total={} cut={} ratio={:.4f}",
            metrics.num_undirected_edges,
            metrics.edge_cut,
            metrics.edge_cut_ratio);
        spdlog::info(
            "Intra density (mean) = {:.4f}  conductance[min/med/mean/max] = "
            "{:.4f}/{:.4f}/{:.4f}/{:.4f}",
            metrics.avg_intra_density,
            metrics.min_conductance,
            metrics.median_conductance,
            metrics.mean_conductance,
            metrics.max_conductance);
        spdlog::info("Boundary ratio: {:.4f}", metrics.boundary_ratio);
    } else {
        spdlog::warn(
            "node_to_patch is unavailable or size mismatch; skipping patch "
            "quality metrics.");
    }

    if (!node_to_etree_mapping.empty()) {
        computeSeparatorLevelSizes(node_to_etree_mapping, metrics);
        spdlog::info("Total separator vertices: {}",
                     metrics.total_separator_vertices);
        spdlog::info("Separator sizes per level: {}",
                     separatorLevelsToString(metrics.sep_level_sizes));
    } else {
        spdlog::warn(
            "node_to_etree_mapping is empty; skipping per-level separator "
            "sizes.");
    }

    // -------------------------------------------------------------------------
    // CSV logging
    // -------------------------------------------------------------------------
    std::string              csv_name = args.output_csv_address;
    std::vector<std::string> header = {
        "mesh_name",
        "G_N",
        "G_NNZ",
        "solver_type",
        "ordering_type",
        "nd_levels",
        "patch_type",
        "use_patch_separator",
        "local_permute_method",
        "patch_size",
        "patch_time",
        "node_to_patch_time",
        "decompose_time",
        "local_permute_time",
        "assemble_time",
        "factor/matrix NNZ ratio",
        "ordering_time",
        "ordering_integration_time",
        "analysis_time",
        "factorization_time",
        "solve_time",
        "residual",
        // clustering-quality metrics
        "num_patches",
        "min_patch_size",
        "max_patch_size",
        "avg_patch_size",
        "std_patch_size",
        "num_undirected_edges",
        "edge_cut",
        "edge_cut_ratio",
        "avg_intra_density",
        "min_conductance",
        "median_conductance",
        "mean_conductance",
        "max_conductance",
        "boundary_ratio",
        "total_separator_vertices",
        "max_sep_level_populated",
        "sep_level_sizes",
    };

    homa::CSVManager runtime_csv(csv_name, "some address", header,
                                          false);
    runtime_csv.addElementToRecord(mesh_name, "mesh_name");
    runtime_csv.addElementToRecord(OL.rows(), "G_N");
    runtime_csv.addElementToRecord(OL.nonZeros(), "G_NNZ");
    runtime_csv.addElementToRecord(args.solver_type, "solver_type");
    if (ordering != nullptr) {
        runtime_csv.addElementToRecord(ordering->typeStr(), "ordering_type");
    } else {
        runtime_csv.addElementToRecord("DEFAULT", "ordering_type");
    }
    int nd_levels = etree.empty() ? 0 : static_cast<int>(std::log2(etree.size() + 1));
    runtime_csv.addElementToRecord(nd_levels, "nd_levels");
    runtime_csv.addElementToRecord(args.use_patch_separator,
                                   "use_patch_separator");
    runtime_csv.addElementToRecord(args.patch_ordering_local_permute_method,
                                   "local_permute_method");

    if (args.ordering_type == "PATCH_ORDERING") {
        std::map<std::string, double> stat;
        assert(ordering != nullptr);
        ordering->getStatistics(stat);
        runtime_csv.addElementToRecord(args.patch_type, "patch_type");
        runtime_csv.addElementToRecord(stat["patch_size"], "patch_size");
        runtime_csv.addElementToRecord(stat["patching_time"], "patch_time");
        runtime_csv.addElementToRecord(stat["node_to_patch_time"],
                                       "node_to_patch_time");
        runtime_csv.addElementToRecord(stat["decompose_time"],
                                       "decompose_time");
        runtime_csv.addElementToRecord(stat["local_permute_time"],
                                       "local_permute_time");
        runtime_csv.addElementToRecord(stat["assemble_time"], "assemble_time");
    } else {
        runtime_csv.addElementToRecord("", "patch_type");
        runtime_csv.addElementToRecord(0, "patch_size");
        runtime_csv.addElementToRecord(0, "patch_time");
        runtime_csv.addElementToRecord(0, "node_to_patch_time");
        runtime_csv.addElementToRecord(0, "decompose_time");
        runtime_csv.addElementToRecord(0, "local_permute_time");
        runtime_csv.addElementToRecord(0, "assemble_time");
    }

    if (args.solver_type == "MKL") {
        runtime_csv.addElementToRecord(
            solver->getFactorNNZ() * 1.0 / OL.nonZeros(),
            "factor/matrix NNZ ratio");
    } else {
        runtime_csv.addElementToRecord(factor_nnz * 1.0 / OL.nonZeros(),
                                       "factor/matrix NNZ ratio");
    }
    runtime_csv.addElementToRecord(ordering_time, "ordering_time");
    runtime_csv.addElementToRecord(ordering_integration_time,
                                   "ordering_integration_time");
    runtime_csv.addElementToRecord(analysis_time, "analysis_time");
    runtime_csv.addElementToRecord(factorization_time, "factorization_time");
    runtime_csv.addElementToRecord(solve_time, "solve_time");
    runtime_csv.addElementToRecord(residual, "residual");

    runtime_csv.addElementToRecord(metrics.num_patches, "num_patches");
    runtime_csv.addElementToRecord(metrics.min_patch_size, "min_patch_size");
    runtime_csv.addElementToRecord(metrics.max_patch_size, "max_patch_size");
    runtime_csv.addElementToRecord(metrics.avg_patch_size, "avg_patch_size");
    runtime_csv.addElementToRecord(metrics.std_patch_size, "std_patch_size");
    runtime_csv.addElementToRecord(metrics.num_undirected_edges,
                                   "num_undirected_edges");
    runtime_csv.addElementToRecord(metrics.edge_cut, "edge_cut");
    runtime_csv.addElementToRecord(metrics.edge_cut_ratio, "edge_cut_ratio");
    runtime_csv.addElementToRecord(metrics.avg_intra_density,
                                   "avg_intra_density");
    runtime_csv.addElementToRecord(metrics.min_conductance, "min_conductance");
    runtime_csv.addElementToRecord(metrics.median_conductance,
                                   "median_conductance");
    runtime_csv.addElementToRecord(metrics.mean_conductance,
                                   "mean_conductance");
    runtime_csv.addElementToRecord(metrics.max_conductance, "max_conductance");
    runtime_csv.addElementToRecord(metrics.boundary_ratio, "boundary_ratio");
    runtime_csv.addElementToRecord(metrics.total_separator_vertices,
                                   "total_separator_vertices");
    runtime_csv.addElementToRecord(metrics.max_sep_level_populated,
                                   "max_sep_level_populated");
    runtime_csv.addElementToRecord(
        separatorLevelsToString(metrics.sep_level_sizes), "sep_level_sizes");

    runtime_csv.addRecord();

    delete solver;
    delete ordering;
    return 0;
}
