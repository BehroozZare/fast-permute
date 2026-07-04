#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <homa/homa.h>
#include <homa/matrix_view.h>
#include <homa/solvers/LinSysSolver.h>
#include <homa/types.h>
#include <homa/utils/remove_diagonal.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using HostIntArray =
    nb::ndarray<int, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class Scalar>
using HostValueArray =
    nb::ndarray<Scalar, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class Scalar>
using HostDense = nb::ndarray<Scalar, nb::device::cpu>;

// Wrap a std::vector<int> as an owning 1-D numpy int32 array.
nb::ndarray<nb::numpy, int, nb::ndim<1>> to_numpy_int(const std::vector<int>& v)
{
    const size_t count = v.size();
    int*         data  = new int[count == 0 ? 1 : count];
    std::copy(v.begin(), v.end(), data);
    nb::capsule owner(data,
                      [](void* p) noexcept { delete[] static_cast<int*>(p); });
    return nb::ndarray<nb::numpy, int, nb::ndim<1>>(data, {count}, owner);
}

// Standalone ordering. `indptr`/`indices` are the CSR pattern of the
// (symmetric) matrix; the diagonal is removed here to match
// LinSysSolver::ordering.
nb::object py_compute_ordering(HostIntArray         indptr,
                               HostIntArray         indices,
                               int                  n,
                               const homa::Options& opts)
{
    if (n <= 0)
        throw std::invalid_argument("compute_ordering: n must be positive");
    if (static_cast<int>(indptr.shape(0)) != n + 1)
        throw std::invalid_argument(
            "compute_ordering: indptr length must be n+1");

    std::vector<int>     Gp, Gi;
    homa::OrderingResult ord;
    {
        nb::gil_scoped_release release;
        homa::remove_diagonal(n, indptr.data(), indices.data(), Gp, Gi);
        ord = homa::compute_ordering(n, Gp.data(), Gi.data(), opts);
    }
    return nb::make_tuple(to_numpy_int(ord.perm), to_numpy_int(ord.etree));
}

template <class Scalar>
class PySolver
{
   public:
    explicit PySolver(homa::LinSysSolverType backend)
    {
        solver_.reset(homa::LinSysSolver<Scalar>::create(backend));
        if (!solver_) {
            throw std::runtime_error(
                "homa: requested solver backend is not available in this "
                "build");
        }
    }

    // Host CSR (borrowed): the caller must keep the arrays alive. cuDSS copies
    // to device internally; CHOLMOD/MKL borrow the pointers.
    void set_matrix_host(HostIntArray           indptr,
                         HostIntArray           indices,
                         HostValueArray<Scalar> values,
                         int                    n)
    {
        const int nnz = static_cast<int>(indices.shape(0));
        solver_->setMatrix(
            indptr.data(), indices.data(), values.data(), n, nnz);
    }

    // Device CSR (borrowed device pointers, cuDSS only). Zero-copy.
    void set_matrix_device(std::uintptr_t indptr,
                           std::uintptr_t indices,
                           std::uintptr_t values,
                           int            n,
                           int            nnz)
    {
        homa::SparseMatrixView<Scalar> view{n,
                                            n,
                                            nnz,
                                            reinterpret_cast<int*>(indptr),
                                            reinterpret_cast<int*>(indices),
                                            reinterpret_cast<Scalar*>(values),
                                            homa::SparseFormat::CSR,
                                            homa::MemoryLocation::Device};
        solver_->setMatrix(view);
    }

    // HOMA ordering. Skip calling this to use the backend's own default.
    void ordering(const homa::Options& opts)
    {
        nb::gil_scoped_release release;
        solver_->ordering(opts);
    }

    void analyze_pattern()
    {
        nb::gil_scoped_release release;
        solver_->analyze_pattern();
    }

    // Numeric factorization. To refactorize with new values (same pattern),
    // overwrite the value buffer handed to set_matrix_* and call this again.
    void factorize()
    {
        nb::gil_scoped_release release;
        solver_->factorize();
    }

    // Host solve: b and out are packed column-major host arrays (1-D or 2-D).
    void solve_host(HostDense<Scalar> b, HostDense<Scalar> out)
    {
        const int rows = static_cast<int>(b.shape(0));
        const int cols = b.ndim() > 1 ? static_cast<int>(b.shape(1)) : 1;
        homa::DenseMatrixView<Scalar> bv{
            b.data(), rows, cols, rows, homa::MemoryLocation::Host};
        homa::DenseMatrixView<Scalar> xv{
            out.data(), rows, cols, rows, homa::MemoryLocation::Host};
        nb::gil_scoped_release release;
        solver_->solve(bv, xv);
    }

    // Device solve: raw device pointers to packed column-major buffers.
    void solve_device(std::uintptr_t b_ptr, std::uintptr_t x_ptr, int n, int k)
    {
        homa::DenseMatrixView<Scalar> bv{reinterpret_cast<Scalar*>(b_ptr),
                                         n,
                                         k,
                                         n,
                                         homa::MemoryLocation::Device};
        homa::DenseMatrixView<Scalar> xv{reinterpret_cast<Scalar*>(x_ptr),
                                         n,
                                         k,
                                         n,
                                         homa::MemoryLocation::Device};
        nb::gil_scoped_release        release;
        solver_->solve(bv, xv);
    }

    void reset()
    {
        solver_->resetSolver();
    }
    int n() const
    {
        return solver_->N;
    }

   private:
    std::unique_ptr<homa::LinSysSolver<Scalar>> solver_;
};

template <class Scalar>
void bind_solver(nb::module_& m, const char* name)
{
    nb::class_<PySolver<Scalar>>(m, name)
        .def(nb::init<homa::LinSysSolverType>(), "backend"_a)
        .def("set_matrix_host",
             &PySolver<Scalar>::set_matrix_host,
             "indptr"_a,
             "indices"_a,
             "values"_a,
             "n"_a)
        .def("set_matrix_device",
             &PySolver<Scalar>::set_matrix_device,
             "indptr"_a,
             "indices"_a,
             "values"_a,
             "n"_a,
             "nnz"_a)
        .def("ordering", &PySolver<Scalar>::ordering, "options"_a)
        .def("analyze_pattern", &PySolver<Scalar>::analyze_pattern)
        .def("factorize", &PySolver<Scalar>::factorize)
        .def("solve_host", &PySolver<Scalar>::solve_host, "b"_a, "out"_a)
        .def("solve_device",
             &PySolver<Scalar>::solve_device,
             "b"_a,
             "x"_a,
             "n"_a,
             "k"_a)
        .def("reset", &PySolver<Scalar>::reset)
        .def_prop_ro("n", &PySolver<Scalar>::n);
}

constexpr bool kHasCudss =
#ifdef USE_CUDSS
    true;
#else
    false;
#endif
constexpr bool kHasCholmod =
#ifdef USE_CHOLMOD
    true;
#else
    false;
#endif
constexpr bool kHasMkl =
#ifdef USE_MKL
    true;
#else
    false;
#endif

}  // namespace

NB_MODULE(_homa, m)
{
    m.doc() = "Low-level Homa bindings (ordering + SPD direct solvers)";

    nb::enum_<homa::LinSysSolverType>(m, "Backend")
        .value("CPU_CHOLMOD", homa::LinSysSolverType::CPU_CHOLMOD)
        .value("CPU_MKL", homa::LinSysSolverType::CPU_MKL)
        .value("GPU_CUDSS", homa::LinSysSolverType::GPU_CUDSS);

    nb::enum_<homa::Options::SeparatorMethod>(m, "SeparatorMethod")
        .value("AUTO", homa::Options::SeparatorMethod::AUTO)
        .value("QUOTIENT", homa::Options::SeparatorMethod::QUOTIENT)
        .value("DIRECT_METIS", homa::Options::SeparatorMethod::DIRECT_METIS);

    nb::enum_<homa::Options::LocalMethod>(m, "LocalMethod")
        .value("AMD", homa::Options::LocalMethod::AMD)
        .value("METIS", homa::Options::LocalMethod::METIS)
        .value("NONE", homa::Options::LocalMethod::NONE);

    nb::enum_<homa::Options::PatchMethod>(m, "PatchMethod")
        .value("LLOYD", homa::Options::PatchMethod::LLOYD)
        .value("METIS", homa::Options::PatchMethod::METIS)
        .value("GREEDY", homa::Options::PatchMethod::GREEDY);

    nb::enum_<homa::Options::LloydSeedMethod>(m, "LloydSeedMethod")
        .value("RANDOM", homa::Options::LloydSeedMethod::RANDOM)
        .value("MORTON", homa::Options::LloydSeedMethod::MORTON)
        .value("FPS", homa::Options::LloydSeedMethod::FPS);

    nb::class_<homa::Options>(m, "Options")
        .def(nb::init<>())
        .def_rw("nd_levels", &homa::Options::nd_levels)
        .def_rw("patch_size", &homa::Options::patch_size)
        .def_rw("separator_method", &homa::Options::separator_method)
        .def_rw("direct_separator_max_level",
                &homa::Options::direct_separator_max_level)
        .def_rw("direct_separator_min_nodes",
                &homa::Options::direct_separator_min_nodes)
        .def_rw("compute_etree", &homa::Options::compute_etree)
        .def_rw("use_gpu", &homa::Options::use_gpu)
        .def_rw("local_method", &homa::Options::local_method)
        .def_rw("patch_method", &homa::Options::patch_method)
        .def_rw("lloyd_iters", &homa::Options::lloyd_iters)
        .def_rw("lloyd_seed", &homa::Options::lloyd_seed)
        .def_rw("lloyd_seed_method", &homa::Options::lloyd_seed_method);

    m.def("compute_ordering",
          &py_compute_ordering,
          "indptr"_a,
          "indices"_a,
          "n"_a,
          "options"_a);

    m.attr("HAS_CUDSS")   = kHasCudss;
    m.attr("HAS_CHOLMOD") = kHasCholmod;
    m.attr("HAS_MKL")     = kHasMkl;
    m.attr("HAS_CUDA")    = kHasCudss;  // CUDA runtime present iff cuDSS built

    bind_solver<double>(m, "_SolverD");
    bind_solver<float>(m, "_SolverF");
}
