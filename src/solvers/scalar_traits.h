#pragma once

// Internal header: per-backend precision dispatch traits.
// Keep all backend includes guarded so a TU only pulls in what it actually
// needs. Public headers must NOT include this header.

#include <type_traits>

namespace homa::detail {

template <class Scalar>
inline constexpr bool is_supported_scalar_v =
    std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>;

} // namespace homa::detail


#ifdef USE_CUDSS
#include <library_types.h>
namespace homa::detail {

template <class Scalar> struct cudss_dtype;
template <> struct cudss_dtype<float>  { static constexpr cudaDataType_t value = CUDA_R_32F; };
template <> struct cudss_dtype<double> { static constexpr cudaDataType_t value = CUDA_R_64F; };

template <class Scalar>
inline constexpr cudaDataType_t cudss_dtype_v = cudss_dtype<Scalar>::value;

} // namespace homa::detail
#endif // USE_CUDSS


#ifdef USE_CHOLMOD
#include <cholmod.h>
namespace homa::detail {

// SuiteSparse 7.0+ exposes CHOLMOD_SINGLE / CHOLMOD_DOUBLE as the dtype values
// stored in cholmod_common::dtype. Newly allocated sparse / dense / factor
// objects inherit that precision.
template <class Scalar> struct cholmod_dtype;
template <> struct cholmod_dtype<float>  { static constexpr int value = CHOLMOD_SINGLE; };
template <> struct cholmod_dtype<double> { static constexpr int value = CHOLMOD_DOUBLE; };

// SuiteSparse 7+ passes xtype+dtype as a single xdtype argument to
// cholmod_allocate_sparse / cholmod_allocate_dense (CHOLMOD_REAL + dtype).
template <class Scalar>
inline constexpr int cholmod_xdtype_v =
    CHOLMOD_REAL + cholmod_dtype<Scalar>::value;

} // namespace homa::detail
#endif // USE_CHOLMOD


#ifdef USE_MKL
namespace homa::detail {

// PARDISO uses iparm[27] to switch between double (0) and single (1) precision
// matrix values; the caller still passes the value buffer through a void* so
// the function signatures themselves do not change with Scalar.
template <class Scalar>
inline constexpr int mkl_iparm27_v = std::is_same_v<Scalar, float> ? 1 : 0;

} // namespace homa::detail
#endif // USE_MKL
