#pragma once

namespace homa {

enum class MemoryLocation
{
    Host,
    Device
};

enum class SparseFormat
{
    CSC,
    CSR
};

template <class Scalar>
struct SparseMatrixView
{
    int rows = 0;
    int cols = 0;
    int nnz  = 0;

    int*    outer  = nullptr;
    int*    inner  = nullptr;
    Scalar* values = nullptr;

    SparseFormat   format   = SparseFormat::CSC;
    MemoryLocation location = MemoryLocation::Host;
};

template <class Scalar>
struct DenseMatrixView
{
    Scalar* values      = nullptr;
    int     rows        = 0;
    int     cols        = 1;
    int     leading_dim = 0;

    MemoryLocation location = MemoryLocation::Host;
};

using SparseMatrixViewD = SparseMatrixView<double>;
using SparseMatrixViewF = SparseMatrixView<float>;
using DenseMatrixViewD  = DenseMatrixView<double>;
using DenseMatrixViewF  = DenseMatrixView<float>;

} // namespace homa
