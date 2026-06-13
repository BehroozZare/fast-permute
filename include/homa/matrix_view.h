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

struct SparseMatrixView
{
    int rows = 0;
    int cols = 0;
    int nnz  = 0;

    int*    outer  = nullptr;
    int*    inner  = nullptr;
    double* values = nullptr;

    SparseFormat   format   = SparseFormat::CSC;
    MemoryLocation location = MemoryLocation::Host;
};

struct DenseMatrixView
{
    double* values      = nullptr;
    int     rows        = 0;
    int     cols        = 1;
    int     leading_dim = 0;

    MemoryLocation location = MemoryLocation::Host;
};

} // namespace homa
