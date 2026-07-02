#pragma once
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <string>

#include <homa/solvers/LinSysSolver.h>
#include <homa/types.h>
#include <spdlog/spdlog.h>

#ifdef USE_CUDSS
#include <cuda_runtime.h>
#include "homa/utils/cuda_error_handler.h"
#endif

struct StageTimes {
    double ordering_ms  = 0.0;
    double reorder_ms   = 0.0;
    double analysis_ms  = 0.0;
    double factorize_ms = 0.0;
    double solve_ms     = 0.0;
    double residual     = 0.0;
};

struct BenchmarkRecord {
    std::string matrix_path;
    std::string solver_name;
    std::string precision;   // "float" or "double"
    int         n          = 0;
    long long   nnz        = 0;
    int         patch_size = 0;
    StageTimes  default_times;
    StageTimes  homa_times;
    double      default_total_ms = 0.0;
    double      homa_total_ms    = 0.0;
};

inline std::string json_escape(const std::string& value)
{
    std::string out;
    out.reserve(value.size() + 2);
    for (unsigned char c : value) {
        switch (c) {
        case '"':  out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\b': out += "\\b";  break;
        case '\f': out += "\\f";  break;
        case '\n': out += "\\n";  break;
        case '\r': out += "\\r";  break;
        case '\t': out += "\\t";  break;
        default:
            if (c < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            } else {
                out.push_back(static_cast<char>(c));
            }
        }
    }
    return out;
}

namespace detail {

inline void write_stage_times(std::ostream&     out,
                              const StageTimes& s,
                              double            total_ms)
{
    out << "    \"ordering_ms\":  " << s.ordering_ms  << ",\n"
        << "    \"reorder_ms\":   " << s.reorder_ms   << ",\n"
        << "    \"analysis_ms\":  " << s.analysis_ms  << ",\n"
        << "    \"factorize_ms\": " << s.factorize_ms << ",\n"
        << "    \"solve_ms\":     " << s.solve_ms     << ",\n"
        << "    \"total_ms\":     " << total_ms       << ",\n"
        << "    \"residual\":     " << s.residual     << "\n";
}

} // namespace detail

inline bool write_results_json(const std::string&     path,
                               const BenchmarkRecord& rec)
{
    std::ofstream out(path);
    if (!out) {
        spdlog::warn("benchmark_json: failed to open output file: {}", path);
        return false;
    }

    const std::string stem =
        std::filesystem::path(rec.matrix_path).stem().string();

    out << std::setprecision(9);

    out << "{\n"
        << "  \"matrix\": {\n"
        << "    \"path\": \"" << json_escape(rec.matrix_path) << "\",\n"
        << "    \"name\": \"" << json_escape(stem)            << "\",\n"
        << "    \"n\": "      << rec.n                        << ",\n"
        << "    \"nnz\": "    << rec.nnz                      << "\n"
        << "  },\n"
        << "  \"solver\": \""    << json_escape(rec.solver_name) << "\",\n"
        << "  \"precision\": \"" << json_escape(rec.precision)   << "\",\n"
        << "  \"patch_size\": "  << rec.patch_size                << ",\n"
        << "  \"default\": {\n";
    detail::write_stage_times(out, rec.default_times, rec.default_total_ms);
    out << "  },\n"
        << "  \"homa\": {\n";
    detail::write_stage_times(out, rec.homa_times, rec.homa_total_ms);
    out << "  }\n"
        << "}\n";

    if (!out.good()) {
        spdlog::warn("benchmark_json: write to {} did not complete cleanly", path);
        return false;
    }
    return true;
}


#ifdef USE_CUDSS
struct GpuTimer {
    GpuTimer()
    {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() { cudaEventRecord(start_); }

    double stop_ms()
    {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return static_cast<double>(ms);
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};
#endif

std::string to_lower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
        });
    return value;
}


std::string solver_display_name(homa::LinSysSolverType solver_type)
{
    switch (solver_type) {
    case homa::LinSysSolverType::CPU_CHOLMOD:
        return "CHOLMOD";
    case homa::LinSysSolverType::CPU_MKL:
        return "MKL PARDISO";
    case homa::LinSysSolverType::GPU_CUDSS:
        return "cuDSS";
    default:
        return "Unknown";
    }
}

homa::LinSysSolverType solver_type_from_name(const std::string& solver_name)
{
    const std::string name = to_lower(solver_name);

    if (name == "cholmod" || name == "cpu_cholmod") {
        return homa::LinSysSolverType::CPU_CHOLMOD;
    }
    if (name == "mkl" || name == "pardiso" || name == "cpu_mkl") {
        return homa::LinSysSolverType::CPU_MKL;
    }
    if (name == "cudss" || name == "gpu_cudss") {
        return homa::LinSysSolverType::GPU_CUDSS;
    }

    throw std::invalid_argument("Unknown solver: " + solver_name);
}

homa::Options::SeparatorMethod separator_method_from_name(const std::string& method_name)
{
    const std::string name = to_lower(method_name);

    if (name == "auto" || name == "heuristic") {
        return homa::Options::SeparatorMethod::AUTO;
    }
    if (name == "quotient" || name == "patch") {
        return homa::Options::SeparatorMethod::QUOTIENT;
    }
    if (name == "direct" || name == "metis" || name == "direct_metis") {
        return homa::Options::SeparatorMethod::DIRECT_METIS;
    }

    throw std::invalid_argument("Unknown separator method: " + method_name +
        " (expected 'auto', 'quotient', or 'direct')");
}

bool is_matrix_market_symmetric(const std::string& filename)
{
    std::ifstream in(filename);
    std::string   header;
    std::getline(in, header);
    header = to_lower(header);
    return header.find(" symmetric") != std::string::npos ||
        header.find(" hermitian") != std::string::npos;
}