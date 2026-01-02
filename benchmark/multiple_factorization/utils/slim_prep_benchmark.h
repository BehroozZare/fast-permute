#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <regex>
#include <tuple>
#include <spdlog/spdlog.h>


namespace RXMESH_SOLVER {

    /**
     * @brief Prepare SLIM benchmark data by extracting obj, gradient, and hessian file paths
     * 
     * Expected file naming convention:
     * - OBJ file: {meshname}.obj
     * - Gradient files: {meshname}_slim_grad_{number}.mtx
     * - Hessian files: {meshname}_slim_hess_{number}.mtx
     * 
     * @param folder_address Path to the folder containing SLIM benchmark data
     * @param grad_addresses Output: sorted list of gradient matrix file paths
     * @param hessian_addresses Output: sorted list of hessian matrix file paths
     * @param obj_address Output: path to the .obj mesh file
     */
    void prepare_benchmark_data(const std::string& folder_address,
                                std::vector<std::string>& grad_addresses,
                                std::vector<std::string>& hessian_addresses,
                                std::string& obj_address) {
        grad_addresses.clear();
        hessian_addresses.clear();
        obj_address.clear();

        // Storage for sorting: (iteration_number, full_path)
        std::vector<std::tuple<int, std::string>> grad_entries;
        std::vector<std::tuple<int, std::string>> hess_entries;

        // Regex patterns for gradient and hessian files
        // Pattern: {meshname}_slim_grad_{number}.mtx
        std::regex grad_pattern(R"((.+)_slim_grad_(\d+)\.mtx)");
        // Pattern: {meshname}_slim_hess_{number}.mtx
        std::regex hess_pattern(R"((.+)_slim_hess_(\d+)\.mtx)");

        for (const auto& entry : std::filesystem::directory_iterator(folder_address)) {
            if (!entry.is_regular_file()) continue;
            
            std::string filename = entry.path().filename().string();
            std::string full_path = entry.path().string();
            std::smatch match;

            // Check for .obj file (skip UV versions like *_uv.obj)
            if (filename.size() > 4 && filename.substr(filename.size() - 4) == ".obj") {
                // Skip files ending with _uv.obj
                if (filename.size() > 7 && filename.substr(filename.size() - 7) == "_uv.obj") {
                    continue;
                }
                obj_address = full_path;
                continue;
            }

            // Check for gradient file
            if (std::regex_match(filename, match, grad_pattern)) {
                int iter_num = std::stoi(match[2].str());
                grad_entries.emplace_back(iter_num, full_path);
                continue;
            }

            // Check for hessian file
            if (std::regex_match(filename, match, hess_pattern)) {
                int iter_num = std::stoi(match[2].str());
                hess_entries.emplace_back(iter_num, full_path);
                continue;
            }
        }

        // Sort by iteration number
        std::sort(grad_entries.begin(), grad_entries.end());
        std::sort(hess_entries.begin(), hess_entries.end());

        // Extract sorted paths
        grad_addresses.reserve(grad_entries.size());
        for (const auto& [iter, path] : grad_entries) {
            grad_addresses.push_back(path);
        }

        hessian_addresses.reserve(hess_entries.size());
        for (const auto& [iter, path] : hess_entries) {
            hessian_addresses.push_back(path);
        }

        // Validation
        if (obj_address.empty()) {
            spdlog::warn("No .obj file found in folder: {}", folder_address);
        }
        if (grad_addresses.empty()) {
            spdlog::warn("No gradient files found in folder: {}", folder_address);
        }
        if (hessian_addresses.empty()) {
            spdlog::warn("No hessian files found in folder: {}", folder_address);
        }
        if (grad_addresses.size() != hessian_addresses.size()) {
            spdlog::warn("Mismatch: {} gradient files vs {} hessian files",
                         grad_addresses.size(), hessian_addresses.size());
        }
    }

}
