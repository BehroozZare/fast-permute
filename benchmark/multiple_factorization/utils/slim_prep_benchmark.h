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
     * @brief Prepare SLIM benchmark data by extracting obj, hessian, and gradient file paths
     * 
     * Expected file naming convention:
     * - OBJ file: *.obj (any .obj file in the folder, excluding *_uv.obj)
     * - Hessian files: tinyAD_<mesh_name>_H_<iter>
     * - Gradient files: tinyAD_<mesh_name>_g_<iter>
     * 
     * Note: RHS vectors should be generated as random dense vectors in the benchmark.
     * 
     * @param folder_address Path to the folder containing SLIM benchmark data
     * @param hessian_addresses Output: sorted list of hessian matrix file paths
     * @param gradient_addresses Output: sorted list of gradient vector file paths
     * @param obj_address Output: path to the .obj mesh file
     */
    void prepare_benchmark_data(const std::string& folder_address,
                                std::vector<std::string>& hessian_addresses,
                                std::vector<std::string>& gradient_addresses,
                                std::string& obj_address) {
        hessian_addresses.clear();
        gradient_addresses.clear();
        obj_address.clear();

        // Storage for sorting: (iteration_number, full_path)
        std::vector<std::tuple<int, std::string>> hess_entries;
        std::vector<std::tuple<int, std::string>> grad_entries;

        // Regex pattern for tinyAD files: tinyAD_<mesh_name>_H_<iter> or tinyAD_<mesh_name>_g_<iter>
        std::regex hess_pattern(R"(tinyAD_.*_H_(\d+))");
        std::regex grad_pattern(R"(tinyAD_.*_g_(\d+))");

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

            // Check for hessian file (tinyAD_*_H_<iter>)
            if (std::regex_match(filename, match, hess_pattern)) {
                int iter_num = std::stoi(match[1].str());
                hess_entries.emplace_back(iter_num, full_path);
                continue;
            }

            // Check for gradient file (tinyAD_*_g_<iter>)
            if (std::regex_match(filename, match, grad_pattern)) {
                int iter_num = std::stoi(match[1].str());
                grad_entries.emplace_back(iter_num, full_path);
                continue;
            }
        }

        // Sort by iteration number
        std::sort(hess_entries.begin(), hess_entries.end());
        std::sort(grad_entries.begin(), grad_entries.end());

        // Extract sorted paths for hessians
        hessian_addresses.reserve(hess_entries.size());
        for (const auto& [iter, path] : hess_entries) {
            hessian_addresses.push_back(path);
        }

        // Extract sorted paths for gradients
        gradient_addresses.reserve(grad_entries.size());
        for (const auto& [iter, path] : grad_entries) {
            gradient_addresses.push_back(path);
        }

        // Validation
        if (obj_address.empty()) {
            spdlog::warn("No .obj file found in folder: {}", folder_address);
        }
        if (hessian_addresses.empty()) {
            spdlog::warn("No hessian files (tinyAD_*_H_*) found in folder: {}", folder_address);
        }
        if (gradient_addresses.empty()) {
            spdlog::warn("No gradient files (tinyAD_*_g_*) found in folder: {}", folder_address);
        }
    }

}
