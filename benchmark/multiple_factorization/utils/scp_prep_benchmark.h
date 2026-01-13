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
     * @brief Prepare SCP benchmark data by extracting obj, hessian, and rhs file paths
     * 
     * Expected file naming convention:
     * - OBJ file: *.obj (any .obj file in the folder)
     * - Hessian file: SCP_hessian.mtx (single file)
     * - RHS files: scp_rhs_{number}.mtx
     * 
     * @param folder_address Path to the folder containing SCP benchmark data
     * @param rhs_addresses Output: sorted list of RHS vector file paths
     * @param hessian_address Output: path to the single hessian matrix file
     * @param obj_address Output: path to the .obj mesh file
     */
    void prepare_scp_benchmark_data(const std::string& folder_address,
                                    std::vector<std::string>& rhs_addresses,
                                    std::string& hessian_address,
                                    std::string& obj_address) {
        rhs_addresses.clear();
        hessian_address.clear();
        obj_address.clear();

        // Storage for sorting: (iteration_number, full_path)
        std::vector<std::tuple<int, std::string>> rhs_entries;

        // Regex pattern for RHS files: scp_rhs_{number}.mtx
        std::regex rhs_pattern(R"(scp_rhs_(\d+)\.mtx)");

        for (const auto& entry : std::filesystem::directory_iterator(folder_address)) {
            if (!entry.is_regular_file()) continue;
            
            std::string filename = entry.path().filename().string();
            std::string full_path = entry.path().string();
            std::smatch match;

            // Check for .obj file
            if (filename.size() > 4 && filename.substr(filename.size() - 4) == ".obj") {
                obj_address = full_path;
                continue;
            }

            // Check for hessian file (SCP_hessian.mtx)
            if (filename == "SCP_hessian.mtx") {
                hessian_address = full_path;
                continue;
            }

            // Check for RHS file
            if (std::regex_match(filename, match, rhs_pattern)) {
                int iter_num = std::stoi(match[1].str());
                rhs_entries.emplace_back(iter_num, full_path);
                continue;
            }
        }

        // Sort by iteration number
        std::sort(rhs_entries.begin(), rhs_entries.end());

        // Extract sorted paths
        rhs_addresses.reserve(rhs_entries.size());
        for (const auto& [iter, path] : rhs_entries) {
            rhs_addresses.push_back(path);
        }

        // Validation
        if (obj_address.empty()) {
            spdlog::warn("No .obj file found in folder: {}", folder_address);
        }
        if (hessian_address.empty()) {
            spdlog::warn("No SCP_hessian.mtx file found in folder: {}", folder_address);
        }
        if (rhs_addresses.empty()) {
            spdlog::warn("No RHS files found in folder: {}", folder_address);
        }
    }

}
