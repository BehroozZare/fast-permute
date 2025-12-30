#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <regex>
#include <tuple>


namespace RXMESH_SOLVER {
    std::vector<std::string> prepare_benchmark_data(std::string folder_address){
        std::vector<std::tuple<int, int, std::string>> entries;
        std::regex pattern(R"(hessian_(\d+)_(\d+)_last_IPC\.mtx)");
        
        for (const auto& entry : std::filesystem::directory_iterator(folder_address)) {
            if (!entry.is_regular_file()) continue;
            std::string filename = entry.path().filename().string();
            std::smatch match;
            if (std::regex_match(filename, match, pattern)) {
                int frame = std::stoi(match[1].str());
                int iter = std::stoi(match[2].str());
                entries.emplace_back(frame, iter, entry.path().string());
            }
        }
        
        std::sort(entries.begin(), entries.end());
        
        std::vector<std::string> result;
        result.reserve(entries.size());
        for (const auto& [frame, iter, path] : entries) {
            result.push_back(path);
        }
        return result;
    }

    bool is_graph_equal(int* Gp_prev, int* Gi_prev, int* Gp_curr, int* Gi_curr, int G_N, int G_N_prev){
        if (G_N != G_N_prev) return false;
        for(int i = 0; i < G_N; i++){
            if(Gp_prev[i] != Gp_curr[i]){
                return false;
            }
        }
        int G_NNZ = Gp_curr[G_N];
        for(int i = 0; i < G_NNZ; i++){
            if(Gi_prev[i] != Gi_curr[i]){
                return false;
            }
        }
        return true;
    }


}