#pragma once

#include <fstream>
#include <vector>
#include <string>

namespace RXMESH_SOLVER {
    void save_etree_to_text(const std::vector<int>& etree, const std::string& filename){
        std::ofstream file(filename);
        for(size_t i = 0; i < etree.size(); i++){
            file << etree[i] << std::endl;
        }
        file.close();
    }
}