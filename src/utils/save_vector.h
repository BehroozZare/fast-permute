#pragma once

#include <fstream>
#include <vector>
#include <string>



namespace RXMESH_SOLVER {
    void save_vector_to_file(const std::vector<int>& vector, const std::string& filename){
        std::ofstream file(filename);
        for(size_t i = 0; i < vector.size(); i++){
            file << vector[i] << std::endl;
        }
        file.close();
    }
}