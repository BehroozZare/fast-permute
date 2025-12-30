#pragma once

#include <Eigen/Core>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>

/**
 * @brief Reads vertex positions and face indices from separate files.
 * 
 * @param vertex_file Path to file containing vertex positions (comma-separated: x, y, z per line)
 * @param face_file Path to file containing face indices (comma-separated: v1, v2, v3 per line)
 * @param V Output matrix for vertex positions (Nx3)
 * @param F Output matrix for face indices (Mx3)
 * @return true if both files were read successfully, false otherwise
 */
inline bool read_face_and_pos(const std::string& vertex_file,
                               const std::string& face_file,
                               Eigen::MatrixXd& V,
                               Eigen::MatrixXi& F)
{
    // Read vertex positions
    {
        std::ifstream file(vertex_file);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open vertex file: " + vertex_file);
        }

        std::vector<Eigen::Vector3d> vertices;
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            
            std::stringstream ss(line);
            std::string token;
            Eigen::Vector3d vertex;
            
            for (int i = 0; i < 3; ++i) {
                if (!std::getline(ss, token, ',')) {
                    throw std::runtime_error("Invalid vertex format in file: " + vertex_file);
                }
                vertex(i) = std::stod(token);
            }
            vertices.push_back(vertex);
        }

        V.resize(vertices.size(), 3);
        for (size_t i = 0; i < vertices.size(); ++i) {
            V.row(i) = vertices[i];
        }
    }

    // Read face indices
    {
        std::ifstream file(face_file);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open face file: " + face_file);
        }

        std::vector<Eigen::Vector3i> faces;
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            
            std::stringstream ss(line);
            std::string token;
            Eigen::Vector3i face;
            
            for (int i = 0; i < 3; ++i) {
                if (!std::getline(ss, token, ',')) {
                    throw std::runtime_error("Invalid face format in file: " + face_file);
                }
                face(i) = std::stoi(token);
            }
            faces.push_back(face);
        }

        F.resize(faces.size(), 3);
        for (size_t i = 0; i < faces.size(); ++i) {
            F.row(i) = faces[i];
        }
    }

    return true;
}

