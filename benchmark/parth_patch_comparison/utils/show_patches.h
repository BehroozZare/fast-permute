//
// Created by behrooz on 2025-12-23.
//

#ifndef GPU_ORDERING_SHOW_PATCH_H
#define GPU_ORDERING_SHOW_PATCH_H

#include <vector>
#include <array>
#include <Eigen/Core>
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

namespace RXMESH_SOLVER {

    void show_patches(int G_N, int* Gp, int* Gi, Eigen::MatrixXd& V, Eigen::MatrixXi& F, std::vector<int>& node_to_patch){
        //First find the boundary vertices
        std::vector<bool> is_boundary_vertex(G_N, false);
        for(int i = 0; i < G_N; i++){
            int patch_id = node_to_patch[i];
            for(int j = Gp[i]; j < Gp[i + 1]; j++){
                int nbr = Gi[j];
                int nbr_patch_id = node_to_patch[nbr];
                if(nbr_patch_id != patch_id){
                    is_boundary_vertex[i] = true;
                    break;
                }
            }
        }

        // Initialize polyscope
        polyscope::init();

        // Convert Eigen matrices to std::vector format for polyscope
        std::vector<std::array<double, 3>> vertices(V.rows());
        for(int i = 0; i < V.rows(); i++){
            vertices[i] = {V(i, 0), V(i, 1), V(i, 2)};
        }

        std::vector<std::array<int, 3>> faces(F.rows());
        for(int i = 0; i < F.rows(); i++){
            faces[i] = {F(i, 0), F(i, 1), F(i, 2)};
        }

        // Register the surface mesh
        polyscope::SurfaceMesh* mesh = polyscope::registerSurfaceMesh("patch_mesh", vertices, faces);

        // Create vertex colors: black for boundary, gray for non-boundary
        std::vector<std::array<double, 3>> vertex_colors(V.rows());
        for(int i = 0; i < V.rows(); i++){
            if(is_boundary_vertex[i]){
                // Black for boundary vertices
                vertex_colors[i] = {0.0, 0.0, 0.0};
            } else {
                // Gray for non-boundary vertices
                vertex_colors[i] = {0.5, 0.5, 0.5};
            }
        }

        // Add vertex colors to the mesh
        mesh->addVertexColorQuantity("patch_boundaries", vertex_colors);

        // Show the polyscope GUI
        polyscope::show();
    }

}
#endif //GPU_ORDERING_SHOW_PATCH_H
