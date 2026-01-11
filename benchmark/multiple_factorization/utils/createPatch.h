//
// Created by behrooz zare on 2024-04-25.
//

#pragma once

#include <Eigen/Core>
#include <igl/vertex_triangle_adjacency.h>
#include <queue>
#include <vector>

namespace PARTHDEMO {

/**
 * @brief Creates a patch of faces around a center face using BFS
 * @param fid Center face ID to start the patch from
 * @param ring_size Fraction of total faces to include in patch (0.0 to 1.0)
 * @param SelectedFaces Output vector of selected face IDs
 * @param F Face matrix (num_faces x 3)
 * @param V Vertex matrix (num_vertices x 3)
 */
void createPatch(int fid,
                 double ring_size,
                 Eigen::VectorXi& SelectedFaces,
                 Eigen::MatrixXi& F,
                 Eigen::MatrixXd& V);

}  // namespace PARTHDEMO
