//
// Created by behrooz zare on 2024-04-25.
//

#include "createPatch.h"

namespace PARTHDEMO {

void createPatch(int fid,
                 double ring_size,
                 Eigen::VectorXi& SelectedFaces,
                 Eigen::MatrixXi& F,
                 Eigen::MatrixXd& V) {
    // Build vertex-to-face adjacency
    Eigen::VectorXi Fp;
    Eigen::VectorXi Fi;
    igl::vertex_triangle_adjacency(F, V.rows(), Fi, Fp);

    std::vector<int> Nfaces;
    std::queue<int> first_ring, second_ring;
    std::queue<int>* empty_ring;
    std::queue<int>* full_ring;
    std::queue<int>* tmp;
    std::vector<bool> visited(F.rows(), false);

    // Start BFS from the center face
    first_ring.push(fid);
    full_ring = &first_ring;
    empty_ring = &second_ring;

    while ((Nfaces.size() * 1.0 / F.rows()) < ring_size) {
        while (!(*full_ring).empty()) {
            int curr_f = (*full_ring).front();
            (*full_ring).pop();
            // For all vertices of the current face
            for (int v_ptr = 0; v_ptr < 3; v_ptr++) {
                int v = F(curr_f, v_ptr);
                // For all faces adjacent to this vertex
                for (int f_ptr = Fp[v]; f_ptr < Fp[v + 1]; f_ptr++) {
                    int f = Fi(f_ptr);
                    if (!visited[f]) {
                        visited[f] = true;
                        Nfaces.emplace_back(f);
                        (*empty_ring).push(f);
                    }
                }
            }
        }
        // Swap ring buffers
        tmp = full_ring;
        full_ring = empty_ring;
        empty_ring = tmp;
    }

    // Assign Nfaces into the SelectedFaces
    SelectedFaces.resize(Nfaces.size());
    for (size_t i = 0; i < Nfaces.size(); i++) {
        SelectedFaces(i) = Nfaces[i];
    }
}

}  // namespace PARTHDEMO
