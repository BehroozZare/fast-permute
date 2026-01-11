//
// Created by behrooz on 2025-12-23.
//

#ifndef GPU_ORDERING_UPDATE_PERM_WITH_BOUNDARY_H
#define GPU_ORDERING_UPDATE_PERM_WITH_BOUNDARY_H

#include <vector>
#include <array>
#include <Eigen/Core>


namespace RXMESH_SOLVER {

    void update_perm_with_boundary(std::vector<int>& perm, std::vector<int>& boundary){
        //First find old_to_new_label 
        std::vector<int> old_to_new_label(perm.size(), -1);
        for(int i = 0; i < perm.size(); i++){
            old_to_new_label[perm[i]] = i;
        }

        //Mark those that are in bounadry
        std::vector<char> is_boundary(perm.size(), 0);
        for(int i = 0; i < boundary.size(); i++){
            is_boundary[boundary[i]] = 1;
        }

        //Create an offset update that update all the labels based on removed labels
        std::vector<int> offset_update(perm.size(), 0);
        offset_update[0] = is_boundary[0];
        for(int i = 1; i < is_boundary.size(); i++){
            if(is_boundary[i]){
                int label = old_to_new_label[i];
                offset_update[label] = 1;
            }
        }

        //prefix scan
        for(int i = 1; i < offset_update.size(); i++){
            offset_update[i] += offset_update[i - 1];
        }

        //Update the perm
        std::vector<int> new_old_to_new_label(perm.size(), -1);
        int cnt = 0;
        for(int i = 0; i < perm.size(); i++){
            if(is_boundary[i]){
                continue;
            }
            int old_label = old_to_new_label[i];
            int new_label = old_label - offset_update[old_label];
            new_old_to_new_label[cnt] = new_label;
            cnt++;
        }
        assert(cnt == perm.size() - boundary.size());
        
        //Compute the perm
        perm.clear();
        perm.resize(old_to_new_label.size() - boundary.size());
        for(int i = 0; i < perm.size(); i++){
            perm[new_old_to_new_label[i]] = i;
        }
    }
}
#endif //GPU_ORDERING_SHOW_PATCH_H
