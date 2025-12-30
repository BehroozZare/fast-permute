#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <parth/parth.h>
#include <queue>
#include <unordered_set>
#include <spdlog/spdlog.h>
#include <metis.h>

namespace RXMESH_SOLVER {

    void create_patch_with_metis(int A_n, int* Ap, int* Ai, int DIM, int patch_size, std::vector<int>& node_to_patch){
        // Use Parth to compress the matrix into a graph
        PARTH::ParthAPI parth;
        parth.setMatrix(A_n, Ap, Ai, DIM);
        
        int G_N = parth.M_n;
        int* Gp = parth.Mp;
        int* Gi = parth.Mi;

        // Convert CSR format to METIS idx_t type
        std::vector<idx_t> xadj(G_N + 1);
        std::vector<idx_t> adjncy(Gp[G_N]);

        for (int i = 0; i <= G_N; ++i) {
            xadj[i] = static_cast<idx_t>(Gp[i]);
        }

        for (int i = 0; i < Gp[G_N]; ++i) {
            adjncy[i] = static_cast<idx_t>(Gi[i]);
        }

        // Set up METIS options
        idx_t options[METIS_NOPTIONS];
        METIS_SetDefaultOptions(options);
        options[METIS_OPTION_PTYPE] = METIS_PTYPE_KWAY;
        options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;  // Total communication volume minimization
        options[METIS_OPTION_NUMBERING] = 0;  // C-style numbering (0-based)
        options[METIS_OPTION_CONTIG] = 0;
        options[METIS_OPTION_COMPRESS] = 0;

        // METIS parameters
        idx_t nvtxs = static_cast<idx_t>(G_N);  // Number of vertices in the graph
        idx_t ncon = 1;                          // Number of balancing constraints
        idx_t* vwgt = NULL;                      // Vertex weights (NULL for equal weights)
        idx_t* vsize = NULL;                     // Vertex sizes (NULL for equal sizes)
        idx_t* adjwgt = NULL;                    // Edge weights (NULL for equal weights)
        
        // Calculate number of partitions based on patch_size
        idx_t nparts = (G_N + patch_size - 1) / patch_size;
        if (nparts < 2) {
            nparts = 1;
        }

        real_t* tpwgts = NULL;  // Target partition weights (NULL for equal)
        real_t* ubvec = NULL;   // Load imbalance tolerance (NULL for default 1.03)
        idx_t objval = 0;       // Edge-cut or total communication volume
        
        std::vector<idx_t> part(nvtxs, 0);  // Partition assignment for each vertex

        // Handle degenerate case with single partition
        if (nparts <= 1) {
            node_to_patch.resize(G_N);
            std::fill(node_to_patch.begin(), node_to_patch.end(), 0);
            spdlog::info("METIS: Single partition, all {} nodes assigned to partition 0", G_N);
            return;
        }

        // Call METIS k-way partitioning
        int metis_status = METIS_PartGraphKway(&nvtxs,
                                       c        &ncon,
                                               xadj.data(),
                                               adjncy.data(),
                                               vwgt,
                                               vsize,
                                               adjwgt,
                                               &nparts,
                                               tpwgts,
                                               ubvec,
                                               options,
                                               &objval,
                                               part.data());

        // Check for METIS errors
        if (metis_status == METIS_ERROR_INPUT) {
            spdlog::error("METIS ERROR: Invalid input");
            throw std::runtime_error("METIS_PartGraphKway failed with input error");
        } else if (metis_status == METIS_ERROR_MEMORY) {
            spdlog::error("METIS ERROR: Memory allocation failed");
            throw std::runtime_error("METIS_PartGraphKway failed with memory error");
        } else if (metis_status == METIS_ERROR) {
            spdlog::error("METIS ERROR: Unknown error");
            throw std::runtime_error("METIS_PartGraphKway failed with unknown error");
        }

        // Copy partition assignments to output
        node_to_patch.resize(G_N);
        for (int i = 0; i < G_N; ++i) {
            node_to_patch[i] = static_cast<int>(part[i]);
        }

        spdlog::info("METIS k-way partitioning completed: {} nodes -> {} partitions (target patch_size: {})",
                     G_N, nparts, patch_size);
    }

}