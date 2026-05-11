#pragma once
#include <vector>

namespace homa {

/// Interface for supplying a node-to-patch assignment.
/// Implement this to plug in a custom patching backend.
class IPatcher {
public:
    virtual ~IPatcher() = default;

    /// Compute node_to_patch[i] = patch index for graph node i.
    /// @param n            number of graph nodes
    /// @param Gp           CSR row pointers (size n+1)
    /// @param Gi           CSR column indices (size Gp[n])
    /// @param node_to_patch  output, size n
    virtual void compute(int n, const int* Gp, const int* Gi,
                         std::vector<int>& node_to_patch) = 0;
};

} // namespace homa
