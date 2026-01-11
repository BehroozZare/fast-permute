// Compatibility header for Eigen/libigl version mismatch
// This brings Eigen::placeholders::all into the Eigen namespace
#pragma once

#include <cassert>
#include <Eigen/Core>

namespace Eigen {
    using Eigen::placeholders::all;
}
