#include "homa/ordering.h"
#include "homa/types.h"
#include <gtest/gtest.h>
#include <vector>

// These tests originally used the legacy PatchOrdering class via
// Ordering::create(). After the driver layer was removed they use
// compute_ordering() directly.

namespace {

void ring_graph(int N, std::vector<int> &Gp, std::vector<int> &Gi) {
  Gp.resize(N + 1, 0);
  Gi.clear();
  Gi.reserve(2 * N);
  for (int i = 0; i < N; ++i) {
    int prev = (i - 1 + N) % N;
    int next = (i + 1) % N;
    if (prev < next) {
      Gi.push_back(prev);
      Gi.push_back(next);
    } else {
      Gi.push_back(next);
      Gi.push_back(prev);
    }
    Gp[i + 1] = static_cast<int>(Gi.size());
  }
}

bool is_valid_permutation(const std::vector<int> &perm, int N) {
  if (static_cast<int>(perm.size()) != N)
    return false;
  std::vector<bool> seen(N, false);
  for (int v : perm) {
    if (v < 0 || v >= N || seen[v])
      return false;
    seen[v] = true;
  }
  return true;
}

} // namespace

TEST(PatchOrderingEtree, CpuWithEtreeSucceeds) {
  constexpr int N = 16;
  std::vector<int> Gp, Gi;
  ring_graph(N, Gp, Gi);

  homa::Options opts;
  opts.use_gpu = false;
  opts.patch_size = 4;
  opts.nd_levels = 2;
  opts.use_patch_separator = true;
  opts.local_method = homa::Options::LocalMethod::AMD;
  opts.compute_etree = true;

  homa::OrderingResult result;
  EXPECT_NO_THROW(result =
                      homa::compute_ordering(N, Gp.data(), Gi.data(), opts));

  EXPECT_TRUE(is_valid_permutation(result.perm, N));
  EXPECT_FALSE(result.etree.empty());
}

// compute_ordering() uses the default internal patcher; verify it produces a
// valid result with default options.
TEST(PatchOrderingEtree, DefaultPatcherAccepted) {
  constexpr int N = 16;
  std::vector<int> Gp, Gi;
  ring_graph(N, Gp, Gi);

  homa::Options opts;
  opts.use_gpu = false;
  opts.patch_size = 4;
  opts.nd_levels = 2;

  homa::OrderingResult result;
  EXPECT_NO_THROW(result =
                      homa::compute_ordering(N, Gp.data(), Gi.data(), opts));
  EXPECT_TRUE(is_valid_permutation(result.perm, N));
}
