#include "homa/ordering/amd_order_helper.h"

#include <amd.h>

namespace homa::detail {

int amd_local_order(int n, int* Ap, int* Ai, int* perm)
{
    return amd_order(n, Ap, Ai, perm, nullptr, nullptr);
}

}  // namespace homa::detail
