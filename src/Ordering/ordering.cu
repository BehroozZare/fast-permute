//
// Created by behrooz on 2025-09-29.
//

#include "ordering.h"
#include <cassert>
#include <iostream>
#include "metis_ordering.h"
#include "neutral_ordering.h"
#include "rxmesh_ordering.h"
#include "patch_ordering.h"
#include "parth_ordering.h"

namespace RXMESH_SOLVER {


Ordering *Ordering::create(const DEMO_ORDERING_TYPE type) {
    switch (type) {
        case DEMO_ORDERING_TYPE::METIS:
            return new MetisOrdering();
        case DEMO_ORDERING_TYPE::RXMESH_ND:
            return new RXMeshOrdering();
        case DEMO_ORDERING_TYPE::PATCH_ORDERING:
            return new PatchOrdering();
        case DEMO_ORDERING_TYPE::NEUTRAL:
            return new NeutralOrdering();
        #ifdef USE_PARTH
        case DEMO_ORDERING_TYPE::PARTH:
            return new ParthOrdering();
        #endif
        default:
            std::cerr << "Unknown Ordering type" << std::endl;
            return nullptr;
    }
}

}