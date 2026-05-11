//
//  LinSysSolver.hpp
//  IPC
//
//  Created by Minchen Li on 6/30/18.
//
#pragma once


#include "ordering_factory.h"

namespace homa {


class NeutralOrdering: public Ordering
{
public:
    virtual ~NeutralOrdering(void);

    virtual DEMO_ORDERING_TYPE type() const override;
    virtual std::string typeStr() const override;

    virtual void setGraph(int*              Gp,
                           int*              Gi,
                           int               G_N,
                           int               NNZ) override;

    virtual void compute_permutation(std::vector<int>& perm, std::vector<int>& etree, bool compute_etree = false) override;
};

}  // namespace homa
