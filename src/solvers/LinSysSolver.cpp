#include <homa/solvers/LinSysSolver.h>
#include <iostream>

#ifdef USE_CHOLMOD
#include <homa/solvers/CHOLMODSolver.h>
#endif

#ifdef USE_CUDSS
#include <homa/solvers/CUDSSSolver.h>
#endif


#ifdef USE_MKL
#include <homa/solvers/MKLSolver.h>
#endif


namespace homa {

    LinSysSolver *LinSysSolver::create(const LinSysSolverType type) {
        switch (type) {


#ifdef USE_CHOLMOD
            case LinSysSolverType::CPU_CHOLMOD:
                return new CHOLMODSolver();
#endif

#ifdef USE_CUDSS
            case LinSysSolverType::GPU_CUDSS:
                return new CUDSSSolver();
#endif


#ifdef USE_MKL
            case LinSysSolverType::CPU_MKL:
                return new MKLSolver();
#endif
            default:
                std::cerr << "Uknown linear system solver type" << std::endl;
                return nullptr;
        }
    }


} // namespace homa
