#pragma once

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <igl/cotmatrix_entries.h>
#include <cassert>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <vector>

namespace RXMESH_SOLVER
{
    /**
     * @brief Projects a symmetric matrix to be positive semi-definite.
     * Flips negative eigenvalues to positive.
     */
    inline void MakeSymmetricMatrixSPD(Eigen::MatrixXd& matrix) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(matrix);
        Eigen::VectorXd eigenValues = eigensolver.eigenvalues();
        double eps = 1e-8; // or something scale-dependent

        for (int i = 0; i < eigenValues.rows(); ++i) {
            if (eigenValues[i] < eps) {
                eigenValues[i] = eps;
            }
        }

        matrix = eigensolver.eigenvectors()
               * eigenValues.asDiagonal()
               * eigensolver.eigenvectors().transpose();
    }

    /**
     * @brief Computes a symmetric positive definite (SPD) cotangent matrix.
     * 
     * For each element (triangle or tetrahedron), this function:
     * 1. Builds a local stiffness matrix from cotangent weights using -L convention
     *    (positive off-diagonal, negative diagonal)
     * 2. Projects the local matrix to be positive semi-definite by flipping
     *    negative eigenvalues to positive
     * 3. Assembles the projected contributions into the global sparse matrix
     * 
     * The output is SPD (symmetric positive definite), suitable for Cholesky factorization.
     * 
     * @param V  #V by 3 matrix of vertex positions
     * @param F  #F by 3 (triangles) or #F by 4 (tetrahedra) matrix of element indices
     * @param L  Output SPD sparse matrix
     */
    inline void computeSPD_cot_matrix(
            const Eigen::MatrixXd & V,
            const Eigen::MatrixXi & F,
            Eigen::SparseMatrix<double>& L)
    {
        using namespace Eigen;
        
        L.resize(V.rows(), V.rows());
        Matrix<int,Dynamic,2> edges;
        int simplex_size = F.cols();
        
        assert(simplex_size == 3 || simplex_size == 4);
        if(simplex_size == 3)
        {
            L.reserve(10 * V.rows());
            edges.resize(3,2);
            edges << 1,2,
                     2,0,
                     0,1;
        }
        else if(simplex_size == 4)
        {
            L.reserve(17 * V.rows());
            edges.resize(6,2);
            edges << 1,2,
                     2,0,
                     0,1,
                     3,0,
                     3,1,
                     3,2;
        }
        else
        {
            return;
        }
        
        if (V.rows() == 0 || V.cols() == 0) {
            throw std::runtime_error("SPD_cot_matrix: V matrix is empty");
        }
        if (F.rows() == 0 || F.cols() == 0) {
            throw std::runtime_error("SPD_cot_matrix: F matrix is empty");
        }
        if (V.cols() != 3) {
            throw std::runtime_error("SPD_cot_matrix: V must have 3 columns (3D vertices)");
        }
        
        // Gather cotangents
        Eigen::MatrixXd C;
        igl::cotmatrix_entries(V, F, C);

        std::vector<Triplet<double>> IJV;
        IJV.reserve(F.rows() * edges.rows() * 4);
        
        Eigen::MatrixXd C_matrix(3,3);
        for(int i = 0; i < F.rows(); i++)
        {
            if(simplex_size == 3){
                C_matrix.setZero();
                for(int e = 0; e < edges.rows(); e++)
                {
                    int source = edges(e,0);
                    int dest = edges(e,1);
                    C_matrix(source,dest) += C(i,e);
                    C_matrix(dest,source) += C(i,e);
                    C_matrix(source,source) += -C(i,e);
                    C_matrix(dest,dest) += -C(i,e);
                }
                MakeSymmetricMatrixSPD(C_matrix);
                for(int j = 0; j < C_matrix.rows(); j++)
                    for(int k = 0; k < C_matrix.cols(); k++)
                        IJV.push_back(Triplet<double>(F(i,j), F(i,k), C_matrix(j,k)));
            } else {
                for(int e = 0; e < edges.rows(); e++)
                {
                    int source = F(i, edges(e,0));
                    int dest = F(i, edges(e,1));
                    IJV.push_back(Triplet<double>(source, dest, -C(i,e)));
                    IJV.push_back(Triplet<double>(dest, source, -C(i,e)));
                    IJV.push_back(Triplet<double>(source, source, C(i,e)));
                    IJV.push_back(Triplet<double>(dest, dest, C(i,e)));
                }
            }
        }
        L.setFromTriplets(IJV.begin(), IJV.end());
    }
}
