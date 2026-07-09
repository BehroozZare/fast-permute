#pragma once
#include <cmath>

#include <Eigen/Core>

#include <igl/doublearea.h>
#include <igl/remove_duplicate_vertices.h>
#include <igl/remove_unreferenced.h>
#include <igl/resolve_duplicated_faces.h>

#include <spdlog/spdlog.h>

inline bool cleanup_mesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F)
{
    constexpr double merge_epsilon_rel   = 1e-8;
    constexpr double min_double_area_rel = 1e-20;

    if (V.rows() == 0 || F.rows() == 0 || F.cols() != 3) {
        return false;
    }

    const Eigen::Index input_vertices = V.rows();
    const Eigen::Index input_faces    = F.rows();

    double bbox_diag = (V.colwise().maxCoeff() - V.colwise().minCoeff()).norm();
    if (!std::isfinite(bbox_diag)) {
        bbox_diag = 0.0;
    }

    const double merge_epsilon   = merge_epsilon_rel * bbox_diag;
    const double min_double_area = min_double_area_rel * bbox_diag * bbox_diag;

    Eigen::MatrixXd V_welded;
    Eigen::MatrixXi F_welded;
    Eigen::VectorXi SVI, SVJ;
    igl::remove_duplicate_vertices(
        V, F, merge_epsilon, V_welded, SVI, SVJ, F_welded);
    V = std::move(V_welded);
    F = std::move(F_welded);

    Eigen::MatrixXi F_valid(F.rows(), 3);
    Eigen::Index    valid_faces = 0;
    for (Eigen::Index i = 0; i < F.rows(); ++i) {
        if (F(i, 0) == F(i, 1) || F(i, 0) == F(i, 2) || F(i, 1) == F(i, 2)) {
            continue;
        }
        F_valid.row(valid_faces++) = F.row(i);
    }
    const Eigen::Index degenerate_faces_removed = F.rows() - valid_faces;
    F = F_valid.topRows(valid_faces).eval();
    if (F.rows() == 0) {
        return false;
    }

    const Eigen::Index faces_before_duplicate_resolve = F.rows();
    Eigen::MatrixXi    F_unique;
    Eigen::VectorXi    duplicate_face_map;
    igl::resolve_duplicated_faces(F, F_unique, duplicate_face_map);
    F = std::move(F_unique);
    const Eigen::Index duplicate_faces_removed =
        faces_before_duplicate_resolve - F.rows();
    if (F.rows() == 0) {
        return false;
    }

    Eigen::VectorXd double_area;
    igl::doublearea(V, F, double_area);

    Eigen::MatrixXi F_area(F.rows(), 3);
    Eigen::Index    area_faces = 0;
    for (Eigen::Index i = 0; i < F.rows(); ++i) {
        if (!std::isfinite(double_area(i)) ||
            double_area(i) <= min_double_area) {
            continue;
        }
        F_area.row(area_faces++) = F.row(i);
    }
    const Eigen::Index small_area_faces_removed = F.rows() - area_faces;
    F = F_area.topRows(area_faces).eval();
    if (F.rows() == 0) {
        return false;
    }

    Eigen::MatrixXd V_compact;
    Eigen::MatrixXi F_compact;
    Eigen::VectorXi I, J;
    igl::remove_unreferenced(V, F, V_compact, F_compact, I, J);
    V = std::move(V_compact);
    F = std::move(F_compact);

    spdlog::info(
        "Mesh cleanup: V {} -> {}, F {} -> {} "
        "(eps={}, min_double_area={}, degenerate={}, duplicate={}, "
        "small_area={})",
        input_vertices,
        V.rows(),
        input_faces,
        F.rows(),
        merge_epsilon,
        min_double_area,
        degenerate_faces_removed,
        duplicate_faces_removed,
        small_area_faces_removed);

    return V.rows() > 0 && F.rows() > 0;
}