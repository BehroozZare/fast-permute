#include <igl/read_triangle_mesh.h>
#include <igl/hessian_energy.h>
#include <igl/curved_hessian_energy.h>
#include <igl/massmatrix.h>
#include <igl/cotmatrix.h>
#include <igl/vertex_components.h>
#include <igl/remove_unreferenced.h>
#include <igl/heat_geodesics.h>

#include <CLI/CLI.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCholesky>

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "imgui.h"

#include <spdlog/spdlog.h>

#include <iostream>
#include <set>
#include <limits>
#include <stdlib.h>

// Global state for polyscope visualization
static polyscope::SurfaceMesh* g_mesh = nullptr;
static int g_current_mode = 0;  // Mode for noisy function section
static int g_current_section = 0; // 0 = noisy function, 1 = step function

// Noisy function data
static Eigen::VectorXd* g_zexact = nullptr;
static Eigen::VectorXd* g_znoisy = nullptr;
static Eigen::VectorXd* g_zl = nullptr;
static Eigen::VectorXd* g_zh = nullptr;
static Eigen::VectorXd* g_zch = nullptr;

// Step function data
static Eigen::VectorXd* g_zstep = nullptr;
static Eigen::VectorXd* g_stepzl = nullptr;
static Eigen::VectorXd* g_stepzh = nullptr;
static Eigen::VectorXd* g_stepzch = nullptr;

// Update visualization based on current mode and section
inline void updateVisualization() {
    const Eigen::VectorXd* z = nullptr;
    std::string name;
    
    if (g_current_section == 0) {
        // Noisy function section
        switch (g_current_mode) {
            case 0:
                z = g_zexact;
                name = "exact";
                break;
            case 1:
                z = g_znoisy;
                name = "noisy";
                break;
            case 2:
                z = g_zl;
                name = "laplacian_smoothed";
                break;
            case 3:
                z = g_zh;
                name = "hessian_smoothed";
                break;
            case 4:
                z = g_zch;
                name = "curved_hessian_smoothed";
                break;
        }
    } else {
        // Step function section
        switch (g_current_mode) {
            case 0:
                z = g_zstep;
                name = "step";
                break;
            case 1:
                z = g_stepzl;
                name = "step_laplacian_smoothed";
                break;
            case 2:
                z = g_stepzh;
                name = "step_hessian_smoothed";
                break;
            case 3:
                z = g_stepzch;
                name = "step_curved_hessian_smoothed";
                break;
        }
    }
    
    if (z && g_mesh) {
        std::vector<double> scalarData(z->rows());
        for (int i = 0; i < z->rows(); i++) {
            scalarData[i] = (*z)(i);
        }
        auto q = g_mesh->addVertexScalarQuantity("scalar_field", scalarData);
        q->setEnabled(true);
        q->setMapRange(std::make_pair(z->minCoeff(), z->maxCoeff()));
    }
}

// ImGui callback for mode selection
inline void dataSmoothingCallback() {
    ImGui::SetNextWindowPos(ImVec2(20, 20), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(350, 350), ImGuiCond_FirstUseEver);
    
    ImGui::Begin("Data Smoothing");
    
    bool changed = false;
    
    // Section selection
    ImGui::Text("Select data section:");
    ImGui::Separator();
    
    if (ImGui::RadioButton("Noisy Function", g_current_section == 0)) {
        g_current_section = 0;
        g_current_mode = 0;
        changed = true;
    }
    if (ImGui::RadioButton("Step Function", g_current_section == 1)) {
        g_current_section = 1;
        g_current_mode = 0;
        changed = true;
    }
    
    ImGui::Separator();
    ImGui::Text("Select visualization mode:");
    ImGui::Separator();
    
    if (g_current_section == 0) {
        // Noisy function modes
        if (ImGui::RadioButton("1. Original (exact)", g_current_mode == 0)) {
            g_current_mode = 0;
            changed = true;
        }
        if (ImGui::RadioButton("2. Noisy", g_current_mode == 1)) {
            g_current_mode = 1;
            changed = true;
        }
        if (ImGui::RadioButton("3. Laplacian smoothed (zero Neumann)", g_current_mode == 2)) {
            g_current_mode = 2;
            changed = true;
        }
        if (ImGui::RadioButton("4. Hessian smoothed (natural planar)", g_current_mode == 3)) {
            g_current_mode = 3;
            changed = true;
        }
        if (ImGui::RadioButton("5. Curved Hessian smoothed (natural curved)", g_current_mode == 4)) {
            g_current_mode = 4;
            changed = true;
        }
    } else {
        // Step function modes
        if (ImGui::RadioButton("1. Step function", g_current_mode == 0)) {
            g_current_mode = 0;
            changed = true;
        }
        if (ImGui::RadioButton("2. Laplacian smoothed (zero Neumann)", g_current_mode == 1)) {
            g_current_mode = 1;
            changed = true;
        }
        if (ImGui::RadioButton("3. Hessian smoothed (natural planar)", g_current_mode == 2)) {
            g_current_mode = 2;
            changed = true;
        }
        if (ImGui::RadioButton("4. Curved Hessian smoothed (natural curved)", g_current_mode == 3)) {
            g_current_mode = 3;
            changed = true;
        }
    }
    
    if (changed) {
        updateVisualization();
    }
    
    ImGui::End();
}

struct CLIArgs {
    std::string mesh_path;
    
    CLIArgs(int argc, char* argv[]) {
        CLI::App app{"Data smoothing benchmark"};
        app.add_option("-m,--mesh", mesh_path, "Path to input mesh")->required();
        try {
            app.parse(argc, argv);
        } catch (const CLI::ParseError& e) {
            exit(app.exit(e));
        }
    }
};

int main(int argc, char * argv[])
{
  CLIArgs args(argc, argv);
  
  typedef Eigen::SparseMatrix<double> SparseMat;
  srand(57);
  
  //Read our mesh
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  if(!igl::read_triangle_mesh(args.mesh_path, V, F)) {
    spdlog::error("Failed to load mesh: {}", args.mesh_path);
    return 1;
  }
  
  //Constructing an exact function to smooth
  igl::HeatGeodesicsData<double> hgData;
  igl::heat_geodesics_precompute(V, F, hgData);
  Eigen::VectorXd heatDist;
  Eigen::VectorXi gamma(1); gamma << std::min(1947, (int)V.rows() - 1);
  igl::heat_geodesics_solve(hgData, gamma, heatDist);
  Eigen::VectorXd zexact =
  0.1*(heatDist.array() + (-heatDist.maxCoeff())).pow(2)
  + 3*V.block(0,1,V.rows(),1).array().cos();
  
  //Make the exact function noisy
  const double s = 0.1*(zexact.maxCoeff() - zexact.minCoeff());
  Eigen::VectorXd znoisy = zexact + s*Eigen::VectorXd::Random(zexact.size());
  
  //Constructing the squared Laplacian and squared Hessian energy
  SparseMat L, M;
  igl::cotmatrix(V, F, L);
  igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);
  Eigen::SimplicialLDLT<SparseMat> solver(M);
  SparseMat MinvL = solver.solve(L);
  SparseMat QL = L.transpose()*MinvL;
  SparseMat QH;
  igl::hessian_energy(V, F, QH);
  SparseMat QcH;
  igl::curved_hessian_energy(V, F, QcH);
  
  // Log matrix sizes
  spdlog::info("V rows: {}", V.rows());
  spdlog::info("F rows: {}", F.rows());
  spdlog::info("L rows: {}", L.rows());
  spdlog::info("M rows: {}", M.rows());
  spdlog::info("QL rows: {}", QL.rows());
  spdlog::info("QH rows: {}", QH.rows());
  spdlog::info("QcH rows: {}", QcH.rows());
  
  //Solve to find Laplacian-smoothed Hessian-smoothed, and
  // curved-Hessian-smoothed solutions
  const double al = 3e-7;
  Eigen::SimplicialLDLT<SparseMat> lapSolver(al*QL + (1.-al)*M);
  Eigen::VectorXd zl = lapSolver.solve(al*M*znoisy);
  const double ah = 2e-7;
  Eigen::SimplicialLDLT<SparseMat> hessSolver(ah*QH + (1.-ah)*M);
  Eigen::VectorXd zh = hessSolver.solve(ah*M*znoisy);
  const double ach = 3e-7;
  Eigen::SimplicialLDLT<SparseMat> curvedHessSolver(al*QcH + (1.-ach)*M);
  Eigen::VectorXd zch = curvedHessSolver.solve(ach*M*znoisy);
  
  //Constructing a step function to smooth
  Eigen::VectorXd zstep = Eigen::VectorXd::Zero(V.rows());
  for(int i=0; i<V.rows(); ++i) {
    zstep(i) = V(i,2)<-0.25 ? 1. : (V(i,2)>0.31 ? 2. : 0);
  }
  
  //Smooth that function
  const double sl = 2e-5;
  Eigen::SimplicialLDLT<SparseMat> stepLapSolver(sl*QL + (1.-sl)*M);
  Eigen::VectorXd stepzl = stepLapSolver.solve(al*M*zstep);
  const double sh = 6e-6;
  Eigen::SimplicialLDLT<SparseMat> stepHessSolver(sh*QH + (1.-sh)*M);
  Eigen::VectorXd stepzh = stepHessSolver.solve(ah*M*zstep);
  const double sch = 2e-5;
  Eigen::SimplicialLDLT<SparseMat> stepCurvedHessSolver(sl*QcH + (1.-sch)*M);
  Eigen::VectorXd stepzch = stepCurvedHessSolver.solve(ach*M*zstep);
  
  // Set up global pointers for polyscope callback
  g_zexact = &zexact;
  g_znoisy = &znoisy;
  g_zl = &zl;
  g_zh = &zh;
  g_zch = &zch;
  g_zstep = &zstep;
  g_stepzl = &stepzl;
  g_stepzh = &stepzh;
  g_stepzch = &stepzch;
  
  // Log value ranges for debugging
  spdlog::info("Exact value range: [{}, {}]", zexact.minCoeff(), zexact.maxCoeff());
  spdlog::info("Noisy value range: [{}, {}]", znoisy.minCoeff(), znoisy.maxCoeff());
  spdlog::info("Laplacian smoothed range: [{}, {}]", zl.minCoeff(), zl.maxCoeff());
  spdlog::info("Hessian smoothed range: [{}, {}]", zh.minCoeff(), zh.maxCoeff());
  spdlog::info("Curved Hessian smoothed range: [{}, {}]", zch.minCoeff(), zch.maxCoeff());
  
  // Initialize polyscope
  polyscope::init();
  
  // Register the surface mesh
  std::vector<std::array<double, 3>> vertices(V.rows());
  for (int i = 0; i < V.rows(); i++) {
      vertices[i] = {V(i, 0), V(i, 1), V(i, 2)};
  }
  std::vector<std::array<int, 3>> faces(F.rows());
  for (int i = 0; i < F.rows(); i++) {
      faces[i] = {F(i, 0), F(i, 1), F(i, 2)};
  }
  g_mesh = polyscope::registerSurfaceMesh("mesh", vertices, faces);
  g_mesh->setEdgeWidth(0.0);  // Hide mesh edges
  
  // Set the user callback for mode selection
  polyscope::state::userCallback = dataSmoothingCallback;
  
  // Initial visualization (show noisy data)
  g_current_section = 0;
  g_current_mode = 1;
  updateVisualization();
  
  std::cout << "Use the GUI to switch between different smoothing modes.\n";
  std::cout << "Two sections available: Noisy Function and Step Function.\n";
  
  // Show the polyscope GUI
  polyscope::show();
  
  // Cleanup
  polyscope::state::userCallback = nullptr;
  g_zexact = nullptr;
  g_znoisy = nullptr;
  g_zl = nullptr;
  g_zh = nullptr;
  g_zch = nullptr;
  g_zstep = nullptr;
  g_stepzl = nullptr;
  g_stepzh = nullptr;
  g_stepzch = nullptr;
  g_mesh = nullptr;

  return 0;
}
