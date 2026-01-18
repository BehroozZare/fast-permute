#include <igl/read_triangle_mesh.h>
#include <igl/cotmatrix.h>
#include <igl/orient_halfedges.h>
#include <igl/cr_vector_laplacian.h>
#include <igl/cr_vector_mass.h>
#include <igl/edge_midpoints.h>
#include <igl/edge_vectors.h>
#include <igl/average_from_edges_onto_vertices.h>
#include <igl/PI.h>

#include <CLI/CLI.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCholesky>
#include <Eigen/Geometry>

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "imgui.h"

#include <iostream>
#include <set>
#include <limits>
#include <stdlib.h>
#include <algorithm>
#include <spdlog/spdlog.h>

// Global state for polyscope visualization
static polyscope::SurfaceMesh* g_mesh = nullptr;
static int g_current_mode = 0;  // 0=raw, 1=noisy, 2=smoothed

// Magnitude data (vertex scalars)
static Eigen::MatrixXd* g_rawcolors = nullptr;
static Eigen::MatrixXd* g_noisycolors = nullptr;
static Eigen::MatrixXd* g_smoothedcolors = nullptr;

// Update visualization based on current mode
inline void updateVisualization() {
    const Eigen::MatrixXd* colors = nullptr;
    
    switch (g_current_mode) {
        case 0:
            colors = g_rawcolors;
            break;
        case 1:
            colors = g_noisycolors;
            break;
        case 2:
            colors = g_smoothedcolors;
            break;
    }
    
    if (colors && g_mesh) {
        // Update vertex colors on mesh (use first column as scalar)
        std::vector<double> scalarColors(colors->rows());
        for (int i = 0; i < colors->rows(); i++) {
            scalarColors[i] = (*colors)(i, 0);
        }
        auto q = g_mesh->addVertexScalarQuantity("magnitude", scalarColors);
        q->setEnabled(true);
        // Use per-mode range so each field shows its own variation
        q->setMapRange(std::make_pair(colors->minCoeff(), colors->maxCoeff()));
    }
}

// ImGui callback for mode selection
inline void vectorFieldCallback() {
    ImGui::SetNextWindowPos(ImVec2(20, 20), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(250, 150), ImGuiCond_FirstUseEver);
    
    ImGui::Begin("Vector Field Smoothing");
    
    ImGui::Text("Select visualization mode:");
    ImGui::Separator();
    
    bool changed = false;
    if (ImGui::RadioButton("Raw (1)", g_current_mode == 0)) {
        g_current_mode = 0;
        changed = true;
    }
    if (ImGui::RadioButton("Noisy (2)", g_current_mode == 1)) {
        g_current_mode = 1;
        changed = true;
    }
    if (ImGui::RadioButton("Smoothed (3)", g_current_mode == 2)) {
        g_current_mode = 2;
        changed = true;
    }
    
    if (changed) {
        updateVisualization();
    }
    
    ImGui::End();
}

struct CLIArgs {
    std::string mesh_path;
    double smooth_factor = 1e-1;
    int iterations = 50;

    CLIArgs(int argc, char* argv[]) {
        CLI::App app{"Vector field smoothing"};
        app.add_option("-m,--mesh", mesh_path, "Path to input mesh")->required();
        app.add_option("-s,--smooth", smooth_factor, "Smoothing factor");
        app.add_option("-i,--iterations", iterations, "Number of smoothing iterations");
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

  //Constants used for smoothing (from CLI arguments)
  const double howMuchToSmoothBy = args.smooth_factor;
  const int howManySmoothingInterations = args.iterations;

  //Read our mesh
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  if(!igl::read_triangle_mesh(args.mesh_path, V, F)) {
    std::cout << "Failed to load mesh: " << args.mesh_path << std::endl;
    return 1;
  }

  //Orient edges for plotting
  Eigen::MatrixXi E, oE;
  igl::orient_halfedges(F, E, oE);

  //Compute edge midpoints & edge vectors
  Eigen::MatrixXd edgeMps, parVec, perpVec;
  igl::edge_midpoints(V, F, E, oE, edgeMps);
  igl::edge_vectors(V, F, E, oE, parVec, perpVec);

  //Constructing a function to add noise to
  const auto zraw_function = [] (const Eigen::Vector3d& x) {
    return Eigen::Vector3d(0.2*x(1) + cos(2*x(1)+0.2),
                           0.5*x(0) + 0.15,
                           0.3*cos(0.2+igl::PI*x(2)));
  };

  Eigen::VectorXd zraw(2*edgeMps.rows());
  for(int i=0; i<edgeMps.rows(); ++i) {
    const Eigen::Vector3d f = zraw_function(edgeMps.row(i));
    zraw(i) = f.dot(parVec.row(i));
    zraw(i+edgeMps.rows()) = f.dot(perpVec.row(i));
  }

  //Add noise
  srand(71);
  const double l = 15;
  Eigen::VectorXd znoisy = zraw + l*Eigen::VectorXd::Random(zraw.size());

  //Denoise function using the vector Dirichlet energy
  Eigen::VectorXd zsmoothed = znoisy;
    //Compute Laplacian and mass matrix
    SparseMat L, M, cot;
    igl::cotmatrix(V,F,cot);
    igl::cr_vector_mass(V, F, E, M);
    igl::cr_vector_laplacian(V, F, E, oE, L);
    Eigen::SimplicialLDLT<SparseMat> rhsSolver(M + howMuchToSmoothBy*L);
    spdlog::info("The size of the F matrix is: {}", F.rows());
    spdlog::info("The size of the V matrix is: {}", V.rows());
    spdlog::info("The size of the M matrix is: {}", M.rows());
    spdlog::info("The size of the L matrix is: {}", L.rows());
    spdlog::info("The size of the nonzeros of cot is: {}", cot.nonZeros());
    spdlog::info("The size of the rhsSolver matrix is: {}", rhsSolver.rows());
    //Implicit step
  for(int i=0; i<howManySmoothingInterations; ++i) {

    zsmoothed = rhsSolver.solve(M*zsmoothed);
  }

  //Convert vector fields for plotting
  const auto cr_result_to_vecs_and_colors = [&]
  (const Eigen::VectorXd& z, Eigen::MatrixXd& vecs, Eigen::MatrixXd& colors) {
    vecs.resize(edgeMps.rows(), 3);
    for(int i=0; i<edgeMps.rows(); ++i) {
      vecs.row(i) = z(i)*parVec.row(i)
      + z(i+edgeMps.rows())*perpVec.row(i);
    }
    igl::average_from_edges_onto_vertices
    (F, E, oE, vecs.rowwise().norm().eval(), colors);
  };
  Eigen::MatrixXd noisyvecs, noisycolors, smoothedvecs, smoothedcolors,
  rawvecs, rawcolors;
  cr_result_to_vecs_and_colors(znoisy, noisyvecs, noisycolors);
  cr_result_to_vecs_and_colors(zsmoothed, smoothedvecs, smoothedcolors);
  cr_result_to_vecs_and_colors(zraw, rawvecs, rawcolors);


  // Set up global pointers for polyscope callback
  g_rawcolors = &rawcolors;
  g_noisycolors = &noisycolors;
  g_smoothedcolors = &smoothedcolors;
  
  // Log magnitude ranges for debugging
  spdlog::info("Raw magnitude range: [{}, {}]", rawcolors.minCoeff(), rawcolors.maxCoeff());
  spdlog::info("Noisy magnitude range: [{}, {}]", noisycolors.minCoeff(), noisycolors.maxCoeff());
  spdlog::info("Smoothed magnitude range: [{}, {}]", smoothedcolors.minCoeff(), smoothedcolors.maxCoeff());
  
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
  polyscope::state::userCallback = vectorFieldCallback;
  
  // Initial visualization (show raw field)
  g_current_mode = 0;
  updateVisualization();
  
  std::cout << "Use the GUI to switch between Raw, Noisy, and Smoothed magnitude fields.\n";
  
  // Show the polyscope GUI
  polyscope::show();
  
  // Cleanup
  polyscope::state::userCallback = nullptr;
  g_rawcolors = nullptr;
  g_noisycolors = nullptr;
  g_smoothedcolors = nullptr;
  g_mesh = nullptr;

  return 0;
}
