//
// Remesher Tool - Control triangle mesh resolution using upsample/decimate
//
// Usage:
//   ./gpu_ordering_remesher -i input.obj -o output.obj -m upsample -n 2
//   ./gpu_ordering_remesher -i input.obj -o output.obj -m decimate -t 1000
//

#include <igl/read_triangle_mesh.h>
#include <igl/writeOBJ.h>
#include <igl/writeMSH.h>
#include <igl/writeOFF.h>
#include <igl/writePLY.h>
#include <igl/writeSTL.h>
#include <igl/upsample.h>
#include <igl/decimate.h>
#include <spdlog/spdlog.h>
#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include <iostream>
#include <filesystem>

// Write mesh to file, auto-detecting format from extension
bool write_mesh(const std::string& filename, 
                const Eigen::MatrixXd& V, 
                const Eigen::MatrixXi& F)
{
    std::filesystem::path path(filename);
    std::string ext = path.extension().string();
    
    // Convert to lowercase for comparison
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    if (ext == ".obj") {
        return igl::writeOBJ(filename, V, F);
    } else if (ext == ".msh") {
        // For surface meshes, pass empty tetrahedra matrix and default tags/fields
        Eigen::MatrixXi Tet;           // Empty tetrahedra (surface mesh only)
        Eigen::VectorXi TriTag = Eigen::VectorXi::Zero(F.rows());  // Surface face tags
        Eigen::VectorXi TetTag;        // Empty tet tags
        std::vector<std::string> XFields, EFields;  // No node/element fields
        std::vector<Eigen::MatrixXd> XF, TriF, TetF;  // Empty field data
        return igl::writeMSH(filename, V, F, Tet, TriTag, TetTag, XFields, XF, EFields, TriF, TetF);
    } else if (ext == ".off") {
        return igl::writeOFF(filename, V, F);
    } else if (ext == ".ply") {
        return igl::writePLY(filename, V, F);
    } else if (ext == ".stl") {
        return igl::writeSTL(filename, V, F);
    } else {
        spdlog::error("Unsupported output format: {}", ext);
        spdlog::info("Supported formats: .obj, .msh, .off, .ply, .stl");
        return false;
    }
}

struct CLIArgs
{
    std::string input_mesh;
    std::string output_mesh;
    std::string mode = "upsample";  // "upsample" or "decimate"
    int upsample_iterations = 1;    // Number of subdivision iterations
    int target_faces = -1;          // Target face count for decimation (-1 = half of input)
    bool block_intersections = false; // For decimate: block self-intersections

    CLIArgs(int argc, char* argv[])
    {
        CLI::App app{"Remesher - Control triangle mesh resolution using upsample/decimate"};
        
        app.add_option("-i,--input", input_mesh, "Input mesh file (OBJ format)")
            ->required()
            ->check(CLI::ExistingFile);
        
        app.add_option("-o,--output", output_mesh, 
            "Output mesh file. Supported formats: .obj, .msh, .off, .ply, .stl (defaults to <input>_<mode>.obj)");
        
        app.add_option("-m,--mode", mode, "Operation mode: 'upsample' or 'decimate'")
            ->check(CLI::IsMember({"upsample", "decimate"}));
        
        app.add_option("-n,--iterations", upsample_iterations, 
            "Number of subdivision iterations for upsample (each iteration quadruples faces)")
            ->check(CLI::PositiveNumber);
        
        app.add_option("-t,--target-faces", target_faces, 
            "Target number of faces for decimate (-1 = half of input faces)")
            ->check(CLI::Range(-1, INT_MAX));
        
        app.add_flag("-b,--block-intersections", block_intersections,
            "Block self-intersections during decimation (slower but safer)");

        try {
            app.parse(argc, argv);
        } catch (const CLI::ParseError& e) {
            exit(app.exit(e));
        }

        // Generate default output filename if not specified
        if (output_mesh.empty()) {
            std::filesystem::path input_path(input_mesh);
            std::string stem = input_path.stem().string();
            std::string extension = input_path.extension().string();
            std::filesystem::path parent = input_path.parent_path();
            
            if (extension.empty()) {
                extension = ".obj";
            }
            
            output_mesh = (parent / (stem + "_" + mode + extension)).string();
        }
    }
};


int main(int argc, char* argv[])
{
    CLIArgs args(argc, argv);

    // Read input mesh
    spdlog::info("Reading mesh from: {}", args.input_mesh);
    
    Eigen::MatrixXd V;  // Vertices
    Eigen::MatrixXi F;  // Faces
    
    if (!igl::read_triangle_mesh(args.input_mesh, V, F)) {
        spdlog::error("Failed to read mesh: {}", args.input_mesh);
        return 1;
    }
    
    spdlog::info("Input mesh: {} vertices, {} faces", V.rows(), F.rows());

    // Output matrices
    Eigen::MatrixXd NV;  // New vertices
    Eigen::MatrixXi NF;  // New faces

    if (args.mode == "upsample") {
        // Upsample (subdivision)
        spdlog::info("Upsampling mesh with {} iteration(s)...", args.upsample_iterations);
        spdlog::info("Expected output: ~{} faces (each iteration quadruples face count)", 
            F.rows() * static_cast<int>(std::pow(4, args.upsample_iterations)));
        
        igl::upsample(V, F, NV, NF, args.upsample_iterations);
        
        spdlog::info("Upsampling complete.");
        
    } else if (args.mode == "decimate") {
        // Decimate (simplification)
        int target = args.target_faces;
        if (target < 0) {
            target = F.rows() / 2;  // Default: half of input faces
        }
        
        if (target >= F.rows()) {
            spdlog::warn("Target faces ({}) >= input faces ({}), no decimation needed.", 
                target, F.rows());
            NV = V;
            NF = F;
        } else {
            spdlog::info("Decimating mesh to {} faces...", target);
            
            Eigen::VectorXi J;  // Map from output faces to input faces
            Eigen::VectorXi I;  // Map from output vertices to input vertices
            
            bool success = igl::decimate(V, F, target, args.block_intersections, NV, NF, J, I);
            
            if (!success) {
                spdlog::warn("Could not reach target face count. Result has {} faces.", NF.rows());
            } else {
                spdlog::info("Decimation complete.");
            }
        }
    }

    spdlog::info("Output mesh: {} vertices, {} faces", NV.rows(), NF.rows());
    
    // Compute and display statistics
    double vertex_ratio = static_cast<double>(NV.rows()) / V.rows();
    double face_ratio = static_cast<double>(NF.rows()) / F.rows();
    spdlog::info("Vertex ratio: {:.2f}x ({} -> {})", vertex_ratio, V.rows(), NV.rows());
    spdlog::info("Face ratio: {:.2f}x ({} -> {})", face_ratio, F.rows(), NF.rows());

    // Write output mesh (format auto-detected from extension)
    spdlog::info("Writing mesh to: {}", args.output_mesh);
    
    if (!write_mesh(args.output_mesh, NV, NF)) {
        spdlog::error("Failed to write mesh: {}", args.output_mesh);
        return 1;
    }
    
    spdlog::info("Done!");
    return 0;
}
