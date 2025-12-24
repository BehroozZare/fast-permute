//
// Created by behrooz on 2025-12-23.
//

#ifndef GPU_ORDERING_SHOW_SEPARATOR_PER_LEVEL_H
#define GPU_ORDERING_SHOW_SEPARATOR_PER_LEVEL_H

#include <vector>
#include <array>
#include <cmath>
#include <string>
#include <iostream>
#include <Eigen/Core>
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "imgui.h"

namespace RXMESH_SOLVER {

    // ============== Static state for the callback ==============
    static std::vector<bool> g_level_enabled;                      // Checkbox states for each level
    static std::vector<std::vector<int>> g_separators;             // Separator vertices per level
    static std::vector<std::array<double, 3>> g_level_colors;      // Precomputed colors per level
    static int g_num_vertices = 0;                                 // Total number of vertices
    static polyscope::SurfaceMesh* g_separator_mesh = nullptr;     // Pointer to the mesh

    // Helper function to convert HSV to RGB
    inline std::array<double, 3> hsv_to_rgb(double h, double s, double v) {
        double c = v * s;
        double x = c * (1 - std::abs(std::fmod(h / 60.0, 2) - 1));
        double m = v - c;
        
        double r, g, b;
        if (h < 60) {
            r = c; g = x; b = 0;
        } else if (h < 120) {
            r = x; g = c; b = 0;
        } else if (h < 180) {
            r = 0; g = c; b = x;
        } else if (h < 240) {
            r = 0; g = x; b = c;
        } else if (h < 300) {
            r = x; g = 0; b = c;
        } else {
            r = c; g = 0; b = x;
        }
        
        return {r + m, g + m, b + m};
    }

    // ============== Update vertex colors based on enabled levels ==============
    inline void update_separator_vertex_colors() {
        if (g_separator_mesh == nullptr || g_num_vertices == 0) return;

        // Start with white for all vertices (non-separator vertices)
        std::vector<std::array<double, 3>> vertex_colors(g_num_vertices, {1.0, 1.0, 1.0});
        
        // Gray color for disabled levels
        const std::array<double, 3> gray_color = {0.5, 0.5, 0.5};

        // Apply colors for all levels - enabled get their color, disabled get gray
        int total_levels = static_cast<int>(g_level_enabled.size());
        for (int level = 0; level < total_levels; level++) {
            for (int vertex_idx : g_separators[level]) {
                if (g_level_enabled[level]) {
                    // Assign this level's color to its vertices
                    vertex_colors[vertex_idx] = g_level_colors[level];
                } else {
                    // Assign gray to disabled level vertices
                    vertex_colors[vertex_idx] = gray_color;
                }
            }
        }

        // Update the mesh color quantity
        g_separator_mesh->addVertexColorQuantity("separator_levels", vertex_colors)->setEnabled(true);
    }

    // ============== ImGui callback function for level selector ==============
    inline void level_selector_callback() {
        // Set initial window position and size (only on first appearance)
        ImGui::SetNextWindowPos(ImVec2(20, 20), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(320, 450), ImGuiCond_FirstUseEver);
        
        // Use a collapsing header style window that's always visible
        ImGui::Begin("Separator Level Controls", nullptr, 
                     ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize);
        
        int total_levels = static_cast<int>(g_level_enabled.size());
        
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.4f, 1.0f), "Per-Level Separator Visibility");
        ImGui::Text("(%d levels found)", total_levels);
        ImGui::Separator();
        
        if (total_levels == 0) {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "No levels found!");
            ImGui::End();
            return;
        }

        bool changed = false;

        // Add "Select All" and "Deselect All" buttons
        if (ImGui::Button("Select All")) {
            for (int level = 0; level < total_levels; level++) {
                g_level_enabled[level] = true;
            }
            changed = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Deselect All")) {
            for (int level = 0; level < total_levels; level++) {
                g_level_enabled[level] = false;
            }
            changed = true;
        }

        ImGui::Separator();

        // Create a checkbox for each level with color indicator
        for (int level = 0; level < total_levels; level++) {
            ImGui::PushID(level);  // Unique ID for each level's widgets
            
            // Show colored square - show actual color when enabled, gray when disabled
            bool is_enabled = g_level_enabled[level];
            ImVec4 display_color;
            if (is_enabled) {
                display_color = ImVec4(
                    static_cast<float>(g_level_colors[level][0]),
                    static_cast<float>(g_level_colors[level][1]),
                    static_cast<float>(g_level_colors[level][2]),
                    1.0f
                );
            } else {
                display_color = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);  // Gray when disabled
            }
            
            ImGui::ColorButton("##color", display_color, 
                               ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoDragDrop, 
                               ImVec2(20, 20));
            ImGui::SameLine();

            // Create the checkbox (use local bool to avoid vector<bool> proxy issue)
            std::string label = "Level " + std::to_string(level) + " (" + 
                               std::to_string(g_separators[level].size()) + " verts)";
            if (ImGui::Checkbox(label.c_str(), &is_enabled)) {
                g_level_enabled[level] = is_enabled;
                changed = true;
            }
            
            ImGui::PopID();
        }

        // Update colors if any checkbox changed
        if (changed) {
            update_separator_vertex_colors();
        }

        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Tip: Unchecked levels appear gray");

        ImGui::End();
    }

    void show_separator_per_level(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
                                   std::vector<int>& new_labels, std::vector<int>& sep_ptr) {
        // Calculate total levels - ensure at least 1 level if we have data
        int total_levels = (sep_ptr.size() > 1) ? static_cast<int>(std::log2(sep_ptr.size())) : 0;
        if (total_levels < 1 && sep_ptr.size() > 1) total_levels = 1;
        
        g_num_vertices = V.rows();
        
        // Debug output
        std::cout << "=======================================" << std::endl;
        std::cout << "[SEPARATOR LEVELS DEBUG]" << std::endl;
        std::cout << "  sep_ptr.size() = " << sep_ptr.size() << std::endl;
        std::cout << "  total_levels   = " << total_levels << std::endl;
        std::cout << "  g_num_vertices = " << g_num_vertices << std::endl;

        // Initialize separator data per level
        g_separators.clear();
        g_separators.resize(total_levels);
        for (int level = 0; level < total_levels; level++) {
            int start_idx = (1 << level) - 1;
            int end_idx = std::min((1 << (level + 1)) - 1, static_cast<int>(sep_ptr.size()) - 1);
            for (int i = start_idx; i < end_idx; i++) {
                if (i + 1 < static_cast<int>(sep_ptr.size())) {
                    for (int j = sep_ptr[i]; j < sep_ptr[i + 1]; j++) {
                        g_separators[level].push_back(new_labels[j]);
                    }
                }
            }
            std::cout << "  Level " << level << ": " << g_separators[level].size() << " vertices" << std::endl;
        }
        std::cout << "=======================================" << std::endl;

        // Initialize checkbox states (all enabled by default)
        g_level_enabled.clear();
        g_level_enabled.resize(total_levels, true);

        // Precompute colors for each level
        g_level_colors.clear();
        g_level_colors.resize(total_levels);
        for (int level = 0; level < total_levels; level++) {
            if (level == total_levels - 1) {
                // Final level: gray
                g_level_colors[level] = {0.5, 0.5, 0.5};
            } else {
                // Other levels: use HSV-based hue rotation for distinct colors
                double hue = (level * 360.0) / (total_levels - 1);
                g_level_colors[level] = hsv_to_rgb(hue, 1.0, 1.0);
            }
        }

        // Initialize polyscope
        polyscope::init();

        // Convert Eigen matrices to std::vector format for polyscope
        std::vector<std::array<double, 3>> vertices(V.rows());
        for (int i = 0; i < V.rows(); i++) {
            vertices[i] = {V(i, 0), V(i, 1), V(i, 2)};
        }

        std::vector<std::array<int, 3>> faces(F.rows());
        for (int i = 0; i < F.rows(); i++) {
            faces[i] = {F(i, 0), F(i, 1), F(i, 2)};
        }

        // Register the surface mesh
        g_separator_mesh = polyscope::registerSurfaceMesh("separator_mesh", vertices, faces);

        // Initial color update (all levels enabled)
        update_separator_vertex_colors();

        // Set the user callback for ImGui controls
        polyscope::state::userCallback = level_selector_callback;

        // Show the polyscope GUI
        polyscope::show();

        // Cleanup: reset the callback after closing
        polyscope::state::userCallback = nullptr;
    }

}
#endif //GPU_ORDERING_SHOW_SEPARATOR_PER_LEVEL_H
