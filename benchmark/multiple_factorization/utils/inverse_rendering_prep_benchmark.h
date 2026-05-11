#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <spdlog/spdlog.h>


namespace homa {

    struct InverseRenderingEntry {
        int idx           = 0;
        int n_vertices    = 0;
        int fwd_count     = 0;
        int bwd_count     = 0;
        std::string matrix_file;
        std::string fwd_rhs_file;
        std::string bwd_rhs_file;
        std::string mesh_file;
    };

    /**
     * @brief Parse <folder>/counts.csv and produce per-row entries with absolute paths.
     *
     * Expected counts.csv header:
     *     idx,n_vertices,fwd_count,bwd_count,matrix_file,fwd_rhs_file,bwd_rhs_file,mesh_file
     *
     * The four *_file columns are stored as relative filenames in the csv; this
     * helper prefixes them with <folder>/ so the caller gets ready-to-open
     * absolute paths. Trailing CR/LF/whitespace is stripped on every cell to
     * tolerate CRLF-terminated CSVs.
     *
     * @param folder_address Path to the dataset folder (must contain counts.csv).
     * @param entries Output list of parsed entries (cleared on entry).
     */
    inline void prepare_inverse_rendering_benchmark_data(
        const std::string& folder_address,
        std::vector<InverseRenderingEntry>& entries)
    {
        entries.clear();

        std::filesystem::path counts_path =
            std::filesystem::path(folder_address) / "counts.csv";
        std::ifstream in(counts_path.string());
        if (!in.is_open()) {
            spdlog::warn("counts.csv not found under: {}", folder_address);
            return;
        }

        auto strip_ws = [](std::string& s) {
            while (!s.empty() && (s.back() == '\r' || s.back() == '\n' ||
                                  s.back() == ' '  || s.back() == '\t')) {
                s.pop_back();
            }
            size_t start = 0;
            while (start < s.size() && (s[start] == ' ' || s[start] == '\t')) {
                ++start;
            }
            if (start > 0) s.erase(0, start);
        };

        std::string line;
        bool header_read = false;
        while (std::getline(in, line)) {
            strip_ws(line);
            if (line.empty()) continue;
            if (!header_read) {
                header_read = true;
                continue;
            }

            std::vector<std::string> cols;
            cols.reserve(8);
            std::stringstream ss(line);
            std::string cell;
            while (std::getline(ss, cell, ',')) {
                strip_ws(cell);
                cols.push_back(cell);
            }
            if (cols.size() < 8) {
                spdlog::warn("Skipping malformed row (expected 8 cols): {}", line);
                continue;
            }

            InverseRenderingEntry e;
            try {
                e.idx        = std::stoi(cols[0]);
                e.n_vertices = std::stoi(cols[1]);
                e.fwd_count  = std::stoi(cols[2]);
                e.bwd_count  = std::stoi(cols[3]);
            } catch (const std::exception& ex) {
                spdlog::warn("Skipping row with unparseable integers ({}): {}",
                             ex.what(), line);
                continue;
            }
            e.matrix_file   = (std::filesystem::path(folder_address) / cols[4]).string();
            e.fwd_rhs_file  = (std::filesystem::path(folder_address) / cols[5]).string();
            e.bwd_rhs_file  = (std::filesystem::path(folder_address) / cols[6]).string();
            e.mesh_file     = (std::filesystem::path(folder_address) / cols[7]).string();
            entries.push_back(std::move(e));
        }

        if (entries.empty()) {
            spdlog::warn("No entries parsed from: {}", counts_path.string());
        }
    }

}  // namespace homa
