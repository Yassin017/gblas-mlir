// #include <iostream>
// #include <vector>
// #include <chrono>

// // Universal MemRef descriptor for passing arrays into MLIR
// template <typename T, size_t N>
// struct MemRef {
//     T* allocated;
//     T* aligned;
//     intptr_t offset;
//     intptr_t sizes[N];
//     intptr_t strides[N];
// };

// extern "C" {
//     // We pass scalars for sizes, and MemRefs for the array pointers
//     void _mlir_ciface_bfs(
//         int32_t num_nodes, 
//         int32_t num_edges, 
//         MemRef<int64_t, 2>* coords, 
//         MemRef<float, 1>* vals, 
//         MemRef<float, 1>* out_dist
//     );
// }

// int main() {
//     // 1. Define dynamic graph bounds
//     int32_t num_nodes = 3;
//     int32_t num_edges = 2;

//     // 2. Prepare dynamic arrays (These can later be populated from a file reader)
//     std::vector<int64_t> coords_data = {
//         0, 1, // Edge from Node 0 to Node 1
//         1, 2  // Edge from Node 1 to Node 2
//     };
//     std::vector<float> vals_data = {1.0f, 1.0f};
    
//     // Allocate distance array and fill with -1.0
//     std::vector<float> dist_data(num_nodes, -1.0f);

//     // 3. Create MLIR descriptors
//     MemRef<int64_t, 2> coords_memref = {
//         coords_data.data(), coords_data.data(), 0,
//         {num_edges, 2}, {2, 1} // size: [E, 2], stride: [2, 1]
//     };
    
//     MemRef<float, 1> vals_memref = {
//         vals_data.data(), vals_data.data(), 0,
//         {num_edges}, {1}
//     };
    
//     MemRef<float, 1> dist_memref = {
//         dist_data.data(), dist_data.data(), 0,
//         {num_nodes}, {1}
//     };

//     std::cout << "--- Starting GBLAS Benchmark ---" << std::endl;
//     std::cout << "Nodes: " << num_nodes << " | Edges: " << num_edges << std::endl;

//     // 4. Start the benchmark timer
//     auto start = std::chrono::high_resolution_clock::now();
    
//     // 5. Call the compiled MLIR graph kernel
//     _mlir_ciface_bfs(num_nodes, num_edges, &coords_memref, &vals_memref, &dist_memref);
    
//     // 6. Stop the benchmark timer
//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double, std::milli> duration = end - start;
    
//     std::cout << "Kernel Execution Time: " << duration.count() << " ms" << std::endl;
    
//     // 7. Print the results only AFTER the timer has stopped
//     std::cout << "Computed Distances: ";
//     for(int i = 0; i < num_nodes; ++i) {
//         std::cout << dist_data[i] << " ";
//     }
//     std::cout << std::endl;
    
//     return 0;
// }

//////////

// #include <iostream>
// #include <fstream>
// #include <sstream>
// #include <vector>
// #include <string>
// #include <chrono>
// #include <algorithm>


// static std::chrono::time_point<std::chrono::high_resolution_clock> bfs_start_time;

// extern "C" {
//     // MLIR will call this right before the math starts
//     void start_timer() {
//         bfs_start_time = std::chrono::high_resolution_clock::now();
//     }

//     // MLIR will call this right after the math finishes
//     void stop_timer() {
//         auto end_time = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double, std::milli> duration = end_time - bfs_start_time;
//         std::cout << "\n>>> BFS Kernel Time: " << duration.count() << " ms <<<\n" << std::endl;
//     }
// }


// struct CooEntry {
//     int64_t row, col;
//     float val;
//     bool operator<(const CooEntry& other) const {
//         if (row != other.row) return row < other.row;
//         return col < other.col;
//     }
// };

// bool readMatrixMarket(const std::string& filename, int32_t& num_nodes, int32_t& num_edges,
//                       std::vector<int64_t>& coords, std::vector<float>& vals) {
//     std::ifstream file(filename);
//     if (!file.is_open()) return false;

//     std::string line;
//     bool dimensions_read = false;
//     bool is_symmetric = false;
//     std::vector<CooEntry> entries;

//     while (std::getline(file, line)) {
//         // Check for symmetry in the header
//         if (line.find("%%MatrixMarket") != std::string::npos && line.find("symmetric") != std::string::npos) {
//             is_symmetric = true;
//         }

//         if (line.empty() || line[0] == '%') continue;

//         std::istringstream iss(line);

//         if (!dimensions_read) {
//             int32_t rows, cols, raw_edges;
//             iss >> rows >> cols >> raw_edges;
//             num_nodes = std::max(rows, cols); 
//             // If symmetric, we might end up with nearly 2x the edges
//             entries.reserve(is_symmetric ? raw_edges * 2 : raw_edges);
//             dimensions_read = true;
//         } else {
//             int64_t u, v;
//             float weight = 1.0f;
//             iss >> u >> v;
//             if (!(iss >> weight)) weight = 1.0f;

//             // Convert 1-based to 0-based indexing
//             entries.push_back({u - 1, v - 1, weight});
            
//             // THE SYMMETRIC FIX: Mirror the edge if needed
//             if (is_symmetric && u != v) {
//                 entries.push_back({v - 1, u - 1, weight});
//             }
//         }
//     }

//     std::cout << "Symmetric graph detected: " << (is_symmetric ? "Yes" : "No") << std::endl;
//     std::cout << "Sorting " << entries.size() << " directed edges..." << std::endl;
//     std::sort(entries.begin(), entries.end());

//     // Update actual edge count after mirroring
//     num_edges = entries.size();
    
//     coords.reserve(num_edges * 2);
//     vals.reserve(num_edges);
//     for (const auto& entry : entries) {
//         coords.push_back(entry.row);
//         coords.push_back(entry.col);
//         vals.push_back(entry.val);
//     }
//     return true;
// }



// template <typename T, size_t N>
// struct MemRef {
//     T* allocated;
//     T* aligned;
//     intptr_t offset;
//     intptr_t sizes[N];
//     intptr_t strides[N];
// };

// extern "C" {
//     void _mlir_ciface_bfs(
//         int32_t num_nodes, 
//         int32_t num_edges, 
//         MemRef<int64_t, 2>* coords, 
//         MemRef<float, 1>* vals, 
//         MemRef<float, 1>* out_dist
//     );
// }



// int main() {
//     int32_t num_nodes = 0;
//     int32_t num_edges = 0;
//     std::vector<int64_t> coords_data;
//     std::vector<float> vals_data;

//     // Assuming you downloaded a graph named "benchmark_graph.mtx"
//     std::string filename = "1138_bus.mtx";
//     std::cout << "Loading graph from " << filename << "..." << std::endl;

//     if (!readMatrixMarket(filename, num_nodes, num_edges, coords_data, vals_data)) {
//         return 1; // Exit if file reading fails
//     }

//     std::cout << "Graph loaded" << std::endl;
//     std::cout << "Nodes: " << num_nodes << ", Edges: " << num_edges << std::endl;

//     std::vector<float> dist_data(num_nodes, -1.0f);

//     // Setup MemRefs
//     MemRef<int64_t, 2> coords_memref = {
//         coords_data.data(), coords_data.data(), 0,
//         {num_edges, 2}, {2, 1}
//     };
//     MemRef<float, 1> vals_memref = {
//         vals_data.data(), vals_data.data(), 0,
//         {num_edges}, {1}
//     };
//     MemRef<float, 1> dist_memref = {
//         dist_data.data(), dist_data.data(), 0,
//         {num_nodes}, {1}
//     };

//     std::cout << "Starting GBLAS BFS"  << std::endl;
//     auto start = std::chrono::high_resolution_clock::now();
    
//     // Call the MLIR code
//     _mlir_ciface_bfs(num_nodes, num_edges, &coords_memref, &vals_memref, &dist_memref);
    
//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double, std::milli> duration = end - start;
    
//     std::cout << "BFS Full Time: " << duration.count() << " ms" << std::endl;
    
//     // Only print distances if the graph is small enough, otherwise just print a sample
//     if (num_nodes <= 100) {
//         std::cout << "Computed Distances: ";
//         for(int i = 0; i < num_nodes; ++i) {
//             std::cout << dist_data[i] << " ";
//         }
//         std::cout << std::endl;
//     } else {
//         std::cout << "Graph too large to print all distances. Top 100 nodes:" << std::endl;
//         for(int i = 0; i < 100; ++i) {
//             std::cout << "Node " << i << ": " << dist_data[i] << std::endl;
//         }
//     }
    
//     return 0;
// }






////////////////////



// #include <iostream>
// #include <vector>
// #include <fstream>
// #include <sstream>
// #include <chrono>
// #include <algorithm>

// // MLIR MemRef Descriptor
// template <typename T, size_t N>
// struct MemRef {
//     T *allocated;
//     T *aligned;
//     intptr_t offset;
//     intptr_t sizes[N];
//     intptr_t strides[N];
// };

// extern "C" {
//     void _mlir_ciface_bfs(int32_t nodes, int32_t edges, MemRef<int64_t, 2> *coords, MemRef<float, 1> *vals, MemRef<float, 1> *out);
//     void _mlir_ciface_randomwalk(int32_t nodes, int32_t edges, MemRef<int64_t, 2> *coords, MemRef<float, 1> *vals, MemRef<float, 1> *out);
//     void _mlir_ciface_tricount(int32_t nodes, int32_t edges, MemRef<int64_t, 2> *coords, MemRef<float, 1> *vals, MemRef<float, 1> *out);
// }

// // Timer globals
// std::chrono::high_resolution_clock::time_point timer_start, timer_end;
// extern "C" void start_timer() { timer_start = std::chrono::high_resolution_clock::now(); }
// extern "C" void stop_timer() { timer_end = std::chrono::high_resolution_clock::now(); }

// // Struct to ensure lexicographic sorting
// struct Edge {
//     int64_t u, v;
//     float w;
    
//     // Sort primarily by row (u), then by column (v)
//     bool operator<(const Edge& other) const {
//         if (u != other.u) return u < other.u;
//         return v < other.v;
//     }
    
//     // Check for duplicate edges
//     bool operator==(const Edge& other) const {
//         return u == other.u && v == other.v;
//     }
// };

// int main(int argc, char **argv) {
//     std::string filename = (argc > 1) ? argv[1] : "1138_bus.mtx";
//     std::ifstream file(filename);
//     if (!file.is_open()) {
//         std::cerr << "Failed to open " << filename << "\n";
//         return 1;
//     }

//     std::string line;
//     while (std::getline(file, line) && line[0] == '%'); // Skip headers

//     int32_t num_nodes, cols, num_edges_raw;
//     std::stringstream(line) >> num_nodes >> cols >> num_edges_raw;

//     std::vector<Edge> edges;

//     // 1. Load and Symmetrize
//     while (std::getline(file, line)) {
//         int64_t u, v; float w = 1.0f;
//         std::stringstream ss(line);
//         ss >> u >> v;
//         if (ss >> w) {} // read weight if present

//         u--; v--; // 1-based to 0-based index

//         edges.push_back({u, v, w});
//         if (u != v) { // Add symmetric edge to ensure undirected traversal
//             edges.push_back({v, u, w});
//         }
//     }
    
//     // 2. Lexicographical Sorting & Deduplication (Crucial for MLIR SparseTensor)
//     std::sort(edges.begin(), edges.end());
//     edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    
//     int32_t num_edges = edges.size();
    
//     std::cout << "========================================\n";
//     std::cout << "Loading graph: " << filename << "\n";
//     std::cout << "Nodes: " << num_nodes << ", Symmetrized & Sorted Edges: " << num_edges << "\n";
//     std::cout << "========================================\n\n";

//     // 3. Prepare Base Coordinates Array
//     std::vector<int64_t> coords_data;
//     coords_data.reserve(num_edges * 2);
//     for (const auto& e : edges) {
//         coords_data.push_back(e.u);
//         coords_data.push_back(e.v);
//     }
//     MemRef<int64_t, 2> coords_memref = { coords_data.data(), coords_data.data(), 0, {num_edges, 2}, {2, 1} };

//     // ==========================================
//     // BFS Preprocessing (Base Weights)
//     // ==========================================
//     std::vector<float> bfs_vals_data(num_edges);
//     for (size_t i = 0; i < num_edges; ++i) bfs_vals_data[i] = edges[i].w;
    
//     std::vector<float> bfs_out(num_nodes, 0.0f);
//     MemRef<float, 1> bfs_vals_memref = { bfs_vals_data.data(), bfs_vals_data.data(), 0, {num_edges}, {1} };
//     MemRef<float, 1> bfs_out_memref = { bfs_out.data(), bfs_out.data(), 0, {num_nodes}, {1} };

//     std::cout << "--- [1/3] Running GraphBLAS BFS ---\n";
//     auto t1 = std::chrono::high_resolution_clock::now();
//     _mlir_ciface_bfs(num_nodes, num_edges, &coords_memref, &bfs_vals_memref, &bfs_out_memref);
//     auto t2 = std::chrono::high_resolution_clock::now();
    
//     double mlir_time = std::chrono::duration<double, std::milli>(timer_end - timer_start).count();
//     double total_time = std::chrono::duration<double, std::milli>(t2 - t1).count();
    
//     std::cout << "      [MLIR Kernel Time]: " << mlir_time << " ms\n";
//     std::cout << "      [Sample Output - Distances]: " << bfs_out[0] << " " << bfs_out[1] << " " << bfs_out[2] << " ...\n\n";

//     // ==========================================
//     // Random Walk Preprocessing (Row Normalization)
//     // ==========================================
//     std::vector<float> rw_vals_data(num_edges, 0.0f);
//     std::vector<float> row_sums(num_nodes, 0.0f);
    
//     for (const auto& e : edges) row_sums[e.u] += e.w;
    
//     for (size_t i = 0; i < num_edges; ++i) {
//         int64_t row = edges[i].u;
//         if (row_sums[row] > 0) {
//             rw_vals_data[i] = edges[i].w / row_sums[row];
//         }
//     }

//     std::vector<float> rw_out(num_nodes, 0.0f);
//     MemRef<float, 1> rw_vals_memref = { rw_vals_data.data(), rw_vals_data.data(), 0, {num_edges}, {1} };
//     MemRef<float, 1> rw_out_memref = { rw_out.data(), rw_out.data(), 0, {num_nodes}, {1} };

//     std::cout << "--- [2/3] Running GraphBLAS Random Walk (10 Iters) ---\n";
//     t1 = std::chrono::high_resolution_clock::now();
//     _mlir_ciface_randomwalk(num_nodes, num_edges, &coords_memref, &rw_vals_memref, &rw_out_memref);
//     t2 = std::chrono::high_resolution_clock::now();
    
//     mlir_time = std::chrono::duration<double, std::milli>(timer_end - timer_start).count();
//     total_time = std::chrono::duration<double, std::milli>(t2 - t1).count();
    
//     std::cout << "      [MLIR Kernel Time]: " << mlir_time << " ms\n";
//     std::cout << "      [Sample Output - Probabilities]: " << rw_out[0] << " " << rw_out[1] << " " << rw_out[2] << " ...\n\n";

//     for (size_t i = 0; i < 1000; ++i) {
//         std::cout << "Node " << i << ": " << rw_out[i] << std::endl;
//     }
//     std::cout << "\n";

//     // ==========================================
//     // Triangle Count Preprocessing (Binary Matrix)
//     // ==========================================
//     // Force all weights to 1.0 for strictly structural adjacency
//     std::vector<float> tc_vals_data(num_edges, 1.0f); 
    
//     std::vector<float> tc_out(num_nodes, 0.0f);
//     MemRef<float, 1> tc_vals_memref = { tc_vals_data.data(), tc_vals_data.data(), 0, {num_edges}, {1} };
//     MemRef<float, 1> tc_out_memref = { tc_out.data(), tc_out.data(), 0, {num_nodes}, {1} };

//     std::cout << "--- [3/3] Running GraphBLAS Triangle Count ---\n";
//     t1 = std::chrono::high_resolution_clock::now();
//     _mlir_ciface_tricount(num_nodes, num_edges, &coords_memref, &tc_vals_memref, &tc_out_memref);
//     t2 = std::chrono::high_resolution_clock::now();
    
//     mlir_time = std::chrono::duration<double, std::milli>(timer_end - timer_start).count();
//     total_time = std::chrono::duration<double, std::milli>(t2 - t1).count();
    
//     std::cout << "      [MLIR Kernel Time]: " << mlir_time << " ms\n";
//     std::cout << "      [Sample Output - Counts]: " << tc_out[0] << " " << tc_out[1] << " " << tc_out[2] << " ...\n";
//     std::cout << "========================================\n";

//     return 0;
// }



////////////////////////////////


#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>

// MLIR MemRef Descriptor
template <typename T, size_t N>
struct MemRef {
    T *allocated;
    T *aligned;
    intptr_t offset;
    intptr_t sizes[N];
    intptr_t strides[N];
};

extern "C" {
    // MLIR Kernels
    void _mlir_ciface_bfs(int32_t nodes, int32_t edges, MemRef<int64_t, 2> *coords, MemRef<float, 1> *vals, MemRef<float, 1> *out);
    void _mlir_ciface_randomwalk(int32_t nodes, int32_t edges, MemRef<int64_t, 2> *coords, MemRef<float, 1> *vals, MemRef<float, 1> *out);
    void _mlir_ciface_tricount(int32_t nodes, int32_t edges, MemRef<int64_t, 2> *coords, MemRef<float, 1> *vals, MemRef<float, 1> *out);

    // SuiteSparse GraphBLAS Kernels (Make sure to rename these in your *_gblas.cpp files!)
    void _ss_ciface_bfs(int32_t nodes, int32_t edges, MemRef<int64_t, 2> *coords, MemRef<float, 1> *vals, MemRef<float, 1> *out);
    void _ss_ciface_randomwalk(int32_t nodes, int32_t edges, MemRef<int64_t, 2> *coords, MemRef<float, 1> *vals, MemRef<float, 1> *out);
    void _ss_ciface_tricount(int32_t nodes, int32_t edges, MemRef<int64_t, 2> *coords, MemRef<float, 1> *vals, MemRef<float, 1> *out);
}

// Timer globals
std::chrono::high_resolution_clock::time_point timer_start, timer_end;
extern "C" void start_timer() { timer_start = std::chrono::high_resolution_clock::now(); }
extern "C" void stop_timer() { timer_end = std::chrono::high_resolution_clock::now(); }

// Struct to ensure lexicographic sorting
struct Edge {
    int64_t u, v;
    float w;
    bool operator<(const Edge& other) const {
        if (u != other.u) return u < other.u;
        return v < other.v;
    }
    bool operator==(const Edge& other) const {
        return u == other.u && v == other.v;
    }
};

int main(int argc, char **argv) {
    std::string filename = (argc > 1) ? argv[1] : "graphs/enron.mtx";
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << "\n";
        return 1;
    }

    std::string line;
    bool is_symmetric = false;

    // 1. Read MTX Header and detect symmetry flag
    if (std::getline(file, line)) {
        std::string header_lower = line;
        std::transform(header_lower.begin(), header_lower.end(), header_lower.begin(), ::tolower);
        if (header_lower.find("symmetric") != std::string::npos) {
            is_symmetric = true;
        }
        
        while (line.empty() || line[0] == '%') {
            auto pos = file.tellg();
            if (!std::getline(file, line)) break;
            if (!line.empty() && line[0] != '%') {
                file.seekg(pos); 
                std::getline(file, line);
                break;
            }
        }
    }

    int32_t num_nodes, cols, num_edges_raw;
    std::stringstream(line) >> num_nodes >> cols >> num_edges_raw;

    std::vector<Edge> edges;

    // 2. Load and Clean
    while (std::getline(file, line)) {
        int64_t u, v;
        std::stringstream ss(line);
        if (!(ss >> u >> v)) continue; 

        u--; v--; 
        if (u == v) continue; 
        float w = 1.0f; 

        edges.push_back({u, v, w});
        if (is_symmetric) {
            edges.push_back({v, u, w});
        }
    }
    
    // 3. Lexicographical Sorting & Deduplication
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    
    int32_t num_edges = edges.size();
    
    std::cout << "========================================\n";
    std::cout << "Loading graph: " << filename << "\n";
    std::cout << "Nodes: " << num_nodes << "\n";
    std::cout << "Graph Type: " << (is_symmetric ? "Symmetric (Undirected)" : "General (Directed)") << "\n";
    std::cout << "Cleaned & Sorted Edges: " << num_edges << "\n";
    std::cout << "========================================\n\n";

    // 4. Prepare Base Coordinates Array
    std::vector<int64_t> coords_data;
    coords_data.reserve(num_edges * 2);
    for (const auto& e : edges) {
        coords_data.push_back(e.u);
        coords_data.push_back(e.v);
    }
    MemRef<int64_t, 2> coords_memref = { coords_data.data(), coords_data.data(), 0, {num_edges, 2}, {2, 1} };

    // ==========================================
    // BFS Benchmark
    // ==========================================
    std::vector<float> bfs_vals_data(num_edges, 1.0f);
    std::vector<float> bfs_out_mlir(num_nodes, 0.0f);
    std::vector<float> bfs_out_ss(num_nodes, 0.0f);
    
    MemRef<float, 1> bfs_vals_memref = { bfs_vals_data.data(), bfs_vals_data.data(), 0, {num_edges}, {1} };
    MemRef<float, 1> bfs_out_mlir_memref = { bfs_out_mlir.data(), bfs_out_mlir.data(), 0, {num_nodes}, {1} };
    MemRef<float, 1> bfs_out_ss_memref = { bfs_out_ss.data(), bfs_out_ss.data(), 0, {num_nodes}, {1} };

    std::cout << "--- [1/3] Running GraphBLAS BFS ---\n";
    
    // // MLIR
    _mlir_ciface_bfs(num_nodes, num_edges, &coords_memref, &bfs_vals_memref, &bfs_out_mlir_memref);
    double mlir_time = std::chrono::duration<double, std::milli>(timer_end - timer_start).count();
    
    // SuiteSparse
    _ss_ciface_bfs(num_nodes, num_edges, &coords_memref, &bfs_vals_memref, &bfs_out_ss_memref);
    double ss_time = std::chrono::duration<double, std::milli>(timer_end - timer_start).count();

    std::cout << "      [MLIR Time]:        " << mlir_time << " ms\n";
    std::cout << "      [SuiteSparse Time]: " << ss_time << " ms\n";
    std::cout << "      [Sample Out MLIR]:  " << bfs_out_mlir[0] << " " << bfs_out_mlir[1] << " " << bfs_out_mlir[2] << " ...\n";
    std::cout << "      [Sample Out SS]:    " << bfs_out_ss[0] << " " << bfs_out_ss[1] << " " << bfs_out_ss[2] << " ...\n\n";


    // ==========================================
    // Random Walk Benchmark
    // ==========================================
    std::vector<float> rw_vals_data(num_edges, 0.0f);
    std::vector<float> row_sums(num_nodes, 0.0f);
    
    for (const auto& e : edges) row_sums[e.u] += 1.0f; 
    for (size_t i = 0; i < num_edges; ++i) {
        int64_t row = edges[i].u;
        if (row_sums[row] > 0) rw_vals_data[i] = 1.0f / row_sums[row]; 
    }

    std::vector<float> rw_out_mlir(num_nodes, 0.0f);
    std::vector<float> rw_out_ss(num_nodes, 0.0f);
    
    MemRef<float, 1> rw_vals_memref = { rw_vals_data.data(), rw_vals_data.data(), 0, {num_edges}, {1} };
    MemRef<float, 1> rw_out_mlir_memref = { rw_out_mlir.data(), rw_out_mlir.data(), 0, {num_nodes}, {1} };
    MemRef<float, 1> rw_out_ss_memref = { rw_out_ss.data(), rw_out_ss.data(), 0, {num_nodes}, {1} };

    std::cout << "--- [2/3] Running GraphBLAS Random Walk (10 Iters) ---\n";
    
    // MLIR
    _mlir_ciface_randomwalk(num_nodes, num_edges, &coords_memref, &rw_vals_memref, &rw_out_mlir_memref);
    mlir_time = std::chrono::duration<double, std::milli>(timer_end - timer_start).count();
    
    // SuiteSparse
    _ss_ciface_randomwalk(num_nodes, num_edges, &coords_memref, &rw_vals_memref, &rw_out_ss_memref);
    ss_time = std::chrono::duration<double, std::milli>(timer_end - timer_start).count();
    
    std::cout << "      [MLIR Time]:        " << mlir_time << " ms\n";
    std::cout << "      [SuiteSparse Time]: " << ss_time << " ms\n";
    std::cout << "      [Sample Out MLIR]:  " << rw_out_mlir[0] << " " << rw_out_mlir[1] << " " << rw_out_mlir[2] << " ...\n";
    std::cout << "      [Sample Out SS]:    " << rw_out_ss[0] << " " << rw_out_ss[1] << " " << rw_out_ss[2] << " ...\n\n";


    // ==========================================
    // Triangle Count Benchmark
    // ==========================================
    std::vector<float> tc_vals_data(num_edges, 1.0f); 
    std::vector<float> tc_out_mlir(num_nodes, 0.0f);
    std::vector<float> tc_out_ss(num_nodes, 0.0f);
    
    MemRef<float, 1> tc_vals_memref = { tc_vals_data.data(), tc_vals_data.data(), 0, {num_edges}, {1} };
    MemRef<float, 1> tc_out_mlir_memref = { tc_out_mlir.data(), tc_out_mlir.data(), 0, {num_nodes}, {1} };
    MemRef<float, 1> tc_out_ss_memref = { tc_out_ss.data(), tc_out_ss.data(), 0, {num_nodes}, {1} };

    std::cout << "--- [3/3] Running GraphBLAS Triangle Count ---\n";
    
    // MLIR
    _mlir_ciface_tricount(num_nodes, num_edges, &coords_memref, &tc_vals_memref, &tc_out_mlir_memref);
    mlir_time = std::chrono::duration<double, std::milli>(timer_end - timer_start).count();
    
    // SuiteSparse
    _ss_ciface_tricount(num_nodes, num_edges, &coords_memref, &tc_vals_memref, &tc_out_ss_memref);
    ss_time = std::chrono::duration<double, std::milli>(timer_end - timer_start).count();
    
    std::cout << "      [MLIR Time]:        " << mlir_time << " ms\n";
    std::cout << "      [SuiteSparse Time]: " << ss_time << " ms\n";
    std::cout << "      [Sample Out MLIR]:  " << tc_out_mlir[0] << " " << tc_out_mlir[1] << " " << tc_out_mlir[2] << " ...\n";
    std::cout << "      [Sample Out SS]:    " << tc_out_ss[0] << " " << tc_out_ss[1] << " " << tc_out_ss[2] << " ...\n";
    std::cout << "========================================\n";

    return 0;
}