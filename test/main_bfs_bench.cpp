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


#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>


static std::chrono::time_point<std::chrono::high_resolution_clock> bfs_start_time;

extern "C" {
    // MLIR will call this right before the math starts
    void start_timer() {
        bfs_start_time = std::chrono::high_resolution_clock::now();
    }

    // MLIR will call this right after the math finishes
    void stop_timer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end_time - bfs_start_time;
        std::cout << "\n>>> BFS Kernel Time: " << duration.count() << " ms <<<\n" << std::endl;
    }
}


struct CooEntry {
    int64_t row, col;
    float val;
    bool operator<(const CooEntry& other) const {
        if (row != other.row) return row < other.row;
        return col < other.col;
    }
};

bool readMatrixMarket(const std::string& filename, int32_t& num_nodes, int32_t& num_edges,
                      std::vector<int64_t>& coords, std::vector<float>& vals) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;

    std::string line;
    bool dimensions_read = false;
    bool is_symmetric = false;
    std::vector<CooEntry> entries;

    while (std::getline(file, line)) {
        // Check for symmetry in the header
        if (line.find("%%MatrixMarket") != std::string::npos && line.find("symmetric") != std::string::npos) {
            is_symmetric = true;
        }

        if (line.empty() || line[0] == '%') continue;

        std::istringstream iss(line);

        if (!dimensions_read) {
            int32_t rows, cols, raw_edges;
            iss >> rows >> cols >> raw_edges;
            num_nodes = std::max(rows, cols); 
            // If symmetric, we might end up with nearly 2x the edges
            entries.reserve(is_symmetric ? raw_edges * 2 : raw_edges);
            dimensions_read = true;
        } else {
            int64_t u, v;
            float weight = 1.0f;
            iss >> u >> v;
            if (!(iss >> weight)) weight = 1.0f;

            // Convert 1-based to 0-based indexing
            entries.push_back({u - 1, v - 1, weight});
            
            // THE SYMMETRIC FIX: Mirror the edge if needed
            if (is_symmetric && u != v) {
                entries.push_back({v - 1, u - 1, weight});
            }
        }
    }

    std::cout << "Symmetric graph detected: " << (is_symmetric ? "Yes" : "No") << std::endl;
    std::cout << "Sorting " << entries.size() << " directed edges..." << std::endl;
    std::sort(entries.begin(), entries.end());

    // Update actual edge count after mirroring
    num_edges = entries.size();
    
    coords.reserve(num_edges * 2);
    vals.reserve(num_edges);
    for (const auto& entry : entries) {
        coords.push_back(entry.row);
        coords.push_back(entry.col);
        vals.push_back(entry.val);
    }
    return true;
}



template <typename T, size_t N>
struct MemRef {
    T* allocated;
    T* aligned;
    intptr_t offset;
    intptr_t sizes[N];
    intptr_t strides[N];
};

extern "C" {
    void _mlir_ciface_bfs(
        int32_t num_nodes, 
        int32_t num_edges, 
        MemRef<int64_t, 2>* coords, 
        MemRef<float, 1>* vals, 
        MemRef<float, 1>* out_dist
    );
}



int main() {
    int32_t num_nodes = 0;
    int32_t num_edges = 0;
    std::vector<int64_t> coords_data;
    std::vector<float> vals_data;

    // Assuming you downloaded a graph named "benchmark_graph.mtx"
    std::string filename = "1138_bus.mtx";
    std::cout << "Loading graph from " << filename << "..." << std::endl;

    if (!readMatrixMarket(filename, num_nodes, num_edges, coords_data, vals_data)) {
        return 1; // Exit if file reading fails
    }

    std::cout << "Graph loaded" << std::endl;
    std::cout << "Nodes: " << num_nodes << ", Edges: " << num_edges << std::endl;

    std::vector<float> dist_data(num_nodes, -1.0f);

    // Setup MemRefs
    MemRef<int64_t, 2> coords_memref = {
        coords_data.data(), coords_data.data(), 0,
        {num_edges, 2}, {2, 1}
    };
    MemRef<float, 1> vals_memref = {
        vals_data.data(), vals_data.data(), 0,
        {num_edges}, {1}
    };
    MemRef<float, 1> dist_memref = {
        dist_data.data(), dist_data.data(), 0,
        {num_nodes}, {1}
    };

    std::cout << "Starting GBLAS BFS"  << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    // Call the MLIR code
    _mlir_ciface_bfs(num_nodes, num_edges, &coords_memref, &vals_memref, &dist_memref);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    
    std::cout << "BFS Full Time: " << duration.count() << " ms" << std::endl;
    
    // Only print distances if the graph is small enough, otherwise just print a sample
    if (num_nodes <= 100) {
        std::cout << "Computed Distances: ";
        for(int i = 0; i < num_nodes; ++i) {
            std::cout << dist_data[i] << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "Graph too large to print all distances. Top 100 nodes:" << std::endl;
        for(int i = 0; i < 100; ++i) {
            std::cout << "Node " << i << ": " << dist_data[i] << std::endl;
        }
    }
    
    return 0;
}