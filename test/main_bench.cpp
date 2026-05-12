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







int run_benchmark(std::string filename) {

    // std::string fnames[] = {
    //     "graphs/1138_bus.mtx", //1.1k nodes, 2.5k edges, weighted, sym
    //     "graphs/enron.mtx",   //69k nodes, 276k edges, unweighted
    //     "graphs/vsp_finan512_scagr7-2c_rlfddd.mtx", // 139k nodes, 552k edges, random, unweighted, sym (random)
    //     "graphs/roadNet-PA.mtx" //1M nodes, 3M edges, unweighted, sym (road network)
    // };

    // std::string filename = (argc > 1) ? argv[1] : fnames[3];
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
    
    // MLIR
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

    std::cout << "--- [2/3] Running GraphBLAS Random Walk (100 Iters) ---\n";
    
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



int main(int argc, char **argv) {

    std::string fnames[] = {
        "graphs/1138_bus.mtx", //1.1k nodes, 2.5k edges, weighted, sym
        // "graphs/enron.mtx",   //69k nodes, 276k edges, unweighted
        // "graphs/vsp_finan512_scagr7-2c_rlfddd.mtx", // 139k nodes, 552k edges, random, unweighted, sym (random)
        // "graphs/roadNet-PA.mtx", //1M nodes, 3M edges, unweighted, sym (road network)

        "graphs/small/G11.mtx",
        "graphs/small/CSphd.mtx",
        "graphs/small/USpowerGrid.mtx",
        "graphs/small/netscience.mtx",

        "graphs/medium/G63.mtx",
        "graphs/medium/bcsstk13.mtx",
        "graphs/medium/ca-HepTh.mtx",


        "graphs/big/rgg_n_2_15_s0.mtx", //rand
        "graphs/big/fe_body.mtx", //road
        "graphs/big/cond-mat-2005.mtx", //power

        "graphs/huge/G_n_pin_pout.mtx", //rand
        "graphs/huge/roadNet-TX.mtx", //road
        "graphs/huge/com-DBLP.mtx" //power
    };

    // std::string filename = (argc > 1) ? argv[1] : fnames[0];

    if (argc > 1){
        std::string filename = argv[1];
        std::cout << "\n\n=============================\n";
        std::cout << "Benchmarking: " << filename << "\n";
        std::cout << "=============================\n";

        run_benchmark(filename);
    }
    else{
        for (int i = 0; i < 13; ++i) {
            std::cout << "\n\n=============================\n";
            std::cout << "Benchmarking: " << fnames[i] << "\n";
            std::cout << "=============================\n";
            run_benchmark(fnames[i]);
        }

    }    
  
    return 0;
}