extern "C" {
    #include <GraphBLAS.h>
}
#include <vector>
#include <cstdint>

template <typename T, size_t N>
struct MemRef {
    T *allocated;
    T *aligned;
    intptr_t offset;
    intptr_t sizes[N];
    intptr_t strides[N];
};

extern "C" void start_timer();
extern "C" void stop_timer();

extern "C" void _ss_ciface_bfs(int32_t nodes, int32_t edges, MemRef<int64_t, 2> *coords, MemRef<float, 1> *vals, MemRef<float, 1> *out) {
    GrB_init(GrB_NONBLOCKING);
    

    // 1. Prepare MLIR MemRef data for standard GraphBLAS
    std::vector<GrB_Index> I(edges);
    std::vector<GrB_Index> J(edges);
    for(int i = 0; i < edges; i++) {
        I[i] = coords->aligned[i * 2];
        J[i] = coords->aligned[i * 2 + 1];
    }
    float* X = vals->aligned;

    // 2. Build the Adjacency Matrix
    GrB_Matrix A;
    GrB_Matrix_new(&A, GrB_FP32, nodes, nodes);
    GrB_Matrix_build_FP32(A, I.data(), J.data(), X, edges, GrB_PLUS_FP32);

    // 3. Initialize BFS state
    GrB_Vector v_curr, visited, v_computed;
    GrB_Vector_new(&v_curr, GrB_FP32, nodes);
    GrB_Vector_new(&visited, GrB_FP32, nodes);
    GrB_Vector_new(&v_computed, GrB_FP32, nodes);

    // Start at Node 0
    GrB_Vector_setElement_FP32(v_curr, 1.0f, 0);
    GrB_Vector_setElement_FP32(visited, 1.0f, 0);

    // Initialize MLIR output array (distances) to -1.0
    for(int i = 0; i < nodes; i++) out->aligned[i] = -1.0f;
    out->aligned[0] = 0.0f;

    // Set Descriptor for mask_complement = true
    GrB_Descriptor desc;
    GrB_Descriptor_new(&desc);
    GrB_Descriptor_set(desc, GrB_MASK, GrB_COMP);

    start_timer(); // MLIR timer start

    // 4. BFS Loop
    for(int i = 0; i < nodes; i++) {
        GrB_Vector_clear(v_computed);
        
        // v_computed<!visited> = v_curr * A
        GrB_vxm(v_computed, visited, GrB_NULL, GrB_PLUS_TIMES_SEMIRING_FP32, v_curr, A, desc);

        GrB_Index nvals;
        GrB_Vector_nvals(&nvals, v_computed);
        if (nvals == 0) break; // Early exit if frontier is empty

        // Extract newly visited nodes
        std::vector<GrB_Index> active_I(nvals);
        std::vector<float> active_X(nvals);
        GrB_Vector_extractTuples_FP32(active_I.data(), active_X.data(), &nvals, v_computed);

        GrB_Vector_clear(v_curr); // Setup next frontier
        for(GrB_Index k = 0; k < nvals; k++) {
            GrB_Index node = active_I[k];
            
            // Task A: Accumulate Distance and Visited Flags
            out->aligned[node] = (float)(i + 1);
            GrB_Vector_setElement_FP32(visited, 1.0f, node);
            
            // Task B: Overwrite v_curr
            GrB_Vector_setElement_FP32(v_curr, active_X[k], node);
        }
    }

    stop_timer(); // MLIR timer stop

    GrB_Matrix_free(&A);
    GrB_Vector_free(&v_curr);
    GrB_Vector_free(&visited);
    GrB_Vector_free(&v_computed);
    GrB_Descriptor_free(&desc);
    GrB_finalize();
}