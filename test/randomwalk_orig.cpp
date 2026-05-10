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

extern "C" void _ss_ciface_randomwalk(int32_t nodes, int32_t edges, MemRef<int64_t, 2> *coords, MemRef<float, 1> *vals, MemRef<float, 1> *out) {
    GrB_init(GrB_NONBLOCKING);
    

    std::vector<GrB_Index> I(edges);
    std::vector<GrB_Index> J(edges);
    for(int i = 0; i < edges; i++) {
        I[i] = coords->aligned[i * 2];
        J[i] = coords->aligned[i * 2 + 1];
    }
    float* X = vals->aligned;

    GrB_Matrix A;
    GrB_Matrix_new(&A, GrB_FP32, nodes, nodes);
    GrB_Matrix_build_FP32(A, I.data(), J.data(), X, edges, GrB_PLUS_FP32);

    GrB_Vector v_curr, v_next;
    GrB_Vector_new(&v_curr, GrB_FP32, nodes);
    GrB_Vector_new(&v_next, GrB_FP32, nodes);

    // Initial state: 1.0 probability at node 0
    GrB_Vector_setElement_FP32(v_curr, 1.0f, 0);

    start_timer();

    // 10 Iterations
    for(int i = 0; i < 10; i++) {
        GrB_Vector_clear(v_next);
        
        // v_next = v_curr * A
        GrB_vxm(v_next, GrB_NULL, GrB_NULL, GrB_PLUS_TIMES_SEMIRING_FP32, v_curr, A, GrB_NULL);
        
        // Swap pointers for next iteration
        GrB_Vector temp = v_curr;
        v_curr = v_next;
        v_next = temp;
    }


    GrB_Index nvals;
    GrB_Vector_nvals(&nvals, v_curr);

    


    // Write final probabilities back to MLIR MemRef Out
    for(int i = 0; i < nodes; i++) out->aligned[i] = 0.0f;

    if (nvals > 0) {
        std::vector<GrB_Index> out_I(nvals);
        std::vector<float> out_X(nvals);
        GrB_Vector_extractTuples_FP32(out_I.data(), out_X.data(), &nvals, v_curr);
        for(GrB_Index k = 0; k < nvals; k++) {
            out->aligned[out_I[k]] = out_X[k];
        }
    }

    stop_timer();

    GrB_Matrix_free(&A);
    GrB_Vector_free(&v_curr);
    GrB_Vector_free(&v_next);
    GrB_finalize();
}