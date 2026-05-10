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

extern "C" void _ss_ciface_tricount(int32_t nodes, int32_t edges, MemRef<int64_t, 2> *coords, MemRef<float, 1> *vals, MemRef<float, 1> *out) {
    GrB_init(GrB_NONBLOCKING);

    std::vector<GrB_Index> I(edges);
    std::vector<GrB_Index> J(edges);
    for(int i = 0; i < edges; i++) {
        I[i] = coords->aligned[i * 2];
        J[i] = coords->aligned[i * 2 + 1];
    }
    float* X = vals->aligned;

    GrB_Matrix A, C;
    GrB_Matrix_new(&A, GrB_FP32, nodes, nodes);
    GrB_Matrix_new(&C, GrB_FP32, nodes, nodes);
    GrB_Matrix_build_FP32(A, I.data(), J.data(), X, edges, GrB_PLUS_FP32);

    start_timer();
    
    // C<A> = A * A (Multiply matrix by itself, but only compute where edges already exist)
    GrB_mxm(C, A, GrB_NULL, GrB_PLUS_TIMES_SEMIRING_FP32, A, A, GrB_NULL);

    // Reduce C along the rows to get the triangle count per node
    GrB_Vector counts;
    GrB_Vector_new(&counts, GrB_FP32, nodes);
    
    GrB_Monoid plus_monoid;
    GrB_Monoid_new_FP32(&plus_monoid, GrB_PLUS_FP32, 0.0f);
    
    GrB_Matrix_reduce_Monoid(counts, GrB_NULL, GrB_NULL, plus_monoid, C, GrB_NULL);

    stop_timer();

    // Zero out the MemRef first
    for(int i = 0; i < nodes; i++) out->aligned[i] = 0.0f;
    
    GrB_Index nvals;
    GrB_Vector_nvals(&nvals, counts);
    if (nvals > 0) {
        std::vector<GrB_Index> out_I(nvals);
        std::vector<float> out_X(nvals);
        GrB_Vector_extractTuples_FP32(out_I.data(), out_X.data(), &nvals, counts);
        
        for(GrB_Index k = 0; k < nvals; k++) {
            // Divide by 2 here to correct for duplicate intersection counting
            out->aligned[out_I[k]] = out_X[k] / 2.0f; 
        }
    }

    GrB_Matrix_free(&A);
    GrB_Matrix_free(&C);
    GrB_Vector_free(&counts);
    GrB_Monoid_free(&plus_monoid);
    GrB_finalize();
}