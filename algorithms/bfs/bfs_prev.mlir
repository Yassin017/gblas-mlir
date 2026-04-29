#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

#SparseVector = #sparse_tensor.encoding<{
  map = (d0) -> (d0 : compressed)
}>

module {
    // 1. Updated BFS signature to take a static 3x3 matrix and return a static 3-length vector
    func.func @bfs(%A: tensor<3x3xf32, #CSR>, %start_node: index) -> tensor<3xf32, #SparseVector> {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %f1 = arith.constant 1.0 : f32
        
        // num_nodes is still useful for the loop boundary
        %num_nodes = gblas.nrows %A : tensor<3x3xf32, #CSR> -> index

        %start_node_i64 = arith.index_cast %start_node : index to i64

        %indices = tensor.empty() : tensor<1x1xi64>
        %indices_filled = tensor.insert %start_node_i64 into %indices[%c0, %c0] : tensor<1x1xi64>
        %vals = tensor.from_elements %f1 : tensor<1xf32>
        
        // --- CHANGE: Output type is now tensor<3xf32... ---
        %v_start = gblas.from_coo %indices_filled, %vals 
            : tensor<1x1xi64>, tensor<1xf32> -> tensor<3xf32, #SparseVector>

        %result:2 = scf.for %i = %c0 to %num_nodes step %c1 
            iter_args(%v_curr = %v_start, %visited_curr = %v_start) 
            -> (tensor<3xf32, #SparseVector>, tensor<3xf32, #SparseVector>) {
            
            // --- CHANGE: Matrix and Vector types are static ---
            %v_next = gblas.vxm %v_curr, %A, %visited_curr
                combine = multiplies 
                reduce = plus 
                {mask_complement = true}
                : tensor<3xf32, #SparseVector>, tensor<3x3xf32, #CSR>, tensor<3xf32, #SparseVector> 
                -> tensor<3xf32, #SparseVector>

            %visited_next = gblas.update %v_next -> %visited_curr 
                {accumulate_operator = "plus"}
                : tensor<3xf32, #SparseVector> -> tensor<3xf32, #SparseVector>

            scf.yield %v_next, %visited_next : tensor<3xf32, #SparseVector>, tensor<3xf32, #SparseVector>
        }

        return %result#1 : tensor<3xf32, #SparseVector>
    }

    func.func @main() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %f1 = arith.constant 1.0 : f32

        %i64_0 = arith.constant 0 : i64
        %i64_1 = arith.constant 1 : i64
        %i64_2 = arith.constant 2 : i64

        %mat_indices = tensor.empty() : tensor<2x2xi64>
        %i0 = tensor.insert %i64_0 into %mat_indices[%c0, %c0] : tensor<2x2xi64>
        %i1 = tensor.insert %i64_1 into %i0[%c0, %c1] : tensor<2x2xi64>
        %i2 = tensor.insert %i64_1 into %i1[%c1, %c0] : tensor<2x2xi64>
        %i3 = tensor.insert %i64_2 into %i2[%c1, %c1] : tensor<2x2xi64>
        
        %mat_vals = tensor.from_elements %f1, %f1 : tensor<2xf32>
        
        // --- CHANGE: Result is static 3x3 ---
        %A = gblas.from_coo %i3, %mat_vals 
            : tensor<2x2xi64>, tensor<2xf32> -> tensor<3x3xf32, #CSR>

        %start_node = arith.constant 0 : index
        
        // --- CHANGE: Function call uses static types ---
        %visited = func.call @bfs(%A, %start_node) : (tensor<3x3xf32, #CSR>, index) -> tensor<3xf32, #SparseVector>

        sparse_tensor.print %visited : tensor<3xf32, #SparseVector>

        return
    }
}