#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

module {
    func.func @main() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        
        %f0 = arith.constant 0.0 : f32
        %f1 = arith.constant 1.0 : f32
        
        // NEW: Positive Infinity for SSSP initialization
        %inf = arith.constant 0x7F800000 : f32

        %i64_0 = arith.constant 0 : i64
        %i64_1 = arith.constant 1 : i64
        %i64_2 = arith.constant 2 : i64

        // --- 1. Graph Setup (Sparse Matrix) ---
        %mat_indices = tensor.empty() : tensor<2x2xi64>
        %i0 = tensor.insert %i64_0 into %mat_indices[%c0, %c0] : tensor<2x2xi64>
        %i1 = tensor.insert %i64_1 into %i0[%c0, %c1] : tensor<2x2xi64>
        %i2 = tensor.insert %i64_1 into %i1[%c1, %c0] : tensor<2x2xi64>
        %i3 = tensor.insert %i64_2 into %i2[%c1, %c1] : tensor<2x2xi64>
        
        // Edge weights are all 1.0
        %mat_vals = tensor.from_elements %f1, %f1 : tensor<2xf32>
        
        %A = gblas.from_coo %i3, %mat_vals 
            : tensor<2x2xi64>, tensor<2xf32> -> tensor<3x3xf32, #CSR>

        // --- 2. Initial State Allocation ---
        // Node 0 starts at distance 0.0. Nodes 1 and 2 start at Infinity.
        %dist_start = tensor.from_elements %f0, %inf, %inf : tensor<3xf32>

        // Bellman-Ford runs for exactly (V - 1) iterations. 
        // For 3 nodes, V-1 = 2. Iterating from %c0 to %c2 step %c1 executes 2 times.
        
        // --- 3. The SSSP Loop ---
        %dist_final = scf.for %i = %c0 to %c2 step %c1 
            iter_args(%dist_curr = %dist_start) 
            -> (tensor<3xf32>) {
            
            // 1. Scratch buffer for VXM (MUST be filled with Infinity for reduce=min)
            %scratch = tensor.empty() : tensor<3xf32>
            %inf_scratch = linalg.fill ins(%inf : f32) outs(%scratch : tensor<3xf32>) -> tensor<3xf32>

            // 2. Compute min-plus matrix multiplication (Relaxing all edges)
            // No mask is needed for Bellman-Ford!
            %tmp_dist = gblas.vxm %dist_curr, %A outs(%inf_scratch)
                combine = plus 
                reduce = min 
                : tensor<3xf32>, tensor<3x3xf32, #CSR>, tensor<3xf32> 
                -> tensor<3xf32>

            // 3. Bulletproof Element-wise Minimum Update
            // This bypasses gblas.update to guarantee safety with dense tensors
            %dist_next = scf.for %j = %c0 to %c3 step %c1 
                iter_args(%d_acc = %dist_curr) 
                -> (tensor<3xf32>) {
                
                // Extract old distance and newly computed distance
                %old_val = tensor.extract %d_acc[%j] : tensor<3xf32>
                %new_val = tensor.extract %tmp_dist[%j] : tensor<3xf32>
                
                // Determine the minimum
                %min_val = arith.minimumf %old_val, %new_val : f32
                
                // Insert back into the accumulator
                %inserted = tensor.insert %min_val into %d_acc[%j] : tensor<3xf32>
                scf.yield %inserted : tensor<3xf32>
            }

            scf.yield %dist_next : tensor<3xf32>
        }

        // --- 4. Print Output ---
        // Expected Output: ( 0, 1, 2 )
        %print_vec = vector.transfer_read %dist_final[%c0], %f0 : tensor<3xf32>, vector<3xf32>
        vector.print %print_vec : vector<3xf32>

        return
    }
}