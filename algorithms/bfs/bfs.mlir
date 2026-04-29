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

        %i64_0 = arith.constant 0 : i64
        %i64_1 = arith.constant 1 : i64
        %i64_2 = arith.constant 2 : i64

        // --- 1. Graph Setup (Sparse Matrix) ---
        %mat_indices = tensor.empty() : tensor<2x2xi64>
        %i0 = tensor.insert %i64_0 into %mat_indices[%c0, %c0] : tensor<2x2xi64>
        %i1 = tensor.insert %i64_1 into %i0[%c0, %c1] : tensor<2x2xi64>
        %i2 = tensor.insert %i64_1 into %i1[%c1, %c0] : tensor<2x2xi64>
        %i3 = tensor.insert %i64_2 into %i2[%c1, %c1] : tensor<2x2xi64>
        
        %mat_vals = tensor.from_elements %f1, %f1 : tensor<2xf32>
        
        %A = gblas.from_coo %i3, %mat_vals 
            : tensor<2x2xi64>, tensor<2xf32> -> tensor<3x3xf32, #CSR>

        %num_nodes = gblas.nrows %A : tensor<3x3xf32, #CSR> -> index

        // --- 2. Initial State Allocation ---
        %v_start_1 = tensor.from_elements %f1, %f0, %f0 : tensor<3xf32>
        %v_start_2 = tensor.from_elements %f1, %f0, %f0 : tensor<3xf32>

        // --- 3. Clean Loop ---
        %visited_final:2 = scf.for %i = %c0 to %num_nodes step %c1 
            iter_args(%v_curr = %v_start_1, %visited_curr = %v_start_2) 
            -> (tensor<3xf32>, tensor<3xf32>) {
            
            // 1. Scratch buffer for VXM
            %scratch = tensor.empty() : tensor<3xf32>
            %zeroed_scratch = linalg.fill ins(%f0 : f32) outs(%scratch : tensor<3xf32>) -> tensor<3xf32>

            // 2. Compute next frontier
            %v_computed = gblas.vxm %v_curr, %A outs(%zeroed_scratch), %visited_curr
                combine = multiplies 
                reduce = plus 
                {mask_complement = true}
                : tensor<3xf32>, tensor<3x3xf32, #CSR>, tensor<3xf32>, tensor<3xf32> 
                -> tensor<3xf32>

            // 3. The Bulletproof Combined Loop
            // This safely overwrites v_curr AND safely accumulates into visited_curr
            %v_next, %visited_next = scf.for %j = %c0 to %c3 step %c1 
                iter_args(%v_acc = %v_curr, %vis_acc = %visited_curr) 
                -> (tensor<3xf32>, tensor<3xf32>) {
                
                // Get the computed value for this node
                %val = tensor.extract %v_computed[%j] : tensor<3xf32>
                
                // Task A: Overwrite v_curr (Copy)
                %v_inserted = tensor.insert %val into %v_acc[%j] : tensor<3xf32>
                
                // Task B: Accumulate into visited_curr (Update)
                %old_vis = tensor.extract %vis_acc[%j] : tensor<3xf32>
                %new_vis = arith.addf %old_vis, %val : f32
                %vis_inserted = tensor.insert %new_vis into %vis_acc[%j] : tensor<3xf32>
                
                scf.yield %v_inserted, %vis_inserted : tensor<3xf32>, tensor<3xf32>
            }

            scf.yield %v_next, %visited_next : tensor<3xf32>, tensor<3xf32>
        }

        // --- 4. Print Output ---
        %print_vec = vector.transfer_read %visited_final#1[%c0], %f0 : tensor<3xf32>, vector<3xf32>
        vector.print %print_vec : vector<3xf32>

        return
    }
}