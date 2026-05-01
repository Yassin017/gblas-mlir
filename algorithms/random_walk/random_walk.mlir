#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

module {
    func.func @main() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c10 = arith.constant 10 : index 
        
        %f0 = arith.constant 0.0 : f32
        %f1 = arith.constant 1.0 : f32
        %f_half = arith.constant 0.5 : f32

        %i64_0 = arith.constant 0 : i64
        %i64_1 = arith.constant 1 : i64
        %i64_2 = arith.constant 2 : i64

        // --- 1. Transition Matrix Setup ---
        %mat_indices = tensor.empty() : tensor<4x2xi64>
        // 0 -> 1
        %i0 = tensor.insert %i64_0 into %mat_indices[%c0, %c0] : tensor<4x2xi64>
        %i1 = tensor.insert %i64_1 into %i0[%c0, %c1] : tensor<4x2xi64>
        // 1 -> 0
        %i2 = tensor.insert %i64_1 into %i1[%c1, %c0] : tensor<4x2xi64>
        %i3 = tensor.insert %i64_0 into %i2[%c1, %c1] : tensor<4x2xi64>
        // 1 -> 2
        %i4 = tensor.insert %i64_1 into %i3[%c2, %c0] : tensor<4x2xi64>
        %i5 = tensor.insert %i64_2 into %i4[%c2, %c1] : tensor<4x2xi64>
        // 2 -> 0
        %i6 = tensor.insert %i64_2 into %i5[%c3, %c0] : tensor<4x2xi64>
        %i7 = tensor.insert %i64_0 into %i6[%c3, %c1] : tensor<4x2xi64>
        
        // Probabilities: 1.0, 0.5, 0.5, 1.0
        %mat_vals = tensor.from_elements %f1, %f_half, %f_half, %f1 : tensor<4xf32>
        
        %A = gblas.from_coo %i7, %mat_vals 
            : tensor<4x2xi64>, tensor<4xf32> -> tensor<3x3xf32, #CSR>

        // --- 2. Initial State Allocation ---
        // We start with 100% of our probability mass at Node 0
        %v_start = tensor.from_elements %f1, %f0, %f0 : tensor<3xf32>

        // --- 3. The Random Walk Loop ---
        %v_final = scf.for %i = %c0 to %c10 step %c1 
            iter_args(%v_curr = %v_start) -> (tensor<3xf32>) {
            
            // 1. Scratch buffer for computation
            %scratch = tensor.empty() : tensor<3xf32>
            %zeroed_scratch = linalg.fill ins(%f0 : f32) outs(%scratch : tensor<3xf32>) -> tensor<3xf32>

            // 2. Compute next probabilities into the scratch buffer
            %v_computed = gblas.vxm %v_curr, %A outs(%zeroed_scratch)
                combine = multiplies 
                reduce = plus 
                : tensor<3xf32>, tensor<3x3xf32, #CSR>, tensor<3xf32> 
                -> tensor<3xf32>

            // 3. THE FIX: The Bulletproof Copy Loop
            // Safely copy values back into the loop-carried variable to satisfy bufferization
            %v_next = scf.for %j = %c0 to %c3 step %c1 
                iter_args(%v_acc = %v_curr) -> (tensor<3xf32>) {
                
                %val = tensor.extract %v_computed[%j] : tensor<3xf32>
                %v_inserted = tensor.insert %val into %v_acc[%j] : tensor<3xf32>
                
                scf.yield %v_inserted : tensor<3xf32>
            }

            scf.yield %v_next : tensor<3xf32>
        }

        // --- 4. Print Output ---
        // After 10 iterations, print the probability distribution
        %print_vec = vector.transfer_read %v_final[%c0], %f0 : tensor<3xf32>, vector<3xf32>
        vector.print %print_vec : vector<3xf32>

        return
    }
}