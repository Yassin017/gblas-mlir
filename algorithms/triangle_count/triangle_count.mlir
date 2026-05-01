#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

module {
    func.func @main() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index

        %f1 = arith.constant 1.0 : f32
        %f0 = arith.constant 0.0 : f32

        %i64_0 = arith.constant 0 : i64
        %i64_1 = arith.constant 1 : i64
        %i64_2 = arith.constant 2 : i64
        %i64_3 = arith.constant 3 : i64

        // --- 1. Graph Setup (8 Directed Edges for an Undirected Graph) ---
        %mat_indices = tensor.empty() : tensor<8x2xi64>
        
        // Node 0 -> 1, 2
        %i0 = tensor.insert %i64_0 into %mat_indices[%c0, %c0] : tensor<8x2xi64>
        %i1 = tensor.insert %i64_1 into %i0[%c0, %c1] : tensor<8x2xi64>
        %i2 = tensor.insert %i64_0 into %i1[%c1, %c0] : tensor<8x2xi64>
        %i3 = tensor.insert %i64_2 into %i2[%c1, %c1] : tensor<8x2xi64>

        // Node 1 -> 0, 2
        %i4 = tensor.insert %i64_1 into %i3[%c2, %c0] : tensor<8x2xi64>
        %i5 = tensor.insert %i64_0 into %i4[%c2, %c1] : tensor<8x2xi64>
        %i6 = tensor.insert %i64_1 into %i5[%c3, %c0] : tensor<8x2xi64>
        %i7 = tensor.insert %i64_2 into %i6[%c3, %c1] : tensor<8x2xi64>

        // Node 2 -> 0, 1, 3
        %i8 = tensor.insert %i64_2 into %i7[%c4, %c0] : tensor<8x2xi64>
        %i9 = tensor.insert %i64_0 into %i8[%c4, %c1] : tensor<8x2xi64>
        %i10 = tensor.insert %i64_2 into %i9[%c5, %c0] : tensor<8x2xi64>
        %i11 = tensor.insert %i64_1 into %i10[%c5, %c1] : tensor<8x2xi64>
        %i12 = tensor.insert %i64_2 into %i11[%c6, %c0] : tensor<8x2xi64>
        %i13 = tensor.insert %i64_3 into %i12[%c6, %c1] : tensor<8x2xi64>

        // Node 3 -> 2
        %i14 = tensor.insert %i64_3 into %i13[%c7, %c0] : tensor<8x2xi64>
        %i15 = tensor.insert %i64_2 into %i14[%c7, %c1] : tensor<8x2xi64>

        // All edges have a weight of 1.0
        %mat_vals = tensor.from_elements %f1, %f1, %f1, %f1, %f1, %f1, %f1, %f1 : tensor<8xf32>

        // Build the Adjacency Matrix
        %A = gblas.from_coo %i15, %mat_vals 
            : tensor<8x2xi64>, tensor<8xf32> -> tensor<4x4xf32, #CSR>

        // --- 2. Triangle Counting (C = A * A masked by A) ---
        // We use (plus, multiplies) to count the overlapping paths.
        // Note: No outs() argument! Your C++ code handles it.
        // Assuming your dialect syntax parses the mask simply as a third operand or with a 'mask' keyword:
        %C = gblas.mxm %A, %A, %A
            combine = multiplies 
            reduce = plus 
            : tensor<4x4xf32, #CSR>, tensor<4x4xf32, #CSR>, tensor<4x4xf32, #CSR> 
            -> tensor<4x4xf32>

        // --- 3. Print Row 0 ---
        // Node 0 is connected to 1 and 2, which form a triangle.
        // We expect to see ( 0, 1, 1, 0 )
        %print_vec = vector.transfer_read %C[%c0, %c0], %f0 : tensor<4x4xf32>, vector<4xf32>
        vector.print %print_vec : vector<4xf32>

        return
    }
}