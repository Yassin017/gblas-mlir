// RUN: ./tools/gblas-opt %s --convert-gblas-to-linalg --linalg-generalize-named-ops --sparsification-and-bufferization

#COO = #sparse_tensor.encoding<{ 
  map = (d0, d1) -> (d0 : compressed, d1 : singleton) 
}>
#CSR = #sparse_tensor.encoding<{ 
  map = (d0, d1) -> (d0 : dense, d1 : compressed) 
}>

module {
  func.func @test_sparse_update_pipeline(
      %r1: tensor<2xi32>, %c1: tensor<2xi32>, %v1: tensor<2xf32>,
      %initial_sparse: tensor<4x4xf32, #CSR>,
      %dense_rhs: tensor<4x4xf32>) -> tensor<4x4xf32> {
    
    // Step A: Build the "Incoming" sparse data
    %incoming = gblas.from_coo %r1, %c1, %v1 : 
                tensor<2xi32>, tensor<2xi32>, tensor<2xf32> -> tensor<4x4xf32, #COO>

    // Step B: Update the existing sparse tensor.
    // Note: We capture the result %updated. Even if gblas conceptually 
    // modifies %initial_sparse, in MLIR we use the new SSA value.
    %updated = gblas.update %incoming -> %initial_sparse {replace = true} : 
               tensor<4x4xf32, #COO> -> tensor<4x4xf32, #CSR>

    // Step C: Verify the update by using it in mxm
    // If the sparsifier works, this mxm will use the merged sparse indices 
    // from both %incoming and %initial_sparse.
    %result = gblas.mxm %updated, %dense_rhs combine = multiplies reduce = plus : 
              tensor<4x4xf32, #CSR>, tensor<4x4xf32> -> tensor<4x4xf32>

    return %result : tensor<4x4xf32>
  }
}