// 1. Define Encodings
#COO = #sparse_tensor.encoding<{ 
  map = (d0, d1) -> (d0 : compressed, d1 : singleton) 
}>
#CSR = #sparse_tensor.encoding<{ 
  map = (d0, d1) -> (d0 : dense, d1 : compressed) 
}>

module {
  func.func @test_hybrid_gblas_pipeline(
      %r1: tensor<2xi32>, %c1: tensor<2xi32>, %v1: tensor<2xf32>,
      %r2: tensor<2xi32>, %c2: tensor<2xi32>, %v2: tensor<2xf32>,
      %dense_rhs: tensor<4x4xf32>) -> tensor<4x4xf32> {
    
    // Step A: Build Sparse LHS 1
    %A = gblas.from_coo %r1, %c1, %v1 : 
         tensor<2xi32>, tensor<2xi32>, tensor<2xf32> -> tensor<4x4xf32, #COO>

    // Step B: Build Sparse LHS 2
    %B = gblas.from_coo %r2, %c2, %v2 : 
         tensor<2xi32>, tensor<2xi32>, tensor<2xf32> -> tensor<4x4xf32, #COO>

    // Step C: Sparse + Sparse -> Sparse (Your New Op!)
    %sparse_sum = gblas.ewise_add %A, %B {op_name = "plus"} : 
                  tensor<4x4xf32, #COO>, tensor<4x4xf32, #COO> -> tensor<4x4xf32, #CSR>

    // Step D: Sparse Matrix * Dense Matrix -> Dense Result
    // This uses the result of your ewise_add as the LHS for the multiply
    %result = gblas.mxm %sparse_sum, %dense_rhs combine = multiplies reduce = plus : 
              tensor<4x4xf32, #CSR>, tensor<4x4xf32> -> tensor<4x4xf32>

    return %result : tensor<4x4xf32>
  }
}