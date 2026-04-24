#COO = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed, d1 : singleton) }>
#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>

func.func @test_graphblas_pipeline(
    %r1: tensor<?xi32>, %c1: tensor<?xi32>, %v1: tensor<?xf32>, 
    %dense_b: tensor<4x4xf32>) -> tensor<4x4xf32> 
    attributes { llvm.emit_c_interface } {
  
  %A = gblas.from_coo %r1, %c1, %v1 : tensor<?xi32>, tensor<?xi32>, tensor<?xf32> -> tensor<4x4xf32, #COO>
  %sparse_sum = gblas.ewise_add %A, %A {op_name = "plus"} : tensor<4x4xf32, #COO>, tensor<4x4xf32, #COO> -> tensor<4x4xf32, #CSR>
  %result = gblas.mxm %sparse_sum, %dense_b combine = multiplies reduce = plus : tensor<4x4xf32, #CSR>, tensor<4x4xf32> -> tensor<4x4xf32>

  return %result : tensor<4x4xf32>
}