// func.func @test_graphblas_pipeline( %rows: tensor<3xi32>, %cols: tensor<3xi32>, %vals: tensor<3xf32>, %rhs: tensor<4x4xf32>) -> tensor<4x4xf32> {
  
//   // Build the lhs matrix from coo
//   %lhs = gblas.from_coo %rows, %cols, %vals : 
//          tensor<3xi32>, tensor<3xi32>, tensor<3xf32> -> tensor<4x4xf32>

//   // Perform the mxm
//   %result = gblas.mxm %lhs, %rhs combine = multiplies reduce = plus : 
//             tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>

//   return %result : tensor<4x4xf32>
// }


// 1. Define the Sparse Encoding Alias at the top of the file
#COO = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed, d1 : singleton) }>

func.func @test_graphblas_pipeline( %rows: tensor<3xi32>, %cols: tensor<3xi32>, %vals: tensor<3xf32>, %rhs: tensor<4x4xf32>) -> tensor<4x4xf32> {
  
  // 2. Build the lhs matrix from coo (Notice the return type now uses #COO)
  %lhs = gblas.from_coo %rows, %cols, %vals : 
         tensor<3xi32>, tensor<3xi32>, tensor<3xf32> -> tensor<4x4xf32, #COO>

  // 3. Perform the mxm (Notice the first input type now uses #COO)
  // This is now a Sparse-Matrix Dense-Matrix Multiplication (SpMM)!
  %result = gblas.mxm %lhs, %rhs combine = multiplies reduce = plus : 
            tensor<4x4xf32, #COO>, tensor<4x4xf32> -> tensor<4x4xf32>

  return %result : tensor<4x4xf32>
}