// func.func @test_graphblas_pipeline( %rows: tensor<3xi32>, %cols: tensor<3xi32>, %vals: tensor<3xf32>, %rhs: tensor<4x4xf32>) -> tensor<4x4xf32> {
  
//   // Build the lhs matrix from coo
//   %lhs = gblas.from_coo %rows, %cols, %vals : 
//          tensor<3xi32>, tensor<3xi32>, tensor<3xf32> -> tensor<4x4xf32>

//   // Perform the mxm
//   %result = gblas.mxm %lhs, %rhs combine = multiplies reduce = plus : 
//             tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>

//   return %result : tensor<4x4xf32>
// }

////////

// // 1. Define the Sparse Encoding Alias at the top of the file
// #COO = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed, d1 : singleton) }>

// func.func @test_graphblas_pipeline( %rows: tensor<3xi32>, %cols: tensor<3xi32>, %vals: tensor<3xf32>, %rhs: tensor<4x4xf32>) -> tensor<4x4xf32> {
  
//   // 2. Build the lhs matrix from coo (Notice the return type now uses #COO)
//   %lhs = gblas.from_coo %rows, %cols, %vals : 
//          tensor<3xi32>, tensor<3xi32>, tensor<3xf32> -> tensor<4x4xf32, #COO>

//   // 3. Perform the mxm (Notice the first input type now uses #COO)
//   // This is now a Sparse-Matrix Dense-Matrix Multiplication (SpMM)!
//   %result = gblas.mxm %lhs, %rhs combine = multiplies reduce = plus : 
//             tensor<4x4xf32, #COO>, tensor<4x4xf32> -> tensor<4x4xf32>

//   return %result : tensor<4x4xf32>
// }


//////

// #COO = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed, d1 : singleton) }>

// func.func @test_graphblas_pipeline(
//   %rows: tensor<3xi32>, 
//   %cols: tensor<3xi32>, 
//   %vals: tensor<3xf32>, 
//   %rhs_rows: tensor<3xi32>,
//   %rhs_cols: tensor<3xi32>,
//   %rhs_vals: tensor<3xf32>
// ) -> tensor<4x4xf32, #COO> {

//   // Build lhs and rhs as sparse tensors
//   %lhs = gblas.from_coo %rows, %cols, %vals : 
//          tensor<3xi32>, tensor<3xi32>, tensor<3xf32> -> tensor<4x4xf32, #COO>
//   %rhs = gblas.from_coo %rhs_rows, %rhs_cols, %rhs_vals : 
//          tensor<3xi32>, tensor<3xi32>, tensor<3xf32> -> tensor<4x4xf32, #COO>

//   // Both inputs and the result are now sparse tensors
//   %result = gblas.mxm %lhs, %rhs combine = multiplies reduce = plus : 
//             tensor<4x4xf32, #COO>, tensor<4x4xf32, #COO> -> tensor<4x4xf32, #COO>

//   return %result : tensor<4x4xf32, #COO>
// }


//////

// 1. Updated to CSR encoding: better for GPU performance
#CSR = #sparse_tensor.encoding<{ 
  map = (d0, d1) -> (d0 : dense, d1 : compressed) 
}>

func.func @test_graphblas_pipeline(
  %rows: tensor<3xi32>, 
  %cols: tensor<3xi32>, 
  %vals: tensor<3xf32>, 
  %rhs_rows: tensor<3xi32>,
  %rhs_cols: tensor<3xi32>,
  %rhs_vals: tensor<3xf32>
) -> tensor<4x4xf32> { // 2. Change: Return a DENSE tensor

  // Build lhs and rhs as sparse tensors (CSR)
  %lhs = gblas.from_coo %rows, %cols, %vals : 
         tensor<3xi32>, tensor<3xi32>, tensor<3xf32> -> tensor<4x4xf32, #CSR>
  %rhs = gblas.from_coo %rhs_rows, %rhs_cols, %rhs_vals : 
         tensor<3xi32>, tensor<3xi32>, tensor<3xf32> -> tensor<4x4xf32, #CSR>

  // 3. Change: The result is now a dense tensor.
  // This resolves the linalg.matmul type mismatch error.
  %result = gblas.mxm %lhs, %rhs combine = multiplies reduce = plus : 
            tensor<4x4xf32, #CSR>, tensor<4x4xf32, #CSR> -> tensor<4x4xf32>

  return %result : tensor<4x4xf32>
}