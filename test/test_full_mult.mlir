#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
#CV  = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

func.func @test_full_suite(
    // Matrix A Data: 3x3 with 5 non-zeros
    %m_idx: tensor<5x2xi32>, %m_val: tensor<5xf32>,
    // Vector X Data: Size 3 with 3 non-zeros
    %v_idx: tensor<3x1xi32>, %v_val: tensor<3xf32>
  ) attributes { llvm.emit_c_interface } {

  // 1. Build Sparse Structures
  %A = gblas.from_coo %m_idx, %m_val : tensor<5x2xi32>, tensor<5xf32> -> tensor<3x3xf32, #CSR>
  %X = gblas.from_coo %v_idx, %v_val : tensor<3x1xi32>, tensor<3xf32> -> tensor<3xf32, #CV>

 // 2. MXM: Matrix-Matrix (A * A)
  %res_mxm = gblas.mxm %A, %A combine = multiplies reduce = plus 
      : tensor<3x3xf32, #CSR>, tensor<3x3xf32, #CSR> -> tensor<3x3xf32, #CSR>

  // 3. MXV: Matrix-Vector (A * X)
  %res_mxv = gblas.mxv %A, %X combine = multiplies reduce = plus 
      : tensor<3x3xf32, #CSR>, tensor<3xf32, #CV> -> tensor<3xf32, #CV>

  // 4. VXM: Vector-Matrix (X * A)
  %res_vxm = gblas.vxm %X, %A combine = multiplies reduce = plus 
      : tensor<3xf32, #CV>, tensor<3x3xf32, #CSR> -> tensor<3xf32, #CV>

  // 5. VXV: Vector-Vector (X * X)
  %res_vxv = gblas.vxv %X, %X combine = multiplies reduce = plus 
      : tensor<3xf32, #CV>, tensor<3xf32, #CV> -> tensor<f32>


  // Verification Printing
  sparse_tensor.print %res_mxm : tensor<3x3xf32, #CSR>
  sparse_tensor.print %res_mxv : tensor<3xf32, #CV>
  sparse_tensor.print %res_vxm : tensor<3xf32, #CV>

  %scalar_vxv = tensor.extract %res_vxv[] : tensor<f32>
  vector.print %scalar_vxv : f32

  return
}