

// 1. Matrix-Matrix (Semirings for matrix multiply)
func.func @test_mxm_standard(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // combine = multiplies, reduce = plus
  %res = gblas.mxm %a, %b combine = multiplies reduce = plus : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
  return %res : tensor<4x4xf32>
}

// 2. Matrix-Vector (Min-Plus semiring -for Shortest Path)
func.func @test_mxv_min_plus(%mat: tensor<4x4xf32>, %vec: tensor<4xf32>) -> tensor<4xf32> {
  // combine = plus, reduce = min
  %res = gblas.mxv %mat, %vec combine = plus reduce = min : tensor<4x4xf32>, tensor<4xf32> -> tensor<4xf32>
  return %res : tensor<4xf32>
}

// 3. Vector-Matrix (Max-Plus Semiring)
func.func @test_vxm_max_plus(%vec: tensor<4xf32>, %mat: tensor<4x4xf32>) -> tensor<4xf32> {
  // combine = plus, reduce = max
  %res = gblas.vxm %vec, %mat combine = plus reduce = max : tensor<4xf32>, tensor<4x4xf32> -> tensor<4xf32>
  return %res : tensor<4xf32>
}

// 4. Vector-Vector Dot Product (Max-Min Semiring)
func.func @test_vxv_max_min(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<f32> attributes { llvm.emit_c_interface } {
  // combine = min, reduce = max
  %res = gblas.vxv %a, %b combine = min reduce = max : tensor<4xf32>, tensor<4xf32> -> tensor<f32>
  return %res : tensor<f32>
}