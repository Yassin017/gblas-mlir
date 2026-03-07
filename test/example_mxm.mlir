module {
  func.func @test_mxm(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = gblas.multiply %arg0, %arg1 combine = multiplies reduce = plus : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}

