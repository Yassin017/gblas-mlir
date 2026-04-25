#CSR = #sparse_tensor.encoding<{ 
  map = (d0, d1) -> (d0 : dense, d1 : compressed) 
}>

module {
  func.func @main() -> i32 {

    %dense_a = arith.constant dense<[
      [1.0, 0.0, 2.0, 0.0], 
      [0.0, 3.0, 0.0, 0.0], 
      [0.0, 0.0, 4.0, 0.0], 
      [5.0, 0.0, 0.0, 6.0]
    ]> : tensor<4x4xf32>
    
    %dense_b = arith.constant dense<[
      [1.0, 0.0, 0.0, 0.0], 
      [0.0, 3.0, 0.0, 0.0], 
      [0.0, 0.0, 0.0, 0.0], 
      [5.0, 0.0, 0.0, 1.0]
    ]> : tensor<4x4xf32>

    %dense_mask = arith.constant dense<[
      [1.0, 1.0, 1.0, 1.0], 
      [1.0, 1.0, 1.0, 1.0], 
      [0.0, 0.0, 0.0, 0.0], 
      [0.0, 0.0, 0.0, 0.0]
    ]> : tensor<4x4xf32>


    %sparse_a = sparse_tensor.convert %dense_a : tensor<4x4xf32> to tensor<4x4xf32, #CSR>
    %sparse_b = sparse_tensor.convert %dense_b : tensor<4x4xf32> to tensor<4x4xf32, #CSR>
    %sparse_mask = sparse_tensor.convert %dense_mask : tensor<4x4xf32> to tensor<4x4xf32, #CSR>


    // Test A: Intersect without mask
    %result_no_mask = gblas.intersect multiplies %sparse_a, %sparse_b : (tensor<4x4xf32, #CSR>, tensor<4x4xf32, #CSR>) -> tensor<4x4xf32, #CSR>
    sparse_tensor.print %result_no_mask : tensor<4x4xf32, #CSR>

    // Test B: Intersect with mask
    %result_with_mask = gblas.intersect min %sparse_a, %sparse_b, %sparse_mask : (tensor<4x4xf32, #CSR>, tensor<4x4xf32, #CSR>, tensor<4x4xf32, #CSR>) -> tensor<4x4xf32, #CSR>
    sparse_tensor.print %result_with_mask : tensor<4x4xf32, #CSR>

    // Test C: Intersect with mask and mask_complement (using attr-dict)
    %result_mask_comp = gblas.intersect max %sparse_a, %sparse_b, %sparse_mask {mask_complement = true} : (tensor<4x4xf32, #CSR>, tensor<4x4xf32, #CSR>, tensor<4x4xf32, #CSR>) -> tensor<4x4xf32, #CSR>
    sparse_tensor.print %result_mask_comp : tensor<4x4xf32, #CSR>


    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
}