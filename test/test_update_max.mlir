#CSR = #sparse_tensor.encoding<{ 
  map = (d0, d1) -> (d0 : dense, d1 : compressed) 
}>

module {
  func.func @main() -> i32 {
    %dense_in = arith.constant dense<[
      [1.0, 0.0, 0.0, 0.0], 
      [0.0, 2.0, 0.0, 0.0], 
      [0.0, 0.0, 3.0, 0.0], 
      [0.0, 0.0, 0.0, 4.0]
    ]> : tensor<4x4xf32>
    
    %dense_out = arith.constant dense<[
      [0.0, 1.0, 0.0, 0.0], 
      [0.0, 2.0, 0.0, 0.0], 
      [0.0, 0.0, 0.0, 0.0], 
      [4.0, 0.0, 0.0, 1.0]
    ]> : tensor<4x4xf32>

    %sparse_in = sparse_tensor.convert %dense_in : tensor<4x4xf32> to tensor<4x4xf32, #CSR>
    %sparse_out = sparse_tensor.convert %dense_out : tensor<4x4xf32> to tensor<4x4xf32, #CSR>

    // Testing the "max" operator
    %result = gblas.update %sparse_in -> %sparse_out {accumulate_operator = "max"} : tensor<4x4xf32, #CSR> -> tensor<4x4xf32, #CSR>

    sparse_tensor.print %result : tensor<4x4xf32, #CSR>

    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
}