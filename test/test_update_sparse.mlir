#CSR = #sparse_tensor.encoding<{ 
  map = (d0, d1) -> (d0 : dense, d1 : compressed) 
}>

module {
  // The Driver Function
  func.func @main() -> i32 {
    // 1. Create a dense "Incoming" matrix (e.g., diagonal values)
    %dense_in = arith.constant dense<[
      [1.0, 0.0, 0.0, 0.0], 
      [0.0, 2.0, 0.0, 0.0], 
      [0.0, 0.0, 3.0, 0.0], 
      [0.0, 0.0, 0.0, 4.0]
    ]> : tensor<4x4xf32>
    
    // 2. Create a dense "Existing State" matrix
    %dense_out = arith.constant dense<[
      [0.0, 1.0, 0.0, 0.0], 
      [0.0, 2.0, 0.0, 0.0], 
      [0.0, 0.0, 0.0, 0.0], 
      [4.0, 0.0, 0.0, 1.0]
    ]> : tensor<4x4xf32>

    // 3. Convert them to #CSR format
    %sparse_in = sparse_tensor.convert %dense_in : tensor<4x4xf32> to tensor<4x4xf32, #CSR>
    %sparse_out = sparse_tensor.convert %dense_out : tensor<4x4xf32> to tensor<4x4xf32, #CSR>

    // 4. Call your gblas.update operation directly right here!
    %result = gblas.update %sparse_in -> %sparse_out {accumulate_operator = "plus"} : tensor<4x4xf32, #CSR> -> tensor<4x4xf32, #CSR>

    // 5. Print the resulting sparse tensor to the terminal
    sparse_tensor.print %result : tensor<4x4xf32, #CSR>

    // 6. Return 0 for OS exit code
    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
}