// RUN: ./tools/gblas-opt %s --convert-gblas-to-linalg --linalg-generalize-named-ops --sparsification-and-bufferization

#CSR = #sparse_tensor.encoding<{ 
  map = (d0, d1) -> (d0 : dense, d1 : compressed) 
}>

module {
  func.func @test_plus(%input: tensor<4x4xf32, #CSR>, %output: tensor<4x4xf32, #CSR>) -> tensor<4x4xf32, #CSR> {
    // This will use the "plus" logic in your C++ pattern
    %res = gblas.update %input -> %output {accumulate_operator = "plus"} : 
           tensor<4x4xf32, #CSR> -> tensor<4x4xf32, #CSR>
    return %res : tensor<4x4xf32, #CSR>
  }
}