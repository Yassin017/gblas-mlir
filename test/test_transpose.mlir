#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
#CSC = #sparse_tensor.encoding<{ map = (d0, d1) -> (d1 : dense, d0 : compressed) }>

func.func @main() {
  // A = [0.0  2.0]
  //     [3.0  0.0]
  %idx = arith.constant dense<[[0, 1], [1, 0]]> : tensor<2x2xi32>
  %val = arith.constant dense<[2.0, 3.0]> : tensor<2xf32>
  
  %A = gblas.from_coo %idx, %val : tensor<2x2xi32>, tensor<2xf32> -> tensor<2x2xf32, #CSR>

  // Transpose: Expected output is a 2x2 where A[0,1] becomes A[1,0]
  %AT = gblas.transpose %A : tensor<2x2xf32, #CSR> to tensor<2x2xf32, #CSC>

  sparse_tensor.print %AT : tensor<2x2xf32, #CSC>
  return
}