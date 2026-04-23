//  To check if folding works:
//  ./build/bin/gblas-opt test_gblas_dims.mlir --canonicalize

//  To check if lowering works:
//  ./build/bin/gblas-opt test_gblas_dims.mlir --convert-gblas-to-linalg


func.func @test_fold_static(%arg0: tensor<100x200xf32>) -> (index, index) {
  // CHECK-FOLD-LABEL: func @test_fold_static
  // CHECK-FOLD-NOT: gblas.nrows
  // CHECK-FOLD-NOT: gblas.ncols
  // CHECK-FOLD-DAG: %[[C100:.*]] = arith.constant 100 : index
  // CHECK-FOLD-DAG: %[[C200:.*]] = arith.constant 200 : index
  // CHECK-FOLD: return %[[C100]], %[[C200]]
  
  %rows = gblas.nrows %arg0 : tensor<100x200xf32> -> index
  %cols = gblas.ncols %arg0 : tensor<100x200xf32> -> index
  return %rows, %cols : index, index
}


func.func @test_lower_dynamic(%arg0: tensor<?x?xf32, #sparse_tensor.encoding<{map = (d0, d1) -> (d0 : compressed, d1 : compressed)}>>) -> (index, index) {
  // CHECK-LOWER-LABEL: func @test_lower_dynamic
  // CHECK-LOWER-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-LOWER-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-LOWER: %[[R:.*]] = tensor.dim %arg0, %[[C0]]
  // CHECK-LOWER: %[[C:.*]] = tensor.dim %arg0, %[[C1]]
  // CHECK-LOWER: return %[[R]], %[[C]]
  
  %rows = gblas.nrows %arg0 : tensor<?x?xf32, #sparse_tensor.encoding<{map = (d0, d1) -> (d0 : compressed, d1 : compressed)}>> -> index
  %cols = gblas.ncols %arg0 : tensor<?x?xf32, #sparse_tensor.encoding<{map = (d0, d1) -> (d0 : compressed, d1 : compressed)}>> -> index
  return %rows, %cols : index, index
}


func.func @test_unranked(%arg0: tensor<*xf32>) -> (index, index) {
  // CHECK-UNRANKED-LABEL: func @test_unranked
  // CHECK-UNRANKED: %[[RANK:.*]] = tensor.rank %arg0
  // CHECK-UNRANKED: %[[IS_VALID:.*]] = arith.cmpi ugt, %[[RANK]], %c1
  // CHECK-UNRANKED: %[[DIM:.*]] = tensor.dim %arg0, %c1
  // CHECK-UNRANKED: %[[RES:.*]] = arith.select %[[IS_VALID]], %[[DIM]], %c0
  
  %cols = gblas.ncols %arg0 : tensor<*xf32> -> index
  %rows = gblas.nrows %arg0 : tensor<*xf32> -> index
  return %rows, %cols : index, index
}