#ifndef GBLAS_PASSES
#define GBLAS_PASSES

include "mlir/Pass/PassBase.td"

def ConvertGBLASToLinalg : Pass<"convert-gblas-to-linalg", "::mlir::ModuleOp"> {
  let summary = "Lower GraphBLAS operations to the Linalg and SCF dialects";
  let description = [{
    Converts gblas.mxm to linalg.matmul and gblas.from_coo to scf.for loops.
  }];
  let constructor = "mlir::gblas::createConvertGBLASToLinalgPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::sparse_tensor::SparseTensorDialect"
  ];
}

#endif // GBLAS_PASSES