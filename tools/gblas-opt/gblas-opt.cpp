#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

// Include only the specific standard dialects we need
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

// Include our custom GraphBLAS dialect
#include "GBLAS/GBLASDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  
  // Register the specific MLIR standard dialects
  registry.insert<mlir::func::FuncDialect,
                  mlir::tensor::TensorDialect,
                  mlir::linalg::LinalgDialect,
                  mlir::arith::ArithDialect,
                  mlir::scf::SCFDialect>();
  
  // Register OUR custom GraphBLAS dialect
  registry.insert<mlir::gblas::GBLASDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "GraphBLAS MLIR optimizer driver\n", registry));
}
