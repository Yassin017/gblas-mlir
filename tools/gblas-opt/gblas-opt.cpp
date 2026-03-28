
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SparseTensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"

// Include only the specific standard dialects we need
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h" 

#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/Passes.h" 

// Include our custom GraphBLAS dialect AND Passes
#include "GBLAS/GBLASDialect.h"
#include "GBLAS/GBLASPasses.h" 

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  
  // Register the specific MLIR standard dialects
  registry.insert<mlir::func::FuncDialect,
                  mlir::tensor::TensorDialect,
                  mlir::linalg::LinalgDialect,
                  mlir::arith::ArithDialect,
                  mlir::scf::SCFDialect,
                  mlir::sparse_tensor::SparseTensorDialect>();
  
  // Register OUR custom GraphBLAS dialect
  registry.insert<mlir::gblas::GBLASDialect>();

  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::sparse_tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
  
  // --- THE FIX ---
  // Register all passes defined in GBLASPasses.td so the 
  // command-line tool recognizes the --convert-gblas-to-linalg flag
  mlir::gblas::registerGBLASPasses();

  mlir::registerSparseTensorPasses(); 

  mlir::registerLinalgPasses(); 

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "GraphBLAS MLIR optimizer driver\n", registry));
}

// int main(int argc, char **argv) {
//   mlir::DialectRegistry registry;
  
//   registry.insert<mlir::func::FuncDialect,
//                   mlir::tensor::TensorDialect,
//                   mlir::linalg::LinalgDialect,
//                   mlir::arith::ArithDialect,
//                   mlir::scf::SCFDialect,
//                   mlir::sparse_tensor::SparseTensorDialect>();

  

//   // Register your custom passes
//   gblas::registerGBLASPasses();

//   // Register MLIR's built-in Sparse Tensor passes!
//   mlir::registerSparseTensorPasses(); 

//   return mlir::asMainReturnCode(
//       mlir::MlirOptMain(argc, argv, "GBLAS optimizer driver\n", registry));
// }