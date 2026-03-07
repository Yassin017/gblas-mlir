#ifndef GBLAS_DIALECT
#define GBLAS_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

def GBLAS_Dialect : Dialect {
  let name = "gblas";
  let cppNamespace = "::mlir::gblas";
  let summary = "GraphBLAS MLIR Dialect";
  let description = [{
    An out-of-tree MLIR dialect for representing GraphBLAS operations.
  }];
}

// Milestone 1: Define the Binary Operator Enum for Combine and Reduce
def GBLAS_BinaryOp_Plus : I32EnumAttrCase<"plus", 0>;
def GBLAS_BinaryOp_Multiplies : I32EnumAttrCase<"multiplies", 1>;
def GBLAS_BinaryOp_Minus : I32EnumAttrCase<"minus", 2>;
def GBLAS_BinaryOp_Divides : I32EnumAttrCase<"divides", 3>;
def GBLAS_BinaryOp_Max : I32EnumAttrCase<"max", 4>;
def GBLAS_BinaryOp_Min : I32EnumAttrCase<"min", 5>;

def GBLAS_BinaryOp : I32EnumAttr<"BinaryOp", "GraphBLAS Binary Operations",
  [GBLAS_BinaryOp_Plus, GBLAS_BinaryOp_Multiplies, GBLAS_BinaryOp_Minus,
   GBLAS_BinaryOp_Divides, GBLAS_BinaryOp_Max, GBLAS_BinaryOp_Min]> {
  let cppNamespace = "::mlir::gblas";
}

#endif // GBLAS_DIALECT
