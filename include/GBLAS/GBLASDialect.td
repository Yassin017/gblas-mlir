#ifndef GBLAS_DIALECT
#define GBLAS_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

def GBLAS_Dialect : Dialect {
  let name = "gblas";
  let cppNamespace = "::mlir::gblas";
  let hasConstantMaterializer = 1;
  let summary = "GraphBLAS MLIR Dialect (mlir-graphblas API)";
}

// Unified Enum for Semiring components (Combine and Reduce)
def GBLAS_BinaryOp_Plus : I32EnumAttrCase<"plus", 0>;
def GBLAS_BinaryOp_Multiplies : I32EnumAttrCase<"multiplies", 1>;
def GBLAS_BinaryOp_Min : I32EnumAttrCase<"min", 2>;
def GBLAS_BinaryOp_Max : I32EnumAttrCase<"max", 3>;
def GBLAS_BinaryOp_First : I32EnumAttrCase<"first", 4>;
def GBLAS_BinaryOp_Second : I32EnumAttrCase<"second", 5>;
def GBLAS_BinaryOp_LogicalAnd : I32EnumAttrCase<"land", 6>;
def GBLAS_BinaryOp_LogicalOr  : I32EnumAttrCase<"lor", 7>;
def GBLAS_BinaryOp_Div : I32EnumAttrCase<"div", 8>;
def GBLAS_BinaryOp_Minus : I32EnumAttrCase<"minus", 9>;

def GBLAS_BinaryOp : I32EnumAttr<"BinaryOp", "GraphBLAS Binary Operator",
  [GBLAS_BinaryOp_Plus, GBLAS_BinaryOp_Multiplies, GBLAS_BinaryOp_Min,
   GBLAS_BinaryOp_Max, GBLAS_BinaryOp_First, GBLAS_BinaryOp_Second,
   GBLAS_BinaryOp_LogicalAnd, GBLAS_BinaryOp_LogicalOr, GBLAS_BinaryOp_Div, GBLAS_BinaryOp_Minus]> {
  let cppNamespace = "::mlir::gblas";
}

#endif // GBLAS_DIALECT