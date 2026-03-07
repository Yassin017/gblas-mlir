#ifndef GBLAS_OPS
#define GBLAS_OPS

include "GBLAS/GBLASDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class GBLAS_Op<string mnemonic, list<Trait> traits = []> :
    Op<GBLAS_Dialect, mnemonic, traits>;

// Milestone 1: The unified multiply operation
def GBLAS_MultiplyOp : GBLAS_Op<"multiply", [Pure]> {
  let summary = "GraphBLAS generalized multiply operation";
  let description = [{
    A generalized multiply operation that handles matrix-matrix, 
    matrix-vector, and vector-matrix multiplications using specific 
    reduce and combine semiring operators.
  }];

  let arguments = (ins
    AnyTensor:$lhs,
    AnyTensor:$rhs,
    GBLAS_BinaryOp:$combine_op,
    GBLAS_BinaryOp:$reduce_op
  );

  let results = (outs AnyTensor:$result);

  // Declarative assembly format for a clean MLIR syntax
  let assemblyFormat = [{
    $lhs `,` $rhs `combine` `=` $combine_op `reduce` `=` $reduce_op attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

#endif // GBLAS_OPS
