#ifndef GBLAS_OPS
#define GBLAS_OPS

include "GBLAS/GBLASDialect.td"

include "mlir/Interfaces/SideEffectInterfaces.td"

class GBLAS_Op<string mnemonic, list<Trait> traits = []> :
    Op<GBLAS_Dialect, mnemonic, traits>;

// 1. Build Operation (from_coo)
def GBLAS_FromCooOp : GBLAS_Op<"from_coo", [Pure]> {
  let summary = "Builds a GraphBLAS matrix from COO (Coordinate) format";
  let arguments = (ins
    AnyTensor:$rows,
    AnyTensor:$cols,
    AnyTensor:$vals
  );
  let results = (outs AnyTensor:$result);

  let assemblyFormat = [{
    $rows `,` $cols `,` $vals attr-dict `:` type($rows) `,` type($cols) `,` type($vals) `->` type($result)
  }];
}
def GBLAS_MxmOp : GBLAS_Op<"mxm", [Pure]> {
  let summary = "GraphBLAS matrix-matrix multiply";
  let description = [{
    Performs a matrix-matrix multiplication using the specified semiring.
    Both inputs must be 2D tensors.
  }];
  let arguments = (ins
    AnyTensor:$a,
    AnyTensor:$b,
    GBLAS_BinaryOp:$combine_op,
    GBLAS_BinaryOp:$reduce_op
  );
  let results = (outs AnyTensor:$result);

  let assemblyFormat = [{
    $a `,` $b `combine` `=` $combine_op `reduce` `=` $reduce_op attr-dict `:` type($a) `,` type($b) `->` type($result)
  }];
}

def GBLAS_EWiseAddOp : GBLAS_Op<"ewise_add"> {
    let summary = "GraphBLAS element-wise addition (union)";
    let description = [{
        Returns a tensor containing the union of the indices of the input tensors.
        If an index exists in both, the `apply` operator is used to combine them.
    }];

    let arguments = (ins 
        AnyTensor:$lhs, 
        AnyTensor:$rhs,
        StrAttr:$op_name // e.g., "plus"
    );

    let results = (outs AnyTensor:$result);

    let hasVerifier = 1;

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

#endif // GBLAS_OPS