#ifndef GBLAS_OPS
#define GBLAS_OPS

include "GBLAS/GBLASDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class GBLAS_Op<string mnemonic, list<Trait> traits = []> :
    Op<GBLAS_Dialect, mnemonic, traits>;

// 1. Build Operation (from_coo)
def GBLAS_FromCooOp : GBLAS_Op<"from_coo", [Pure]> {
  let summary = "Builds a GraphBLAS matrix from COO format";
  let arguments = (ins AnyTensor:$rows, AnyTensor:$cols, AnyTensor:$vals);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = [{
    $rows `,` $cols `,` $vals attr-dict `:` type($rows) `,` type($cols) `,` type($vals) `->` type($result)
  }];
}

// 2. Matrix-Matrix Multiply (mxm)
def GBLAS_MxmOp : GBLAS_Op<"mxm", [Pure]> {
  let summary = "GraphBLAS matrix-matrix multiply";
  let arguments = (ins
    2DTensorOf<[AnyType]>:$a,
    2DTensorOf<[AnyType]>:$b,
    Optional<2DTensorOf<[AnyType]>>:$mask,
    GBLAS_BinaryOp:$combine_op,
    GBLAS_BinaryOp:$reduce_op,
    DefaultValuedAttr<BoolAttr, "false">:$mask_complement
  );
  let results = (outs 2DTensorOf<[AnyType]>:$result);
  let assemblyFormat = [{
    $a `,` $b (`,` $mask^)? `combine` `=` $combine_op `reduce` `=` $reduce_op attr-dict `:` type($a) `,` type($b) (`,` type($mask)^)? `->` type($result)
  }];
}

// 3. Matrix-Vector Multiply (mxv)
def GBLAS_MxvOp : GBLAS_Op<"mxv", [Pure]> {
  let summary = "GraphBLAS matrix-vector multiply";
  let arguments = (ins
    2DTensorOf<[AnyType]>:$matrix,
    1DTensorOf<[AnyType]>:$vector,
    Optional<1DTensorOf<[AnyType]>>:$mask,
    GBLAS_BinaryOp:$combine_op,
    GBLAS_BinaryOp:$reduce_op,
    DefaultValuedAttr<BoolAttr, "false">:$mask_complement
  );
  let results = (outs 1DTensorOf<[AnyType]>:$result);
  let assemblyFormat = [{
    $matrix `,` $vector (`,` $mask^)? `combine` `=` $combine_op `reduce` `=` $reduce_op attr-dict `:` type($matrix) `,` type($vector) (`,` type($mask)^)? `->` type($result)
  }];
}

// 4. Vector-Matrix Multiply (vxm)
def GBLAS_VxmOp : GBLAS_Op<"vxm", [Pure]> {
  let summary = "GraphBLAS vector-matrix multiply";
  let arguments = (ins
    1DTensorOf<[AnyType]>:$vector,
    2DTensorOf<[AnyType]>:$matrix,
    Optional<1DTensorOf<[AnyType]>>:$mask,
    GBLAS_BinaryOp:$combine_op,
    GBLAS_BinaryOp:$reduce_op,
    DefaultValuedAttr<BoolAttr, "false">:$mask_complement
  );
  let results = (outs 1DTensorOf<[AnyType]>:$result);
  let assemblyFormat = [{
    $vector `,` $matrix (`,` $mask^)? `combine` `=` $combine_op `reduce` `=` $reduce_op attr-dict `:` type($vector) `,` type($matrix) (`,` type($mask)^)? `->` type($result)
  }];
}

// 5. Vector-Vector Multiply (vxv)
def GBLAS_VxvOp : GBLAS_Op<"vxv", [Pure]> {
  let summary = "GraphBLAS vector-vector multiply (Dot Product)";
  let arguments = (ins
    1DTensorOf<[AnyType]>:$a,
    1DTensorOf<[AnyType]>:$b,
    GBLAS_BinaryOp:$combine_op,
    GBLAS_BinaryOp:$reduce_op
  );
  let results = (outs 0DTensorOf<[AnyType]>:$result); // Scalar output
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

def GBLAS_NRowsOp : GBLAS_Op<"nrows", [Pure]> {
    let summary = "Returns the number of rows.";
    let arguments = (ins AnyTensor:$input);
    let results = (outs Index:$res);
    
    // Manual flag to enable folding in C++
    let hasFolder = 1;
    
    let assemblyFormat = "$input attr-dict `:` type($input) `->` type($res)";
}

def GBLAS_NColsOp : GBLAS_Op<"ncols", [Pure]> {
    let summary = "Returns the number of columns.";
    let arguments = (ins AnyTensor:$input);
    let results = (outs Index:$res);
    
    // Manual flag to enable folding in C++
    let hasFolder = 1;
    
    let assemblyFormat = "$input attr-dict `:` type($input) `->` type($res)";
}

#endif // GBLAS_OPS