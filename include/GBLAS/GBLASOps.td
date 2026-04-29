#ifndef GBLAS_OPS
#define GBLAS_OPS

include "GBLAS/GBLASDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class GBLAS_Op<string mnemonic, list<Trait> traits = []> :
    Op<GBLAS_Dialect, mnemonic, traits>;

def GBLAS_FromCooOp : GBLAS_Op<"from_coo", [Pure]> {
  let summary = "Builds a GraphBLAS tensor from coordinate tensors";
  let description = [{
    Builds a new sparse tensor using a 2D dense tensor representing the
    indices as coordinates and a 1D tensor of associated values.
    Returns either a 1D vector or a 2D matrix based on the output type.
  }];
  let arguments = (ins 
    2DTensorOf<[AnyInteger]>:$indices, 
    1DTensorOf<[AnyType]>:$values
  );
  let results = (outs AnyTensor:$result);
  let assemblyFormat = [{
    $indices `,` $values attr-dict `:` type($indices) `,` type($values) `->` type($result)
  }];
}

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

def GBLAS_VxmOp : GBLAS_Op<"vxm", [Pure]> {
  let summary = "GraphBLAS vector-matrix multiply";
  let arguments = (ins
    1DTensorOf<[AnyType]>:$vector,
    2DTensorOf<[AnyType]>:$matrix,
    1DTensorOf<[AnyType]>:$outs,    // <-- 1. Add outs here
    Optional<1DTensorOf<[AnyType]>>:$mask,
    GBLAS_BinaryOp:$combine_op,
    GBLAS_BinaryOp:$reduce_op,
    DefaultValuedAttr<BoolAttr, "false">:$mask_complement
  );
  let results = (outs 1DTensorOf<[AnyType]>:$result);
  
  // 2. Add `outs` `(` $outs `)` to the assembly format and `type($outs)`
  let assemblyFormat = [{
    $vector `,` $matrix `outs` `(` $outs `)` (`,` $mask^)? `combine` `=` $combine_op `reduce` `=` $reduce_op attr-dict `:` type($vector) `,` type($matrix) `,` type($outs) (`,` type($mask)^)? `->` type($result)
  }];
}

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

def GBLAS_UpdateOp : GBLAS_Op<"update", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    AllTypesMatch<["output", "res"]> // This tells MLIR res and output have the same type
]> {
    let summary = "In-place update of a sparse tensor or vector.";
    let description = [{
        Updates the '$output' tensor using '$input'. 
        If '$mask' is present, it restricts the update.
    }];

    let arguments = (ins
        AnyTensor:$input,
        AnyTensor:$output,
        Optional<AnyTensor>:$mask,
        OptionalAttr<StrAttr>:$accumulate_operator,
        DefaultValuedAttr<BoolAttr, "false">:$replace,
        DefaultValuedAttr<BoolAttr, "false">:$mask_complement
    );

    let results = (outs AnyTensor:$res); 

    let assemblyFormat = [{
        $input `->` $output (`(` $mask^ `)`)? attr-dict 
        `:` type($input) `->` type($output) (`(` type($mask)^ `)`)?
    }];

    let hasVerifier = 1;
}


def GBLAS_IntersectOp : GBLAS_Op<"intersect", [Pure]> {
  let summary = "Element-wise intersection of two sparse tensors";
  let description = [{
    Computes the set intersection of two sparse tensors using a specified binary operator.
  }];

  let arguments = (ins
    AnyTensor:$a,
    AnyTensor:$b,
    Optional<AnyTensor>:$mask,
    GBLAS_BinaryOp:$intersect_operator,
    DefaultValuedAttr<BoolAttr, "false">:$mask_complement
  );

  let results = (outs AnyTensor:$result);

  let assemblyFormat = [{
    $intersect_operator operands attr-dict `:` functional-type(operands, results)
  }];
}


def GBLAS_TransposeOp : GBLAS_Op<"transpose", [Pure]> {
  let summary = "Transpose operation.";
  let description = [{
    Returns a new sparse matrix that's the transpose of the input matrix.
  }];

  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `to` type($output)
  }];
}

#endif // GBLAS_OPS