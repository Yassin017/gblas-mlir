#include "GBLAS/GBLASDialect.h"
#include "GBLAS/GBLASPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"

#include "mlir/AsmParser/AsmParser.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"


// -- Generate Pass Implementation 
namespace mlir {
namespace gblas {
#define GEN_PASS_DEF_CONVERTGBLASTOLINALG
#include "GBLASPasses.h.inc"
} // namespace gblas
} // namespace mlir
// --

using namespace mlir;

namespace {

/** 
//  Lowering for gblas.from_coo 
struct FromCooOpLowering : public OpRewritePattern<gblas::FromCooOp> {
  using OpRewritePattern<gblas::FromCooOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gblas::FromCooOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());

    // // 1. Use the Parser to build the encoding! This is immune to C++ API changes.
    // // We define a COO format (compressed, singleton)
    // auto configAttr = mlir::parseAttribute(
    //     "#sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed, d1 : singleton) }>",
    //     rewriter.getContext());
    // auto config = cast<sparse_tensor::SparseTensorEncodingAttr>(configAttr);

    // // 2. Create the new Sparse Tensor Type
    // auto sparseResultType = RankedTensorType::get(
    //     resultType.getShape(), resultType.getElementType(), config);

    // Just get the exact return type the user asked for (which includes the #CSR encoding)
    auto sparseResultType = cast<RankedTensorType>(op.getResult().getType());

    // 3. FIXED: Use tensor::EmptyOp with the sparse type!
    Value sparseEmpty = rewriter.create<tensor::EmptyOp>(
        loc, sparseResultType, ValueRange{});

    // 4. The Loop
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value length = rewriter.create<tensor::DimOp>(loc, op.getRows(), c0);

    auto forOp = rewriter.create<scf::ForOp>(
        loc, c0, length, c1, ValueRange{sparseEmpty},
        [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
            Value rowIdx = b.create<tensor::ExtractOp>(loc, op.getRows(), iv);
            Value colIdx = b.create<tensor::ExtractOp>(loc, op.getCols(), iv);
            Value val = b.create<tensor::ExtractOp>(loc, op.getVals(), iv);

            Value row = b.create<arith::IndexCastOp>(loc, b.getIndexType(), rowIdx);
            Value col = b.create<arith::IndexCastOp>(loc, b.getIndexType(), colIdx);

            Value updatedSparse = b.create<tensor::InsertOp>(
                loc, val, iterArgs[0], ValueRange{row, col});
            
            b.create<scf::YieldOp>(loc, updatedSparse);
        });

    // 1. Create the load operation to finalize the sparse tensor.
    // The "true" boolean tells MLIR that we just inserted elements, 
    // so it needs to sort and compress the memory structure now.
    mlir::Value finalizedTensor = rewriter.create<mlir::sparse_tensor::LoadOp>(
        loc, 
        forOp.getResult(0),        // The output of your scf.for loop
        true
    );

    rewriter.replaceOp(op, finalizedTensor);
    return success();
  }

};

**/

/** 
// ============================================================================
// Lowering for gblas.from_coo
// ============================================================================
struct FromCooOpLowering : public OpRewritePattern<gblas::FromCooOp> {
  using OpRewritePattern<gblas::FromCooOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gblas::FromCooOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    int rank = resultType.getRank();
    
    // We only support generating 1D vectors or 2D matrices
    if (rank != 1 && rank != 2) return failure();

    // 1. Allocate and fill the output tensor with 0.0
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(resultType.getElementType()));
    Value filledTensor = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{emptyTensor}).getResult(0);

    // 2. Prepare loop bounds (0 to length of values array)
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value length = rewriter.create<tensor::DimOp>(loc, op.getValues(), 0);

    // 3. scf.for loop to insert values into the dense tensor
    auto forOp = rewriter.create<scf::ForOp>(
        loc, c0, length, c1, ValueRange{filledTensor},
        [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
            // Extract the value
            Value val = b.create<tensor::ExtractOp>(loc, op.getValues(), ValueRange{iv});
            
            Value updatedTensor;
            if (rank == 1) {
                // Vector: Extract idx from indices[iv, 0]
                Value idxRaw = b.create<tensor::ExtractOp>(loc, op.getIndices(), ValueRange{iv, c0});
                Value idx = b.create<arith::IndexCastOp>(loc, b.getIndexType(), idxRaw);
                updatedTensor = b.create<tensor::InsertOp>(loc, val, iterArgs[0], ValueRange{idx});
            } else {
                // Matrix: Extract row from indices[iv, 0], col from indices[iv, 1]
                Value rowRaw = b.create<tensor::ExtractOp>(loc, op.getIndices(), ValueRange{iv, c0});
                Value colRaw = b.create<tensor::ExtractOp>(loc, op.getIndices(), ValueRange{iv, c1});
                Value row = b.create<arith::IndexCastOp>(loc, b.getIndexType(), rowRaw);
                Value col = b.create<arith::IndexCastOp>(loc, b.getIndexType(), colRaw);
                updatedTensor = b.create<tensor::InsertOp>(loc, val, iterArgs[0], ValueRange{row, col});
            }
            b.create<scf::YieldOp>(loc, updatedTensor);
        });

    rewriter.replaceOp(op, forOp.getResult(0));
    return success();
  }
};

**/

/**

struct FromCooOpLowering : public OpRewritePattern<gblas::FromCooOp> {
  using OpRewritePattern<gblas::FromCooOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gblas::FromCooOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    int rank = resultType.getRank();
    
    if (rank != 1 && rank != 2) return failure();

    // 1. Check if the output is meant to be sparse
    bool isSparse = sparse_tensor::getSparseTensorEncoding(resultType) != nullptr;

    // 2. Initialize the tensor
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
    Value initTensor = emptyTensor;

    if (!isSparse) {
        // Only dense tensors need explicit filling
        Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(resultType.getElementType()));
        initTensor = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{emptyTensor}).getResult(0);
    }

    // 3. Prepare loop bounds
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value length = rewriter.create<tensor::DimOp>(loc, op.getValues(), 0);

    // 4. scf.for loop to insert values
    auto forOp = rewriter.create<scf::ForOp>(
        loc, c0, length, c1, ValueRange{initTensor},
        [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
            Value val = b.create<tensor::ExtractOp>(loc, op.getValues(), ValueRange{iv});
            Value updatedTensor;
            if (rank == 1) {
                Value idxRaw = b.create<tensor::ExtractOp>(loc, op.getIndices(), ValueRange{iv, c0});
                Value idx = b.create<arith::IndexCastOp>(loc, b.getIndexType(), idxRaw);
                updatedTensor = b.create<tensor::InsertOp>(loc, val, iterArgs[0], ValueRange{idx});
            } else {
                Value rowRaw = b.create<tensor::ExtractOp>(loc, op.getIndices(), ValueRange{iv, c0});
                Value colRaw = b.create<tensor::ExtractOp>(loc, op.getIndices(), ValueRange{iv, c1});
                Value row = b.create<arith::IndexCastOp>(loc, b.getIndexType(), rowRaw);
                Value col = b.create<arith::IndexCastOp>(loc, b.getIndexType(), colRaw);
                updatedTensor = b.create<tensor::InsertOp>(loc, val, iterArgs[0], ValueRange{row, col});
            }
            b.create<scf::YieldOp>(loc, updatedTensor);
        });

    Value finalResult = forOp.getResult(0);

   // 5. If Sparse, we must finalize the inserts to compress the data structure
    if (isSparse) {
        finalResult = rewriter.create<sparse_tensor::LoadOp>(loc, finalResult, rewriter.getUnitAttr());
    }

    rewriter.replaceOp(op, finalResult);
    return success();
  }
};

**/

// ============================================================================
// Lowering for gblas.from_coo (Modern Sparse-Compatible)
// ============================================================================
struct FromCooOpLowering : public OpRewritePattern<gblas::FromCooOp> {
  using OpRewritePattern<gblas::FromCooOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gblas::FromCooOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    int rank = resultType.getRank();
    
    if (rank != 1 && rank != 2) return failure();

    // 1. Determine if the requested output is sparse
    bool isSparse = sparse_tensor::getSparseTensorEncoding(resultType) != nullptr;

    // 2. Always create a DENSE equivalent type first as our assembly workspace
    auto denseType = RankedTensorType::get(resultType.getShape(), resultType.getElementType());
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, denseType.getShape(), denseType.getElementType());
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(denseType.getElementType()));
    Value initTensor = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{emptyTensor}).getResult(0);

    // 3. Prepare loop bounds
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value length = rewriter.create<tensor::DimOp>(loc, op.getValues(), 0);

    // 4. scf.for loop to insert values into the DENSE workspace
    auto forOp = rewriter.create<scf::ForOp>(
        loc, c0, length, c1, ValueRange{initTensor},
        [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
            Value val = b.create<tensor::ExtractOp>(loc, op.getValues(), ValueRange{iv});
            Value updatedTensor;
            if (rank == 1) {
                Value idxRaw = b.create<tensor::ExtractOp>(loc, op.getIndices(), ValueRange{iv, c0});
                Value idx = b.create<arith::IndexCastOp>(loc, b.getIndexType(), idxRaw);
                updatedTensor = b.create<tensor::InsertOp>(loc, val, iterArgs[0], ValueRange{idx});
            } else {
                Value rowRaw = b.create<tensor::ExtractOp>(loc, op.getIndices(), ValueRange{iv, c0});
                Value colRaw = b.create<tensor::ExtractOp>(loc, op.getIndices(), ValueRange{iv, c1});
                Value row = b.create<arith::IndexCastOp>(loc, b.getIndexType(), rowRaw);
                Value col = b.create<arith::IndexCastOp>(loc, b.getIndexType(), colRaw);
                updatedTensor = b.create<tensor::InsertOp>(loc, val, iterArgs[0], ValueRange{row, col});
            }
            b.create<scf::YieldOp>(loc, updatedTensor);
        });

    Value finalResult = forOp.getResult(0);

    // 5. THE MAGIC STEP: Convert the dense workspace to the requested Sparse Tensor.
    // This triggers the MLIR Sparse Compiler to natively run the exact algorithm 
    // you found in graphblas-mlir (Count, CumSum, Scatter) highly optimized in C.
    if (isSparse) {
        finalResult = rewriter.create<sparse_tensor::ConvertOp>(loc, resultType, finalResult);
    }

    rewriter.replaceOp(op, finalResult);
    return success();
  }
};



// // Lowering for gblas.mxm 
// struct MxmOpLowering : public OpRewritePattern<gblas::MxmOp> {
//   using OpRewritePattern<gblas::MxmOp>::OpRewritePattern;

//   LogicalResult matchAndRewrite(gblas::MxmOp op, PatternRewriter &rewriter) const override {
    
//     Location loc = op.getLoc();
//     auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
//     if (!resultType) return failure();

//     if (op.getCombineOp() != gblas::BinaryOp::multiplies || 
//         op.getReduceOp() != gblas::BinaryOp::plus) {
//       return failure(); 
//     }

//     Value emptyTensor = rewriter.create<tensor::EmptyOp>( loc, resultType.getShape(), resultType.getElementType());
//     Value zero = rewriter.create<arith::ConstantOp>( loc, rewriter.getZeroAttr(resultType.getElementType()));
//     Value filledTensor = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{emptyTensor}).getResult(0);

//     Value matmulResult = rewriter.create<linalg::MatmulOp>( loc, resultType, ValueRange{op.getA(), op.getB()}, ValueRange{filledTensor}).getResult(0);

//     rewriter.replaceOp(op, matmulResult);
//     return success();
//   }
// };

// // Lowering for gblas.mxm (Updated to handle optional mask rejection)
// struct MxmOpLowering : public OpRewritePattern<gblas::MxmOp> {
//   using OpRewritePattern<gblas::MxmOp>::OpRewritePattern;
//   LogicalResult matchAndRewrite(gblas::MxmOp op, PatternRewriter &rewriter) const override {
//     Location loc = op.getLoc();
//     auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
//     if (!resultType) return failure();

//     // Reject if a mask is provided (for now, until we build mask support)
//     if (op.getMask()) return failure();

//     if (op.getCombineOp() != gblas::BinaryOp::multiplies || 
//         op.getReduceOp() != gblas::BinaryOp::plus) {
//       return failure();
//     }

//     Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
//     Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(resultType.getElementType()));
//     Value filledTensor = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{emptyTensor}).getResult(0);

//     Value matmulResult = rewriter.create<linalg::MatmulOp>(loc, resultType, ValueRange{op.getA(), op.getB()}, ValueRange{filledTensor}).getResult(0);

//     rewriter.replaceOp(op, matmulResult);
//     return success();
//   }
// };

// Helper function to map GraphBLAS Semirings to MLIR Arith operations
static Value buildSemiringOperation(OpBuilder &builder, Location loc, gblas::BinaryOp opType, Value lhs, Value rhs) {
    // In a production compiler, you'd check the Type and emit AddI/MulI for integers.
    switch (opType) {
        case gblas::BinaryOp::plus:       return builder.create<arith::AddFOp>(loc, lhs, rhs);
        case gblas::BinaryOp::multiplies: return builder.create<arith::MulFOp>(loc, lhs, rhs);
        case gblas::BinaryOp::min:        return builder.create<arith::MinimumFOp>(loc, lhs, rhs);
        case gblas::BinaryOp::max:        return builder.create<arith::MaximumFOp>(loc, lhs, rhs);
        // Defaulting to plus if an unhandled enum is passed
        default:                          return builder.create<arith::AddFOp>(loc, lhs, rhs);
    }
}
// Common helper to get the initialized output tensor (Empty for Sparse, Zero-Filled for Dense)
static Value getInitializedOutputTensor(PatternRewriter &rewriter, Location loc, RankedTensorType resultType) {
    // 1. Create an empty tensor that perfectly preserves the type (including sparse encoding)
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, resultType, ValueRange{});
    
    // 2. Check if it has a sparse encoding
    if (sparse_tensor::getSparseTensorEncoding(resultType)) {
        // Sparse tensors are intrinsically empty/zero-filled upon creation.
        // We do NOT want to run linalg::FillOp on them.
        return emptyTensor;
    } else {
        // Dense tensors allocate dirty memory, so we must explicitly fill with zeros.
        Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(resultType.getElementType()));
        return rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{emptyTensor}).getResult(0);
    }
}

// ============================================================================
// Updated Lowerings
// ============================================================================

struct VxmOpLowering : public OpRewritePattern<gblas::VxmOp> {
  using OpRewritePattern<gblas::VxmOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(gblas::VxmOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());

    Value outTensor = getInitializedOutputTensor(rewriter, loc, resultType);

    MLIRContext *context = rewriter.getContext();
    AffineExpr n, k; bindDims(context, n, k);

    SmallVector<Value> inputs = {op.getVector(), op.getMatrix()};
    SmallVector<AffineMap> indexingMaps = { AffineMap::get(2, 0, {k}, context), AffineMap::get(2, 0, {k, n}, context) };

    if (op.getMask()) {
        inputs.push_back(op.getMask());
        indexingMaps.push_back(AffineMap::get(2, 0, {n}, context));
    }
    indexingMaps.push_back(AffineMap::get(2, 0, {n}, context)); // Output mapping

    SmallVector<utils::IteratorType> iteratorTypes = {utils::IteratorType::parallel, utils::IteratorType::reduction};

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, inputs, ValueRange{outTensor}, indexingMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
            Value combined = buildSemiringOperation(b, loc, op.getCombineOp(), args[0], args[1]);
            Value out_val = op.getMask() ? args[3] : args[2];
            Value reduced = buildSemiringOperation(b, loc, op.getReduceOp(), combined, out_val);

            if (op.getMask()) {
                Value mask_cond = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UGT, args[2], b.create<arith::ConstantOp>(loc, b.getFloatAttr(b.getF32Type(), 0.0)));
                if (op.getMaskComplement()) mask_cond = b.create<arith::XOrIOp>(loc, mask_cond, b.create<arith::ConstantIntOp>(loc, 1, 1));
                b.create<linalg::YieldOp>(loc, b.create<arith::SelectOp>(loc, mask_cond, reduced, out_val).getResult());
            } else { b.create<linalg::YieldOp>(loc, reduced); }
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

struct MxvOpLowering : public OpRewritePattern<gblas::MxvOp> {
  using OpRewritePattern<gblas::MxvOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(gblas::MxvOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    
    Value outTensor = getInitializedOutputTensor(rewriter, loc, resultType);

    MLIRContext *context = rewriter.getContext();
    AffineExpr m, k; bindDims(context, m, k);

    SmallVector<Value> inputs = {op.getMatrix(), op.getVector()};
    SmallVector<AffineMap> indexingMaps = {AffineMap::get(2, 0, {m, k}, context), AffineMap::get(2, 0, {k}, context)};
    
    if (op.getMask()) {
        inputs.push_back(op.getMask());
        indexingMaps.push_back(AffineMap::get(2, 0, {m}, context));
    }
    indexingMaps.push_back(AffineMap::get(2, 0, {m}, context));

    SmallVector<utils::IteratorType> iteratorTypes = {utils::IteratorType::parallel, utils::IteratorType::reduction};

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, inputs, ValueRange{outTensor}, indexingMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
            Value combined = buildSemiringOperation(b, loc, op.getCombineOp(), args[0], args[1]);
            Value out_val = op.getMask() ? args[3] : args[2];
            Value reduced = buildSemiringOperation(b, loc, op.getReduceOp(), combined, out_val);
            if (op.getMask()) {
                Value mask_cond = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UGT, args[2], b.create<arith::ConstantOp>(loc, b.getFloatAttr(b.getF32Type(), 0.0)));
                if (op.getMaskComplement()) mask_cond = b.create<arith::XOrIOp>(loc, mask_cond, b.create<arith::ConstantIntOp>(loc, 1, 1));
                b.create<linalg::YieldOp>(loc, b.create<arith::SelectOp>(loc, mask_cond, reduced, out_val).getResult());
            } else { b.create<linalg::YieldOp>(loc, reduced); }
        });
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

// ============================================================================
// Fixed MxmOpLowering (Uses getA() and getB())
// ============================================================================
struct MxmOpLowering : public OpRewritePattern<gblas::MxmOp> {
  using OpRewritePattern<gblas::MxmOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(gblas::MxmOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    
    Value outTensor = getInitializedOutputTensor(rewriter, loc, resultType);

    MLIRContext *context = rewriter.getContext();
    AffineExpr m, n, k; bindDims(context, m, n, k);

    // FIXED: Use getA() and getB() instead of getLhs() / getRhs()
    SmallVector<Value> inputs = {op.getA(), op.getB()};
    SmallVector<AffineMap> indexingMaps = {AffineMap::get(3, 0, {m, k}, context), AffineMap::get(3, 0, {k, n}, context)};
    
    if (op.getMask()) {
        inputs.push_back(op.getMask());
        indexingMaps.push_back(AffineMap::get(3, 0, {m, n}, context));
    }
    indexingMaps.push_back(AffineMap::get(3, 0, {m, n}, context));

    SmallVector<utils::IteratorType> iteratorTypes = {utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction};

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, inputs, ValueRange{outTensor}, indexingMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
            Value combined = buildSemiringOperation(b, loc, op.getCombineOp(), args[0], args[1]);
            Value out_val = op.getMask() ? args[3] : args[2];
            Value reduced = buildSemiringOperation(b, loc, op.getReduceOp(), combined, out_val);
            if (op.getMask()) {
                Value mask_cond = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UGT, args[2], b.create<arith::ConstantOp>(loc, b.getFloatAttr(b.getF32Type(), 0.0)));
                if (op.getMaskComplement()) mask_cond = b.create<arith::XOrIOp>(loc, mask_cond, b.create<arith::ConstantIntOp>(loc, 1, 1));
                b.create<linalg::YieldOp>(loc, b.create<arith::SelectOp>(loc, mask_cond, reduced, out_val).getResult());
            } else { b.create<linalg::YieldOp>(loc, reduced); }
        });
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

// ============================================================================
// Fixed VxvOpLowering (Uses getA() / getB(), and removes invalid mask logic)
// ============================================================================
struct VxvOpLowering : public OpRewritePattern<gblas::VxvOp> {
  using OpRewritePattern<gblas::VxvOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(gblas::VxvOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    
    Value outTensor = getInitializedOutputTensor(rewriter, loc, resultType);

    MLIRContext *context = rewriter.getContext();
    AffineExpr k; bindDims(context, k);

    // FIXED: Use getA() and getB()
    SmallVector<Value> inputs = {op.getA(), op.getB()};
    
    // No mask logic here! VXV returns a scalar, so masking is not applicable.
    SmallVector<AffineMap> indexingMaps = {
        AffineMap::get(1, 0, {k}, context), 
        AffineMap::get(1, 0, {k}, context),
        AffineMap::get(1, 0, {}, context) // Output is a 0D scalar tensor
    };

    SmallVector<utils::IteratorType> iteratorTypes = {utils::IteratorType::reduction};

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, inputs, ValueRange{outTensor}, indexingMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
            // Apply the semiring correctly without worrying about mask condition checks
            Value combined = buildSemiringOperation(b, loc, op.getCombineOp(), args[0], args[1]);
            Value reduced = buildSemiringOperation(b, loc, op.getReduceOp(), combined, args[2]);
            b.create<linalg::YieldOp>(loc, reduced);
        });
        
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

struct EWiseAddLowering : public OpConversionPattern<gblas::EWiseAddOp> {
  using OpConversionPattern<gblas::EWiseAddOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gblas::EWiseAddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    
    auto loc = op.getLoc();
    auto resultType = llvm::cast<RankedTensorType>(op.getResult().getType());

    // Fix: Use bufferization::AllocTensorOp instead of sparse_tensor::AllocOp
    auto allocOp = rewriter.create<bufferization::AllocTensorOp>(
        loc, resultType, ValueRange{}, Value(), IntegerAttr());

    // Replace with linalg.add
    rewriter.replaceOpWithNewOp<linalg::AddOp>(
        op, 
        resultType, 
        ValueRange{adaptor.getLhs(), adaptor.getRhs()}, 
        ValueRange{allocOp.getResult()});

    return success();
  }
};

struct NRowsLowering : public OpConversionPattern<gblas::NRowsOp> {
  using OpConversionPattern<gblas::NRowsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(gblas::NRowsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = adaptor.getInput();
    Type type = input.getType();

    // 1. Existing Ranked Logic
    if (auto rankedType = llvm::dyn_cast<RankedTensorType>(type)) {
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      rewriter.replaceOpWithNewOp<tensor::DimOp>(op, input, zero);
      return success();
    }

    // 2. NEW Unranked Logic
    if (auto unrankedType = llvm::dyn_cast<UnrankedTensorType>(type)) {
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value rank = rewriter.create<tensor::RankOp>(loc, input);
      
      // Check: is rank > 0?
      Value isValid = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ugt, rank, zero);

      Value dimSize = rewriter.create<tensor::DimOp>(loc, input, zero);
      
      // If valid, return dim, else return 0
      rewriter.replaceOpWithNewOp<arith::SelectOp>(op, isValid, dimSize, zero);
      return success();
    }
    return failure();
  }
};

// --- Updated Pattern for NColsOp ---
struct NColsLowering : public OpConversionPattern<gblas::NColsOp> {
  using OpConversionPattern<gblas::NColsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(gblas::NColsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = adaptor.getInput();
    Type type = input.getType();

    if (auto rankedType = llvm::dyn_cast<RankedTensorType>(type)) {
      // Safety: Only lower if rank is actually >= 2
      if (rankedType.getRank() < 2) return failure();
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      rewriter.replaceOpWithNewOp<tensor::DimOp>(op, input, one);
      return success();
    }

    if (auto unrankedType = llvm::dyn_cast<UnrankedTensorType>(type)) {
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      Value rank = rewriter.create<tensor::RankOp>(loc, input);

      // Check: is rank > 1?
      Value isValid = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ugt, rank, one);

      Value dimSize = rewriter.create<tensor::DimOp>(loc, input, one);
      
      rewriter.replaceOpWithNewOp<arith::SelectOp>(op, isValid, dimSize, zero);
      return success();
    }
    return failure();
  }
};

struct UpdateOpLowering : public OpConversionPattern<gblas::UpdateOp> {
  using OpConversionPattern<gblas::UpdateOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(gblas::UpdateOp op, OpAdaptor adaptor, 
                                ConversionPatternRewriter &rewriter) const override {
    
    auto output = adaptor.getOutput(); 
    auto input = adaptor.getInput();
    Type resType = output.getType(); 

    std::string accOp = op.getAccumulateOperator() ? op.getAccumulateOperator()->str() : "plus";

    // --- THE FIX IS HERE ---
    if (op.getReplace()) {
        if (input.getType() == resType) {
            // If the encodings are identical, we can just forward the value safely
            rewriter.replaceOp(op, input);
        } else {
            // If encodings are different (e.g., COO -> CSR), we MUST use a convert op
            // so the sparsifier knows to rebuild the index structures.
            rewriter.replaceOpWithNewOp<sparse_tensor::ConvertOp>(op, resType, input);
        }
        return success();
    }
    // -----------------------

    // 1. Map the blueprint variables to your actual operation's data
    Location loc = op.getLoc();
    // Change `op.getAccumulateOperator()` to whatever gets your "max" string!
    // .value() returns the StringRef inside the optional
    StringRef accumulateOpStr = op.getAccumulateOperator().value();

    if (accOp == "plus") {
        rewriter.replaceOpWithNewOp<linalg::AddOp>(op, resType, ValueRange{input, output}, ValueRange{output});
    } else if (accOp == "max") {
        // 2. Set up the Linalg boilerplate based on your tensor's rank
        // Assuming op.getA() and op.getB() are your input/output sparse tensors
        auto inputType = llvm::cast<RankedTensorType>(op.getOperand(0).getType());
        int64_t rank = inputType.getRank();
        
        SmallVector<AffineMap> indexingMaps(3, rewriter.getMultiDimIdentityMap(rank));
        SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);
        
        SmallVector<Value, 2> inputValues = {op.getOperand(0), op.getOperand(1)};
        ValueRange inputs(inputValues);

        SmallVector<Value, 1> outputValues = {op.getOperand(1)};
        ValueRange outputs(outputValues);
        // Use the Type directly; TypeRange can be constructed from a single Type
        Type resultType = outputs[0].getType();
        TypeRange resultTypes(resultType);

        // 3. Create the linalg::GenericOp
        auto genericOp = rewriter.create<linalg::GenericOp>(
            loc,
            resultTypes,
            inputs,
            outputs,
            indexingMaps,
            iteratorTypes,
            [&](OpBuilder &b, Location loc, ValueRange args) {
                Value inVal = args[0]; 
                Value outVal = args[1];

                // 4. Create the sparse_tensor::BinaryOp (No enums, just the inputs)
                auto binaryOp = b.create<sparse_tensor::BinaryOp>(
                    loc,
                    inVal.getType(),
                    inVal,
                    outVal
                );

                // 5. OVERLAP REGION: What to do when both have a value -> arith.maximumf
                Block *overlapBlock = b.createBlock(&binaryOp.getOverlapRegion());
                overlapBlock->addArgument(inVal.getType(), loc);
                overlapBlock->addArgument(outVal.getType(), loc);
                Value maxVal = b.create<arith::MaximumFOp>(loc, overlapBlock->getArgument(0), overlapBlock->getArgument(1));
                b.create<sparse_tensor::YieldOp>(loc, maxVal);

                // 6. LEFT REGION (Identity): Left exists, right is implicit zero -> keep left
                Block *leftBlock = b.createBlock(&binaryOp.getLeftRegion());
                leftBlock->addArgument(inVal.getType(), loc);
                b.create<sparse_tensor::YieldOp>(loc, leftBlock->getArgument(0));

                // 7. RIGHT REGION (Identity): Right exists, left is implicit zero -> keep right
                Block *rightBlock = b.createBlock(&binaryOp.getRightRegion());
                rightBlock->addArgument(outVal.getType(), loc);
                b.create<sparse_tensor::YieldOp>(loc, rightBlock->getArgument(0));

                // 8. Yield the result of the BinaryOp back to the generic loop
                b.setInsertionPointAfter(binaryOp);
                b.create<linalg::YieldOp>(loc, binaryOp.getResult());
            });

        rewriter.replaceOp(op, genericOp.getResults());
    } else if (accOp == "min") {
        rewriter.replaceOpWithNewOp<linalg::MinOp>(op, resType, ValueRange{input, output}, ValueRange{output});
    } else {
        return op.emitError("unsupported accumulate operator: ") << accOp;
    }
    return success();
  }
};


struct ConvertGBLASToLinalgPass 
    : public gblas::impl::ConvertGBLASToLinalgBase<ConvertGBLASToLinalgPass> {

      void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<linalg::LinalgDialect, 
                      bufferization::BufferizationDialect,
                      scf::SCFDialect,
                      tensor::TensorDialect,
                      arith::ArithDialect,
                      sparse_tensor::SparseTensorDialect>();
      }

    void runOnOperation() override {

    ConversionTarget target(getContext());

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    target.addIllegalDialect<gblas::GBLASDialect>();
    target.addLegalDialect<linalg::LinalgDialect, tensor::TensorDialect, 
                           arith::ArithDialect, scf::SCFDialect>();

    target.addLegalDialect<bufferization::BufferizationDialect>();
    target.addLegalOp<bufferization::AllocTensorOp>();
    target.addLegalOp<bufferization::AllocTensorOp>();

    target.addLegalDialect<
      linalg::LinalgDialect,
      scf::SCFDialect,
      tensor::TensorDialect,
      arith::ArithDialect,
      bufferization::BufferizationDialect,
      sparse_tensor::SparseTensorDialect>();

    target.addIllegalOp<gblas::EWiseAddOp>();
    target.addIllegalOp<gblas::UpdateOp>();

    RewritePatternSet patterns(&getContext());
    // Register standard RewritePatterns (NO typeConverter needed)
    patterns.add<FromCooOpLowering, 
                 MxmOpLowering, 
                 MxvOpLowering, 
                 VxmOpLowering, 
                 VxvOpLowering>(&getContext());

    // Register ConversionPatterns (MUST pass typeConverter to prevent Segfault)
    patterns.add<EWiseAddLowering, NRowsLowering, NColsLowering, UpdateOpLowering>(typeConverter, &getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace gblas {
std::unique_ptr<Pass> createConvertGBLASToLinalgPass() {
  return std::make_unique<ConvertGBLASToLinalgPass>();
}
} // namespace gblas
} // namespace mlir 